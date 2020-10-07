# -*- coding: utf-8 -*-
# @Time   : 2020/8/28 14:32
# @Author : Yujie Lu
# @Email  : yujielu1998@gmail.com

# UPDATE
# @Time   : 2020/10/2
# @Author : Yujie Lu
# @Email  : yujielu1998@gmail.com
r"""
recbox.model.sequential_recommender.fpmc
################################################

Reference:
Steffen Rendle et al. "Factorizing Personalized Markov Chains for Next-Basket Recommendation." in WWW 2010.

"""
import torch
from torch import nn
from torch.nn.init import xavier_normal_
from recbox.utils import InputType
from recbox.model.loss import BPRLoss
from recbox.model.abstract_recommender import SequentialRecommender


class FPMC(SequentialRecommender):
    r"""The FPMC model is mainly used in the recommendation system to predict the possibility of unknown items arousing user interest,
     and to discharge the item recommendation list.

     Note:

            In order that the generation method we used is common to other sequential models,
            We set the size of the basket mentioned in the paper equal to 1.
            For comparison with other models, the loss function used is BPR.

    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(FPMC, self).__init__()
        # load parameters info
        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.ITEM_ID_LIST = self.ITEM_ID + config['LIST_SUFFIX']
        self.max_item_list_length = config['MAX_ITEM_LIST_LENGTH']
        self.TARGET_ITEM_ID = self.ITEM_ID
        self.ITEM_LIST_LEN = config['ITEM_LIST_LENGTH_FIELD']

        self.NEG_ITEM_ID = config['NEG_PREFIX'] + self.ITEM_ID
        self.user_count = dataset.user_num
        self.item_count = dataset.item_num

        self.embedding_size = config['embedding_size']
        # user embedding matrix
        self.UI_emb = nn.Embedding(self.user_count, self.embedding_size)
        # label embedding matrix
        self.IU_emb = nn.Embedding(self.item_count, self.embedding_size)
        # last click item embedding matrix
        self.LI_emb = nn.Embedding(self.item_count,
                                   self.embedding_size,
                                   padding_idx=0)
        # label embedding matrix
        self.IL_emb = nn.Embedding(self.item_count, self.embedding_size)
        # define loss
        self.loss = BPRLoss()

        # weight initialization
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)

    def forward(self, interaction):
        item_id_list = interaction[self.ITEM_ID_LIST]
        item_list_len = interaction[self.ITEM_LIST_LEN]
        item_last_click_index = item_list_len - 1
        item_last_click = torch.gather(
            item_id_list, dim=1, index=item_last_click_index.unsqueeze(1))
        item_list_emb = self.LI_emb(item_last_click)  # [b,1,emb]

        user_emb = self.UI_emb(interaction[self.USER_ID])
        user_emb = torch.unsqueeze(user_emb, dim=1)  # [b,1,emb]

        pos_iu = self.IU_emb(interaction[self.TARGET_ITEM_ID])
        pos_iu = torch.unsqueeze(pos_iu, dim=1)  # [b,1,emb]

        pos_il = self.IL_emb(interaction[self.TARGET_ITEM_ID])
        pos_il = torch.unsqueeze(pos_il, dim=1)  # [b,1,emb]

        pos_score = self.pmfc(user_emb, pos_iu, pos_il, item_list_emb)

        pos_score = torch.squeeze(pos_score)
        return pos_score

    def pmfc(self, Vui, Viu, Vil, Vli):
        r"""This is the core part of the FPMC model,can be expressed by a combination of a MF and a FMC model.


        Args:
            Vui(torch.FloatTensor): The embedding tensor of a batch of user, shape of [batch_size, 1, embedding_size]
            Viu(torch.FloatTensor): The embedding matrix of pos item or neg item, shape of [batch_size, n, embedding_size]
            Vil(torch.FloatTensor): The embedding matrix of pos item or neg item, shape of [batch_size, n, embedding_size]
            Vli(torch.FloatTensor): The embedding matrix of last click item, shape of [batch_size, 1, embedding_size]

        Returns:
            torch.Tensor:score, shape of [batch_size, 1]
        """
        #     MF
        mf = torch.matmul(Vui, Viu.permute(0, 2, 1))
        mf = torch.squeeze(mf, dim=1)  #[B,1]

        #     FMC
        fmc = torch.matmul(Vil, Vli.permute(0, 2, 1))
        fmc = torch.squeeze(fmc, dim=1)  #[B,1]
        x = mf + fmc
        return x

    def calculate_loss(self, interaction):
        user_emb = self.UI_emb(interaction[self.USER_ID])
        user_emb = torch.unsqueeze(user_emb, dim=1)  # [b,1,emb]

        item_id_list = interaction[self.ITEM_ID_LIST]
        item_list_len = interaction[self.ITEM_LIST_LEN]
        item_last_click_index = item_list_len - 1
        item_last_click = torch.gather(
            item_id_list, dim=1, index=item_last_click_index.unsqueeze(1))
        item_list_emb = self.LI_emb(item_last_click)  # [b,1,emb]

        neg_item = interaction[self.NEG_ITEM_ID]
        neg_iu = self.IU_emb(neg_item)
        neg_iu = torch.unsqueeze(neg_iu, dim=1)  # [b,1,emb]

        neg_il = self.IL_emb(neg_item)
        neg_il = torch.unsqueeze(neg_il, dim=1)  # [b,1,emb]

        neg_score = self.pmfc(user_emb, neg_iu, neg_il, item_list_emb)
        pos_score = self.forward(interaction)
        neg_score = torch.squeeze(neg_score)
        loss = self.loss(pos_score, neg_score)
        return loss

    def predict(self, interaction):
        score = self.forward(interaction)
        return score

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_emb = self.UI_emb(user)
        all_iu_emb = self.IU_emb.weight
        mf = torch.matmul(user_emb, all_iu_emb.transpose(0, 1))
        all_il_emb = self.IL_emb.weight
        item_list = interaction[self.ITEM_ID_LIST]
        item_list_len = interaction[self.ITEM_LIST_LEN]
        item_last_click_index = item_list_len - 1
        item_last_click = torch.gather(
            item_list, dim=1, index=item_last_click_index.unsqueeze(1))
        item_list_emb = self.LI_emb(item_last_click)  # [b,1,emb]
        fmc = torch.matmul(item_list_emb, all_il_emb.transpose(0, 1))
        fmc = torch.squeeze(fmc, dim=1)
        score = mf + fmc
        return score
