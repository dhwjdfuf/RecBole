# -*- coding: utf-8 -*-
# @Time    : 2020/9/18 11:33
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn

"""
SASRec
################################################

Reference:
    Wang-Cheng Kang et al. "Self-Attentive Sequential Recommendation." in ICDM 2018.

Reference:
    https://github.com/kang205/SASRec

"""

import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss
import pandas as pd
import numpy as np
import json

class SASRec(SequentialRecommender):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, config, dataset):
        super(SASRec, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]  # same as embedding_size
        self.inner_size = config[
            "inner_size"
        ]  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]

        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]


        self.item_embedding = nn.Embedding( 
            self.n_items, self.hidden_size, padding_idx=0 
        )
        
        self.token_map = json.load(open('map.json', 'r')) # 이걸로 해야할 일은, ['user_id', 'item_id']
        self.df= pd.read_csv('dataset/shorts/ryeol_filby.csv')
        self.df['userID'] = self.df['userID'].apply(lambda x: self.token_map['user_id'][str(x)]) 
        self.df['itemID'] = self.df['itemID'].apply(lambda x: self.token_map['item_id'][str(x)]) # 이거다. user도 할 수 있으면 좋음. 

        self.itemID = self.df['itemID'].to_numpy()




        #self.negative_pos_emb = torch.nn.Embedding(100, self.hidden_size) # enhancing L_p
        self.negative_pos_emb = torch.nn.Embedding(100, 1) # enhancing L_p


        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]

    def calculate_loss(self, interaction): # loss 여기. 
        item_seq = interaction[self.ITEM_SEQ] # (1024, 200)
        item_seq_len = interaction[self.ITEM_SEQ_LEN] # (1024, ) . valid len만 모아둔 것. 
        seq_output = self.forward(item_seq, item_seq_len) # (1024, 64). 아 이거 sliding window 방식이 아니고, 마지막의 embedding만 뽑아오는 꼴. 
        pos_items = interaction[self.POS_ITEM_ID] #(1024, )


        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE' . 여기임. 여기서 contrastive loss를 return해야됨. 
            test_item_emb = self.item_embedding.weight # (M, 64)
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1)) #(1024, M)
            loss = self.loss_fct(logits, pos_items)

            # user_id, torch.Size([1024]), cuda, torch.int64 
            # item_id, torch.Size([1024]), cuda, torch.int64
            # timestamp, torch.Size([1024]), cuda, torch.float32
            # item_length, torch.Size([1024]), cuda, torch.int64

            # item_id_list, torch.Size([1024, 200]), cuda, torch.int64  input session
            # timestamp_list, torch.Size([1024, 200]), cuda, torch.float32  


            #----------------------------------------------Lp----------------------------------------------------begin
            batch_size, _ = item_seq.shape
            SASs = seq_output # SASs : (1024, 64) 
            GT_embs = self.item_embedding(pos_items) # (1024, 64)
            dev = item_seq.device
            item_seq_cpu = item_seq.cpu().numpy() # (1024, 200)
            

            timestamp = interaction['timestamp_list'].cpu().numpy().astype(int)  #(1024, 200)
            negF = np.zeros((batch_size, 100), dtype= int) # 이거 hyperparameter tuning은 해야해. 

            item_seq_len_cpu = item_seq_len.cpu().numpy() # 

            start_indices = timestamp[:,0] + 1 # (1024, )
            end_indices = timestamp[np.arange(batch_size),item_seq_len_cpu-1] # (1024, ) 

            for i in range(batch_size): 
                line = self.itemID[start_indices[i]:end_indices[i]]  
                to_delete = item_seq_cpu[i]
                filtered_line = line[np.in1d(line, to_delete, invert=True)]
                valid_len = min(100, len(filtered_line))
                if valid_len == 0 : continue
                negF[i][-valid_len:] = filtered_line[-valid_len:]


            negF=torch.from_numpy(negF)

            # SASs : (1024, 64)
            negFembs = self.item_embedding(negF.to(dev)).to(dev) # (1024, 100, 64)   이건 어차피 right pushed. right pushed는 상관 없다. 

            
            sim = torch.nn.CosineSimilarity(dim=-1, eps=1e-6) 
            numerator = torch.exp(sim(SASs, GT_embs) * 2 )  # (1024, ) 
            
            repeatedSASs = SASs.unsqueeze(1).repeat(1,100,1) # (1024,100,64)

            denom_sim = torch.exp(sim(repeatedSASs,negFembs)  * 2) # (1024, 100)   
            
            denominator = torch.sum(denom_sim, dim=-1) 
            Lp_loss = -torch.log( numerator / denominator) # (1024 ,)

            Lp = torch.mean(Lp_loss)

            # print(loss) #10.71
            # print(Lp) #4.6

            loss += Lp * 0.1
            #----------------------------------------------Lp----------------------------------------------------end
            
            
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores
