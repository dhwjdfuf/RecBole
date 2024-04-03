from recbole.quick_start import run_recbole
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--version', required=True)

args = parser.parse_args()

run_recbole(model='SASRec', dataset='shorts', config_file_list=['our_default.yaml'], version = args.version)