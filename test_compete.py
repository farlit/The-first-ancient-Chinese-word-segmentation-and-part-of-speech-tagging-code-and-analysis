import os
os.environ["CUDA_VISIBLE_DEVICES"]="4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import config
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from model import BertSegPos
from evaluate_test import evaluate
from data_process_test import load_data
from test_file import generate_file
from transformers import AutoModel
from torch.utils.data import DataLoader
from data_loader_test import AnChinaDataset
# from metrics import f1_score, bad_case, output_write, output2res
from transformers.optimization import get_cosine_schedule_with_warmup, AdamW
data_dir = './EvaHan_testb_raw.txt'
# data_dir = '/EvaHan_testb_raw.txt'
sentences,segs,poss,flag=load_data(data_dir)
# generate_file(test_sentences,test_seg,test_pos,'temp.txt',flag[train_size+test_size:])
test_dataset =AnChinaDataset(sentences,segs,poss,flag)
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device('cuda:0')
test_loader  = DataLoader(test_dataset, batch_size=config.batch_size,\
                          collate_fn=test_dataset.collate_fn,num_workers=0,pin_memory=True)
model=BertSegPos(config,None)
model.to(device)
model.load_state_dict(torch.load('sikuRoberta_model_crf0.pth', map_location="cuda:0"))
evaluate(test_loader, model, 'test')