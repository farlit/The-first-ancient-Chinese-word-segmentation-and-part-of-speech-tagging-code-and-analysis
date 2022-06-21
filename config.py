import os
import torch

berta_model = "sikuRoberta_model"
model_dir = os.getcwd() + '/experiments/'
data_dir = os.getcwd() + '/zuozhuan_train_utf8.txt'
train_dir = data_dir + 'training.npz'
test_dir = data_dir + 'test.npz'
files = ['training', 'test']
vocab_path = data_dir + 'vocab.npz'
exp_dir = os.getcwd() + '/experiments/'
log_dir = exp_dir + 'train.log'
case_dir = os.getcwd() + '/case/bad_case.txt'
output_dir = data_dir + 'output.txt'
res_dir = data_dir + 'res.txt'
test_ans = data_dir + 'test.txt'
pretrain_datadir = 'merge.txt'
print(os.getcwd())
max_vocab_size = 1000000
max_len = 512   #前后结尾需要cls和sep
sep_word = '@'  # 拆分句子的文本分隔符
sep_label = 'S'  # 拆分句子的标签分隔符

# 训练集、验证集划分比例
dev_split_size = 0.1

# 是否加载训练好的Seg模型
load_before = True

# 是否对整个BERT进行fine tuning
full_fine_tuning = True

# hyper-parameter
learning_rate = 1e-5
weight_decay = 0.01
clip_grad = 5

batch_size = 16
epoch_num = 20
min_epoch_num = 5
patience = 0.0002
patience_num = 4

gpu = '0'
device = torch.device("cuda")

use_attention = False
cat_num = 10
ngram_length = 3
cat_type = 'freq'
ngram_type = 'pmi'
ngram_threshold = 2

attention_probs_dropout_prob=0.1
directionality='bidi'
gradient_checkpointing=False
hidden_act='gelu'
hidden_dropout_prob=0.1
hidden_size=768
initializer_range=0.02
intermediate_size=3072
layer_norm_eps =1e-12
max_position_embeddings=512
model_type='bert'
num_attention_heads=12
num_hidden_layers=12
pad_token_id=0
pooler_fc_size=768
pooler_num_attention_heads=12
pooler_num_fc_layers=3
pooler_size_per_head=128
pooler_type='first_token_transform'
type_vocab_size=2
vocab_size=29791
