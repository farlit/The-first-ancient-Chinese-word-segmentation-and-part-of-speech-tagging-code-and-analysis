import os
os.environ["CUDA_VISIBLE_DEVICES"]="6,7"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import torch
import config
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from model import BertSegPos
from evaluate import evaluate
from data_process import load_data
from transformers import AutoModel
from torch.utils.data import DataLoader
from data_loader import AnChinaDataset
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
# from metrics import f1_score, bad_case, output_write, output2res
from transformers.optimization import get_cosine_schedule_with_warmup, AdamW
sentences,seg,pos,segpos,flag,gram_list,positions,gram_maxlen,gram2id=load_data(config.data_dir)
test_size=len(sentences)//10
train_size=len(sentences)-2*test_size

train_sentences=sentences[:train_size]
train_seg=seg[:train_size]
train_pos=pos[:train_size]
train_segpos=segpos[:train_size]
if config.use_attention:
    train_gram_list=gram_list[:train_size]
    train_positions=positions[:train_size]
    train_gram_maxlen=gram_maxlen[:train_size]
else:
    train_gram_list=None
    train_positions=None
    train_gram_maxlen=None

val_sentences=sentences[train_size:train_size+test_size]
val_seg=seg[train_size:train_size+test_size]
val_pos=pos[train_size:train_size+test_size]
val_segpos=segpos[train_size:train_size+test_size]
if config.use_attention:
    val_gram_list=gram_list[train_size:train_size+test_size]
    val_positions=positions[train_size:train_size+test_size]
    val_gram_maxlen=gram_maxlen[train_size:train_size+test_size]
else:
    val_gram_list=None
    val_positions=None
    val_gram_maxlen=None

test_sentences=sentences[train_size+test_size:]
test_seg=seg[train_size+test_size:]
test_pos=pos[train_size+test_size:]
test_segpos=segpos[train_size+test_size:]
if config.use_attention:
    test_gram_list=gram_list[train_size+test_size:]
    test_positions=positions[train_size+test_size:]
    test_gram_maxlen=gram_maxlen[train_size+test_size:]
else:
    test_gram_list=None
    test_positions=None
    test_gram_maxlen=None

train_sentences=sentences[train_size:]
train_seg=seg[train_size:]
train_pos=pos[train_size:]
train_segpos=segpos[train_size:]
if config.use_attention:
    train_gram_list=gram_list[train_size:]
    train_positions=positions[train_size:]
    train_gram_maxlen=gram_maxlen[train_size:]
else:
    train_gram_list=None
    train_positions=None
    train_gram_maxlen=None

val_sentences=sentences[:train_size]
val_seg=seg[:train_size]
val_pos=pos[:train_size]
val_segpos=segpos[:train_size]
if config.use_attention:
    val_gram_list=gram_list[:train_size]
    val_positions=positions[:train_size]
    val_gram_maxlen=gram_maxlen[:train_size]
else:
    val_gram_list=None
    val_positions=None
    val_gram_maxlen=None

train_dataset=AnChinaDataset(train_sentences,train_seg,train_pos,train_segpos,train_gram_list,train_positions,train_gram_maxlen,gram2id)
val_dataset=AnChinaDataset(val_sentences,val_seg,val_pos,val_segpos,val_gram_list,val_positions,val_gram_maxlen,gram2id)
test_dataset =AnChinaDataset(test_sentences, test_seg, test_pos,test_segpos,test_gram_list,test_positions,test_gram_maxlen,gram2id)

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=-1)
args = parser.parse_args()

torch.cuda.set_device(args.local_rank)
device = torch.device('cuda', args.local_rank)
torch.distributed.init_process_group(backend='nccl')

if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

train_loader = DataLoader(train_dataset, batch_size=config.batch_size,sampler=DistributedSampler(train_dataset),\
                          collate_fn=train_dataset.collate_fn,num_workers=os.cpu_count(),pin_memory=True)
val_loader  = DataLoader(val_dataset, batch_size=config.batch_size,sampler=DistributedSampler(val_dataset),\
                         collate_fn=val_dataset.collate_fn,num_workers=os.cpu_count(),pin_memory=True)
test_loader  = DataLoader(test_dataset, batch_size=config.batch_size,sampler=DistributedSampler(test_dataset),\
                          collate_fn=test_dataset.collate_fn,pin_memory=True)
model=BertSegPos(config, gram2id)
model.to(device)
# if config.load_before:
#     map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
#     model.load_state_dict(torch.load('sikuRoberta_model.pth', map_location=map_location))

if args.local_rank == 0:
    if config.load_before:
        model.load_state_dict(torch.load('sikuRoberta_model_crf.pth'))
    torch.distributed.barrier()   # Make sure only the first process in distributed training will download model & vocab

if config.full_fine_tuning:
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, correct_bias=False)
else:
    unfreeze_layers = ['classifier_seg', 'classifier_pos']
    for name, param in model.named_parameters():
        param.requires_grad = False
        for active in unfreeze_layers:
            if active in name:
                param.requires_grad = True
                break
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate, correct_bias=False)

train_size=len(train_dataset)
train_steps_per_epoch = train_size // config.batch_size
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=train_steps_per_epoch,
                                            num_training_steps=config.epoch_num * train_steps_per_epoch)
model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
best_val_f1 = 0.0
patience_counter = 0
for epoch in range(1, config.epoch_num + 1):
    train_losses = 0
    model.train()
    for idx, batch_samples in enumerate(tqdm(train_loader)):
        batch_data, batchseg_labels, batchpos_labels, batchsegpos_labels,\
        batchgram_list, matching_matrix, channel_ids, _, _0 = batch_samples
        # shift tensors to GPU if available
        batch_data = batch_data.to(device)
        batch_seglabels = batchseg_labels.to(device)
        batch_poslabels = batchpos_labels.to(device)
        batch_segposlabels = batchsegpos_labels.to(device)
        batch_masks = batch_data.gt(0).to(device)  # get padding mask
        batch_gramlist = batchgram_list.to(device)
        matching_matrix = matching_matrix.to(device)
        channel_ids = channel_ids.to(device)
        loss = model(batch_data, token_type_ids=None, attention_mask=batch_masks, seglabels=batch_seglabels, poslabels=batch_poslabels,\
                     segposlabels=batch_segposlabels,gram_list=batch_gramlist,matching_matrix=matching_matrix,channel_ids=channel_ids)[0]
        train_losses += float(loss.item())
        # clear previous gradients, compute gradients of all variables wrt loss
        model.zero_grad()
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.clip_grad)
        # performs updates using calculated gradients
        optimizer.step()
        scheduler.step()
        if args.local_rank==0 and (idx+1)%10==0:
            print("Epoch: {}, batch:{}, train loss: {}".format(epoch, idx+1, loss.item()))

    seg_metrics, pos_metrics=evaluate(val_loader, model)
    train_loss = train_losses / len(train_loader)
    if args.local_rank == 0:
        print("Epoch: {}, train loss: {}, \n seg_metrics: {}, pos_metrics: {}".format(epoch, train_loss, seg_metrics, pos_metrics))

        val_f1 = seg_metrics['f1']
        improve_f1 = val_f1 - best_val_f1
        if improve_f1 > 1e-5:
            best_val_f1 = val_f1
            #  选择一个进程保存
        # if args.local_rank == 0:
            torch.save(model.module.state_dict(), 'sikuRoberta_model_crf0.pth')
            print("--------Save best model!--------")
            if improve_f1 < config.patience:
                patience_counter += 1
            else:
                patience_counter = 0
        else:
            patience_counter += 1
        # Early stopping and logging best f1
        # print("Saving end!")
        # print(model.device)
        if (patience_counter >= config.patience_num and epoch > config.min_epoch_num) or epoch == config.epoch_num:
            print("Best val f1: {}".format(best_val_f1))
            break
print("Training Finished!")
