import torch
import label
import torch.nn as nn
from torchcrf import CRF
from transformers import  AutoModel
from torch.nn.utils.rnn import pad_sequence
# from transformers.models.bert.modeling_bert import *

class MultiChannelAttention(nn.Module):
    def __init__(self, ngram_size, hidden_size, cat_num):
        super(MultiChannelAttention, self).__init__()
        self.word_embedding = nn.Embedding(ngram_size, hidden_size, padding_idx=0)
        self.channel_weight = nn.Embedding(cat_num, 1)
        self.temper = hidden_size ** 0.5

    def forward(self, word_seq, hidden_state, char_word_mask_matrix, channel_ids):
        # word_seq: (batch_size, channel, word_seq_len)
        # hidden_state: (batch_size, character_seq_len, hidden_size)
        # mask_matrix: (batch_size, channel, character_seq_len, word_seq_len)

        # embedding (batch_size, channel, word_seq_len, word_embedding_dim)
        batch_size, character_seq_len, hidden_size = hidden_state.shape
        channel = char_word_mask_matrix.shape[1]
        word_seq_length = word_seq.shape[2]

        embedding = self.word_embedding(word_seq)

        tmp = embedding.permute(0, 1, 3, 2)  #(batch_size, channel, hidden_size, word_seq_len)

        tmp_hidden_state = torch.stack([hidden_state] * channel, 1)   #(batch_size, channel, character_seq_len, hidden_size)

        # u (batch_size, channel, character_seq_len, word_seq_len)
        u = torch.matmul(tmp_hidden_state, tmp) / self.temper

        # attention (batch_size, channel, character_seq_len, word_seq_len)
        tmp_word_mask_metrix = torch.clamp(char_word_mask_matrix, 0, 1)

        exp_u = torch.exp(u)
        delta_exp_u = torch.mul(exp_u, tmp_word_mask_metrix)

        sum_delta_exp_u = torch.stack([torch.sum(delta_exp_u, 3)] * delta_exp_u.shape[3], 3)
        # denominator (batch_size, channel, character_seq_len, word_seq_len) 分母

        attention = torch.div(delta_exp_u, sum_delta_exp_u + 1e-10)

        attention = attention.view(batch_size * channel, character_seq_len, word_seq_length)
        embedding = embedding.view(batch_size * channel, word_seq_length, hidden_size)

        character_attention = torch.bmm(attention, embedding)

        character_attention = character_attention.view(batch_size, channel, character_seq_len, hidden_size)

        channel_w = self.channel_weight(channel_ids)
        channel_w = nn.Softmax(dim=1)(channel_w)

        channel_w = channel_w.view(batch_size, -1, 1, 1)

        character_attention = torch.mul(character_attention, channel_w)

        character_attention = character_attention.permute(0, 2, 1, 3)
        #(batch_size, character_seq_len, channel, hidden_size)
        character_attention = character_attention.flatten(start_dim=2)

        return character_attention

class BertSegPos(nn.Module):
    def __init__(self, config, gram2id):
        super(BertSegPos, self).__init__( )
        
        self.num_seglabels = label.num_seglabels
        self.num_poslabels = label.num_poslabels
        # self.num_segposlabels = config.num_segposlabels
        self.bert = AutoModel.from_pretrained(config.berta_model,add_pooling_layer=False)
        #add_pooling_layer=False命令可以让模型没有池化层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.gram2id = gram2id
        # self.classifier_seg = nn.Linear(config.hidden_size, self.num_seglabels)
        # self.classifier_pos = nn.Linear(config.hidden_size, self.num_poslabels)
        # self.classifier_segpos = nn.Linear(config.hidden_size, self.num_segposlabels)

        if config.use_attention:
            self.multi_attention = MultiChannelAttention(len(self.gram2id), config.hidden_size, config.cat_num)
            self.classifier_seg = nn.Linear(config.hidden_size * (1 + config.cat_num), self.num_seglabels)
            self.classifier_pos = nn.Linear(config.hidden_size * (1 + config.cat_num), self.num_poslabels)
            # self.classifier = nn.Linear(config.hidden_size * (1 + self.cat_num), self.num_labels, bias=False)
        else:
            self.multi_attention = None
            self.classifier_seg = nn.Linear(config.hidden_size, self.num_seglabels)
            self.classifier_pos = nn.Linear(config.hidden_size, self.num_poslabels)

        self.crf_seg = CRF(self.num_seglabels, batch_first=True)
        self.crf_pos = CRF(self.num_poslabels, batch_first=True)
        # self.crf_segpos = CRF(config.num_segposlabels, batch_first=True)
        # self.init_weights()

    def forward(self, input_data, token_type_ids=None, attention_mask=None, seglabels=None, 
                poslabels=None, segposlabels=None, gram_list=None, matching_matrix=None, \
                channel_ids=None, position_ids=None, inputs_embeds=None, head_mask=None):
        input_ids = input_data
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)
        sequence_output = outputs[0]

        if self.multi_attention is not None:
            attention_output = self.multi_attention(gram_list, sequence_output, matching_matrix, channel_ids)
            sequence_output = torch.cat([sequence_output, attention_output], dim=2)

        # 去除[CLS]、[SEP]标签等位置，获得与label对齐的pre_label表示
        origin_sequence_output = [ layer[1:] for layer in sequence_output ]

        # 将sequence_output的pred_label维度padding到最大长度
        padded_sequence_output = pad_sequence(origin_sequence_output, batch_first=True, padding_value=0)

        # dropout pred_label的一部分feature
        padded_sequence_output = self.dropout(padded_sequence_output)

        #得到判别值
        logits_seg = self.classifier_seg(padded_sequence_output)
        logits_pos = self.classifier_pos(padded_sequence_output)
        # logits_segpos = self.classifier_segpos(padded_sequence_output)
        outputs = ((logits_seg,logits_pos),)
        # outputs = ((logits_segpos),)
        if segposlabels is not None:
            lossseg_mask = seglabels.gt(-1)
            losspos_mask = poslabels.gt(-1)
            # losssegpos_mask = segposlabels.gt(-1)

            loss_seg = self.crf_seg
            loss_pos = self.crf_pos
            # loss_segpos = self.crf_segpos
            # Only keep active parts of the loss
            if lossseg_mask is not None:
                activeseg_mask = lossseg_mask == 1
                seg_loss = loss_seg(logits_seg, seglabels, activeseg_mask) * (-1)
            else:
                seg_loss = loss_seg(logits_seg, seglabels) * (-1)

            if losspos_mask is not None:
                activepos_mask = losspos_mask == 1
                pos_loss = loss_pos(logits_pos, poslabels, activepos_mask) * (-1)
            else:
                pos_loss = loss_pos(logits_pos, poslabels) * (-1)

            # if losssegpos_mask is not None:
            #     activesegpos_mask = losssegpos_mask == 1
            #     segpos_loss = loss_segpos(logits_segpos, segposlabels, activesegpos_mask) * (-1)
            # else:
            #     segpos_loss = loss_segpos(logits_pos, segposlabels) * (-1)
            outputs = (seg_loss+pos_loss,) + outputs
        return outputs
