import torch
import config
import label
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from data_process import load_data
from torch.nn.utils.rnn import pad_sequence

class AnChinaDataset(Dataset):
    def __init__(self, words, labels, labels0, labels1, ngram_list, matching_positions, max_gram_maxlen, gram2id, flags=None, word_pad_idx=0, label_pad_idx=-1):
        self.tokenizer = AutoTokenizer.from_pretrained(config.berta_model)
        self.label_seg2id = label.label_seg2id
        self.id_seg2label = label.id_seg2label

        self.label_pos2id = label.label_pos2id
        self.id_pos2label = label.id_pos2label

        self.label_segpos2id = label.label_segpos2id
        self.id_segpos2label = label.id_segpos2label

        self.gram2id = gram2id
        self.dataset = self.preprocess(words, labels, labels0, labels1, ngram_list, matching_positions, max_gram_maxlen, flags)
        self.word_pad_idx = word_pad_idx
        self.label_pad_idx = label_pad_idx
        # self.gram_pad_idx = gram2id['[PAD]']

    def preprocess(self, origin_sentences, origin_labels, origin_labels0, origin_labels1, ngram_list, matching_positions, max_gram_maxlen, flags):
        """
        Maps tokens and tags to their indices and stores them in the dict data.
        examples: 
            word:['[CLS]', '浙', '商', '银', '行', '企', '业', '信', '贷', '部']
            sentence:([101, 3851, 1555, 7213, 6121, 821, 689, 928, 6587, 6956],
                        array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10]))
            label:[3, 13, 13, 13, 0, 0, 0, 0, 0]
        """
        if not flags:
            flags = [0]*len(origin_sentences)
        data = []
        sentences = []
        seg_labels = []
        pos_labels = []
        segpos_labels = []
        gram_list = []
        for line in origin_sentences:
            words = []
            for token in line:
                if token == '“' or token == '”':
                    token = '"'
                if token == '‘' or token == '’':
                    token = "'"
                words.append(self.tokenizer.tokenize(token))
            words = [item for token in words for item in token]
            # print(line)
            # print(words)
            sentences.append((self.tokenizer.convert_tokens_to_ids(words), line))
            
        for tag in origin_labels:
            label_id = [self.label_seg2id.get(t) for t in tag]
            seg_labels.append(label_id)
            
        for tag in origin_labels0:
            label_id = [self.label_pos2id.get(t) for t in tag]
            pos_labels.append(label_id)

        for tag in origin_labels1:
            label_id = [self.label_segpos2id.get(t) for t in tag]
            segpos_labels.append(label_id)

        if config.use_attention:
            for gram in ngram_list:
                gram_id = []
                for i in range(config.cat_num):
                    gram_id.append([])
                    for j in range(len(gram[i])):
                        try:
                            gram_id[i].append(self.gram2id[gram[i][j]])
                        except KeyError:
                            print(gram[i][j])
                            print(gram[i])
                            raise KeyError()
                gram_list.append(gram_id)

            for sentence, seg_label, pos_label, segpos_label, flag, gram, position, gram_len in \
                    zip(sentences, seg_labels, pos_labels, segpos_labels, flags, gram_list, matching_positions,
                        max_gram_maxlen):
                data.append((sentence, seg_label, pos_label, segpos_label, gram, position, gram_len, flag))
            return data

        else:
            for sentence, seg_label, pos_label, segpos_label, flag in \
                zip(sentences, seg_labels, pos_labels, segpos_labels, flags):
                data.append((sentence, seg_label, pos_label, segpos_label, [], [], [], flag))
            return data

    def __getitem__(self, idx):
        """sample data to get batch"""
        word = self.dataset[idx][0]
        seg_label = self.dataset[idx][1]
        pos_label = self.dataset[idx][2]
        segpos_label = self.dataset[idx][3]
        gram_list = self.dataset[idx][4]
        positions = self.dataset[idx][5]
        gram_maxlen = self.dataset[idx][6]
        flag = self.dataset[idx][7]
        return [word, seg_label, pos_label, segpos_label, gram_list, positions, gram_maxlen, flag]

    def __len__(self):
        """get dataset size"""
        return len(self.dataset)

    def collate_fn(self, batch):
        """
        process batch data, including:
            1. padding: 将每个batch的data padding到同一长度（batch中最长的data长度）
            2. aligning: 找到每个sentence sequence里面有label项，文本与label对齐
            3. tensor：转化为tensor
        """
        sentences =  [x[0][0] for x in batch]
        ori_sents = [x[0][1] for x in batch]
        seg_labels = [x[1] for x in batch]
        pos_labels = [x[2] for x in batch]
        segpos_labels = [x[3] for x in batch]
        gram_list = [x[4] for x in batch]
        match_positions = [x[5] for x in batch]
        gram_len = [x[6] for x in batch]
        flags = [x[7] for x in batch]
        # print(sentences)
        # print(len(sentences[0]))
        # print(len(sentences[1]))
        # print(ori_sents)
        # print(len(ori_sents[0]))
        # print(len(ori_sents[1]))
        # print(seg_labels)
        # print(len(seg_labels[0]))
        # print(len(seg_labels[1]))
        # print("---")
        batch_data=pad_sequence([torch.LongTensor(ex) for ex in sentences],\
                                batch_first=True,padding_value=self.word_pad_idx)
        
        batchseg_labels=pad_sequence([torch.LongTensor(ex) for ex in seg_labels],\
                                batch_first=True,padding_value=self.label_pad_idx)
            
        batchpos_labels=pad_sequence([torch.LongTensor(ex) for ex in pos_labels],\
                                batch_first=True,padding_value=self.label_pad_idx)

        batchsegpos_labels = pad_sequence([torch.LongTensor(ex) for ex in segpos_labels], \
                                       batch_first=True, padding_value=self.label_pad_idx)
        if config.use_attention:
            max_gram_len = max(max(gram_len), 1)
            batch_len = len(batch_data)
            max_seq_length = len(batch_data[0])

            batchgram_list = pad_sequence([torch.LongTensor(gram_list[i][j]) for i in range(batch_len) for j in range(config.cat_num)],\
                                      batch_first=True,padding_value=self.gram_pad_idx).reshape(batch_len, config.cat_num, -1)

            matching_matrix = np.zeros((batch_len, config.cat_num, max_seq_length, max_gram_len), dtype=np.int)
            channel_ids = []
            for i in range(batch_len):
                channel_id = []
                for j in range(config.cat_num):
                    channel_id.append(j)
                    for position in match_positions[i][j]:
                        char_p = position[0] + 1
                        word_p = position[1]
                        matching_matrix[i][j][char_p][word_p] = 1
                channel_ids.append(channel_id)
            channel_ids = torch.LongTensor(channel_ids)
            matching_matrix = torch.LongTensor(matching_matrix)
        else:
            batchgram_list = torch.LongTensor([])
            matching_matrix = torch.LongTensor([])
            channel_ids = torch.LongTensor([])

        return [batch_data, batchseg_labels, batchpos_labels, batchsegpos_labels, batchgram_list, matching_matrix, channel_ids, ori_sents, flags]
