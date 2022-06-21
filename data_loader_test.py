import torch
import config
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
class AnChinaDataset(Dataset):
    def __init__(self, words, labels, labels0, flags=None, word_pad_idx=0, label_pad_idx=-1):
        self.tokenizer = AutoTokenizer.from_pretrained(config.berta_model)
        self.dataset = self.preprocess(words, labels, labels0, flags)
        self.word_pad_idx = word_pad_idx
        self.label_pad_idx = label_pad_idx
    def preprocess(self, origin_sentences, origin_labels, origin_labels0, flags):

        if not flags:
            flags = [0] * len(origin_sentences)
        data = []
        sentences = []
        seg_labels = []
        pos_labels = []
        for line in origin_sentences:
            words = []
            for token in line:
                if token == '“' or token == '”' or token == '「' or token == '」':
                    token = '"'
                if token == '‘' or token == '’' or token == '『' or token == '』' \
                        or token == '（' or token == '）' or token == '(' or token == ')':
                    token = "'"
                words.append(self.tokenizer.tokenize(token))
            words = [item for token in words for item in token]
            # print(line)
            # print(words)
            sentences.append((self.tokenizer.convert_tokens_to_ids(words), line))

        for sentence, seg_label, pos_label, flag in zip(sentences, origin_labels, origin_labels0, flags):
            data.append((sentence, seg_label, pos_label, flag))
        return data

    def __getitem__(self, idx):
        """sample data to get batch"""
        word = self.dataset[idx][0]
        seg = self.dataset[idx][1]
        pos = self.dataset[idx][2]
        flag = self.dataset[idx][3]
        return [word, seg, pos, flag]

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
        sentences = [x[0][0] for x in batch]
        ori_sents = [x[0][1] for x in batch]
        seg_labels = [x[1] for x in batch]
        pos_labels = [x[2] for x in batch]
        flags = [x[3] for x in batch]
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
        batch_data = pad_sequence([torch.LongTensor(ex) for ex in sentences], \
                                  batch_first=True, padding_value=self.word_pad_idx)
        batchseg_labels = pad_sequence([torch.LongTensor(ex) for ex in seg_labels], \
                                       batch_first=True, padding_value=self.label_pad_idx)

        batchpos_labels = pad_sequence([torch.LongTensor(ex) for ex in pos_labels], \
                                       batch_first=True, padding_value=self.label_pad_idx)
        return [batch_data, batchseg_labels, batchpos_labels, ori_sents, flags]
