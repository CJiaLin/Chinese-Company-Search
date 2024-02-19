import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset


class NERDataset(Dataset):
    def __init__(self, words, labels, config, tokenizer, word_pad_idx=0, label_pad_idx=-1, mode='train'):
        self.tokenizer = tokenizer
        # BertTokenizer.from_pretrained(config.bert_model, do_lower_case=True)
        self.label2id = config.label2id
        self.id2label = {_id: _label for _label, _id in list(config.label2id.items())}
        if mode == 'train':
            self.dataset = self.preprocess(words, labels)
        else:
            self.dataset = self.preprocess_test(words, labels)
        self.word_pad_idx = word_pad_idx
        self.label_pad_idx = label_pad_idx
        self.device = config.device
        
    def preprocess(self, origin_sentences, origin_labels):
        """
        Maps tokens and tags to their indices and stores them in the dict data.
        examples: 
            word:['[CLS]', '浙', '商', '银', '行', '企', '业', '信', '贷', '部']
            sentence:([101, 3851, 1555, 7213, 6121, 821, 689, 928, 6587, 6956],
                        array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10]))
            label:[3, 13, 13, 13, 0, 0, 0, 0, 0]
        """
        data = []
        sentences = []
        labels = []
        words_all = []
        for line in tqdm(origin_sentences):
            # print(line)
            # line  ['浙', '商', '银', '行', '企', '业', '信', '贷', '部']
            # replace each token by its index
            # we can not use encode_plus because our sentences are aligned to labels in list type
            words = ['[CLS]'] + line
            word_lens = []
            
#             print(line)
            for token in line:
#                 print(token)
                word_lens.append(1) # len(token))      
            
            # 变成单个字的列表，开头加上[CLS]
#             words = ['[CLS]'] + [item for token in words for item in token]
#             print(words)
            token_start_idxs = 1 + np.cumsum([0] + word_lens[:-1])
            sentences.append((self.tokenizer.convert_tokens_to_ids(words), token_start_idxs))
            words_all.append(words)

            # label_id = [self.label2id.get(t) for t in tag]
            # print(line,words,tag,label_id)


        for tag in origin_labels:
            label_id = [self.label2id.get(t) for t in tag]
            labels.append(label_id)
        for sentence, label, words, oss, oll in zip(sentences, labels, words_all,origin_sentences,origin_labels):
#             if label[0]!=-1:
            data.append((sentence, label))
#             print(len(sentence[0]),len(sentence[1]),len(label),len(words),len(oss),len(oll))
#             print(sentence,label,words,oss,oll)
#             print('_______________')
        return data

    def preprocess_test(self, origin_sentences, origin_labels):
        """
        Maps tokens and tags to their indices and stores them in the dict data.
        examples: 
            word:['[CLS]', '浙', '商', '银', '行', '企', '业', '信', '贷', '部']
            sentence:([101, 3851, 1555, 7213, 6121, 821, 689, 928, 6587, 6956],
                        array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10]))
            label:[3, 13, 13, 13, 0, 0, 0, 0, 0]
        """
        data = []
        sentences = []
        labels = []
        words_all = []
        for line in origin_sentences:
#             print(line)
            # line  ['浙', '商', '银', '行', '企', '业', '信', '贷', '部']
            # replace each token by its index
            # we can not use encode_plus because our sentences are aligned to labels in list type
            words = []
            word_lens = []
            
            bad_tokens = [' ','\n',' ',' ','​','  ',' �','   ',' 　 ',' � ',' ﻿ ',' 	 ',' 	 ',' � ','   ',' ﻿ ',' 　 ',' 　 ',' 　 ',' 　 ',' 　 ',' ‍ ',' 　 ','   ','   ',' 　 ',' ﻿ ','   ']
#             print(line)
            for token in line:
                token = token.lower()
#                 print(token)
                if token in bad_tokens:
                    token = '.'
                    words.append(self.tokenizer.tokenize(token))
                    word_lens.append(len(token))
                    continue
                
                token = token.replace("“",'"').replace("”",'"')
                if self.tokenizer.tokenize(token):
                    words.append(self.tokenizer.tokenize(token))
                    word_lens.append(len(token))
                else:
                    print('bad token：【',token,'】')
                    bad_tokens.append(token)
                    token = '.'
                    words.append(self.tokenizer.tokenize(token))
                    word_lens.append(len(token))
            
            
            # 变成单个字的列表，开头加上[CLS]
            words = ['[CLS]'] + [item for token in words for item in token]
#             print(words)
            token_start_idxs = 1 + np.cumsum([0] + word_lens[:-1])
            sentences.append((self.tokenizer.convert_tokens_to_ids(words), token_start_idxs))
            words_all.append(words)
        for tag in origin_labels:
            label_id = [self.label2id.get(t) for t in tag]
            labels.append(label_id)
        for sentence, label, words, oss, oll in zip(sentences, labels, words_all,origin_sentences,origin_labels):
#             if label[0]!=-1:
            data.append((sentence, label))
#             print(len(sentence[0]),len(sentence[1]),len(label),len(words),len(oss),len(oll))
#             print(sentence,label,words,oss,oll)
#             print('_______________')
        return data

    def __getitem__(self, idx):
        """sample data to get batch"""
        word = self.dataset[idx][0]
        label = self.dataset[idx][1]
        return [word, label]

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
        sentences = [x[0] for x in batch]
        labels = [x[1] for x in batch]

        # batch length
        batch_len = len(sentences)

        # compute length of longest sentence in batch
        max_len = max([len(s[0]) for s in sentences])
        max_label_len = 0

        # padding data 初始化
        batch_data = self.word_pad_idx * np.ones((batch_len, max_len))
        batch_label_starts = []

        # padding and aligning
        for j in range(batch_len):
            cur_len = len(sentences[j][0])
            batch_data[j][:cur_len] = sentences[j][0]
            # 找到有标签的数据的index（[CLS]不算）
            label_start_idx = sentences[j][-1]
            label_starts = np.zeros(max_len)
            label_starts[[idx for idx in label_start_idx if idx < max_len]] = 1
            batch_label_starts.append(label_starts)
            max_label_len = max(int(sum(label_starts)), max_label_len)

        # padding label
        batch_labels = self.label_pad_idx * np.ones((batch_len, max_label_len))
#         print(max_label_len)
        for j in range(batch_len):
            cur_tags_len = len(labels[j])
#             print(cur_tags_len)
            try:
                if None in labels[j]:
                    print(sentences[j], labels[j])
                batch_labels[j][:cur_tags_len] = labels[j]
            except:
                print(batch_labels[j][:cur_tags_len], labels[j])
                print(sentences, labels)


        # convert data to torch LongTensors
        batch_data = torch.tensor(batch_data, dtype=torch.long)
        batch_label_starts = torch.tensor(batch_label_starts, dtype=torch.long)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)
        # if batch_labels.min() < -1:
        #     print(sentences, labels)
#         print(batch_data.shape, batch_labels.shape)
#         print(batch_data, batch_labels)
        # shift tensors to GPU if available
        batch_data, batch_label_starts = batch_data.to(self.device), batch_label_starts.to(self.device)
        batch_labels = batch_labels.to(self.device)
        return [batch_data, batch_label_starts, batch_labels]

