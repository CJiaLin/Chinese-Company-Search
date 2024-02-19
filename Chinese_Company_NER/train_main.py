from utils import util
import torch
import config
import logging
import numpy as np
# from data_process import Processor
from utils.data_loader import NERDataset
from train_file.model import BertNER
from train_file.train import train, evaluate
from transformers import (
  BertTokenizerFast,
)
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers.optimization import get_cosine_schedule_with_warmup, AdamW
import re
import warnings
import os
warnings.filterwarnings('ignore')
import pickle
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

filename = config.model_dir
batch_size=config.batch_size
device = config.device
if not os.path.exists(filename):               #判断文件夹是否存在
    os.makedirs(filename)                       #新建文件夹
"""train the model"""
# set the logger
util.set_logger(config.log_dir)
logging.info("device: {}".format(device))
logging.info("batch_size: {}".format(batch_size))
tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')


import json
class Processor:
    def __init__(self, config):
        self.data_dir = config.data_dir
        self.config = config

    def process(self):
        """
        process train and test data
        """
        for file_name in self.config.files:
#             try:
            self.preprocess(file_name)
#             except:
#                 continue

    def preprocess(self, mode):
        """
        params:
            words：将json文件每一行中的文本分离出来，存储为words列表
            labels：标记文本对应的标签，存储为labels
        examples:
            words示例：['生', '生', '不', '息', 'C', 'S', 'O', 'L']
            labels示例：['O', 'O', 'O', 'O', 'B-game', 'I-game', 'I-game', 'I-game']
        """
        input_dir = self.data_dir + str(mode) + '.json'
        output_dir = self.data_dir + str(mode) + '.npz'
#         if os.path.exists(output_dir) is True:
#             return
        word_list = []
        label_list = []
        with open(input_dir, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                line = line.replace('\\xa0','')
                # loads()：用于处理内存中的json对象，strip去除可能存在的空格
                json_line = json.loads(line.strip())
                text = json_line['text']
                text = text.lower().replace("“",'"').replace("”",'"')
                text = re.sub(r'[^\u4e00-\u9fffa-zA-Z0-9()（）"\'“”‘’]', '', text)
                words = tokenizer.tokenize(text)
                   
                # 如果没有label，则返回None
                label_entities = json_line.get('label', None)
                labels = ['O'] * len(words)
                flag = 0
                for key, value in label_entities.items():
                    if key == 'symbol':
                        continue
                    for item in value:
                        sub_name = item[0]
                        start_index = item[1]
                        end_index = item[2]
                        # if ''.join(words[item[1]:item[2]]) == sub_name:
        #               print(sub_name, ''.join(words[start_index:end_index + 1]))
                        flag = 1
                        # if len(sub_name) == 1:
                        #     labels[start_index] = 'S-' + key
                        # else:
                        labels[start_index] = 'B-' + key
                        labels[start_index + 1:end_index] = ['I-' + key] * (len(sub_name) - 1)
                if len(words) < 510 and flag == 1:
        #             print(words,labels)
                    word_list.append(words)
                    label_list.append(labels)

            # 保存成二进制文件
#         print(word_list,label_list)
        np.savez(output_dir, words=np.asanyarray(word_list, dtype=object), \
                 labels=np.asanyarray(label_list, dtype=object), dtype=object)
        logging.info("--------{} data process DONE!--------".format(mode))

def dev_split(dataset_dir):
    """split dev set"""
    data = np.load(dataset_dir, allow_pickle=True)
    words = data["words"]
    labels = data["labels"]
    x_train, x_dev, y_train, y_dev = train_test_split(words, labels, test_size=config.dev_split_size, random_state=0)
    return x_train, x_dev, y_train, y_dev


def load_dev(mode):
    if mode == 'train':
        # 分离出验证集
        word_train, word_dev, label_train, label_dev = dev_split(config.train_dir)
    elif mode == 'test':
        train_data = np.load(config.train_dir, allow_pickle=True)
        dev_data = np.load(config.test_dir, allow_pickle=True)
        word_train = train_data["words"]
        label_train = train_data["labels"]
        word_dev = dev_data["words"]
        label_dev = dev_data["labels"]
    elif mode == 'predict':
        train_data = np.load(config.train_dir, allow_pickle=True)
        predict_data = np.load(config.predict_dir, allow_pickle=True)
        word_train = train_data["words"]
        label_train = train_data["labels"]
        word_dev = predict_data["words"]
        label_dev = predict_data["labels"]
    else:
        word_train = None
        label_train = None
        word_dev = None
        label_dev = None
    return word_train, word_dev, label_train, label_dev


processor = Processor(config)
processor.process()
logging.info("--------Process Done!--------")


# 分离出验证集
word_train, word_dev, label_train, label_dev = load_dev('train')
logging.info("--------load_dev !--------")
# build dataset
train_dataset = NERDataset(word_train, label_train, config, tokenizer)
dev_dataset = NERDataset(word_dev, label_dev, config, tokenizer)
with open(filename+'train_dataset.pkl', 'wb') as f:
    pickle.dump(train_dataset, f)
with open(filename+'dev_dataset.pkl', 'wb') as f:
    pickle.dump(dev_dataset, f)
logging.info("--------Dataset Build!--------")
            
for i in range(len(label_train)):
    if len(word_train[i]) != len(label_train[i]):
        print(word_train[i], label_train[i])

# get dataset size
train_size = len(train_dataset)
print(train_size)
# build data_loader
train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                            shuffle=True, collate_fn=train_dataset.collate_fn)
dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size,
                        shuffle=True, collate_fn=dev_dataset.collate_fn)
with open(filename+'train_loader'+str(batch_size)+'.pkl', 'wb') as f:
        pickle.dump(train_loader, f)
with open(filename+'dev_loader'+str(batch_size)+'.pkl', 'wb') as f:
    pickle.dump(dev_loader, f)
logging.info("--------Get Dataloader!--------")


check_model_dir = ''
# Prepare model
if check_model_dir != '':
    model = BertNER.from_pretrained(check_model_dir)
    logging.info("--------Load model from {}--------".format(check_model_dir))
else:
    model = BertNER.from_pretrained('bert-base-chinese',num_labels=len(config.label2id))
    logging.info("--------Create model from {}--------".format('bert-base-chinese'))
model.to(device)
# train_loader.to(device)
# dev_loader.to(device)
# Prepare optimizer


if config.full_fine_tuning:
    # model.named_parameters(): [bert, bilstm, classifier, crf]
    bert_optimizer = list(model.bert.named_parameters())
    # lstm_optimizer = list(model.bilstm.named_parameters())
    classifier_optimizer = list(model.classifier.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': config.weight_decay},
        {'params': [p for n, p in bert_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0},
        # {'params': [p for n, p in lstm_optimizer if not any(nd in n for nd in no_decay)],
            # 'lr': config.learning_rate * 20, 'weight_decay': config.weight_decay},
        # {'params': [p for n, p in lstm_optimizer if any(nd in n for nd in no_decay)],
            # 'lr': config.learning_rate * 20, 'weight_decay': 0.0},
        {'params': [p for n, p in classifier_optimizer if not any(nd in n for nd in no_decay)],
            'lr': config.learning_rate * 20, 'weight_decay': config.weight_decay},
        {'params': [p for n, p in classifier_optimizer if any(nd in n for nd in no_decay)],
            'lr': config.learning_rate * 20, 'weight_decay': 0.0},
        {'params': model.crf.parameters(), 'lr': config.learning_rate * 20}
    ]
# only fine-tune the head classifier
else:
    param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer]}]
optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, correct_bias=False)

train_steps_per_epoch = train_size // config.batch_size
scheduler = get_cosine_schedule_with_warmup(optimizer,
                                            num_warmup_steps=(config.epoch_num // 10) * train_steps_per_epoch,
                                            num_training_steps=config.epoch_num * train_steps_per_epoch)
# Train the model
logging.info("--------Start Training!--------")
train(train_loader, dev_loader, model, optimizer, scheduler, config.model_dir)
torch.save(model, config.model_dir + 'last_model/' + 'model.bin')