import os
import logging
from logging import handlers
import warnings
import torch
import json
from predict import CompanyPredict
from transformers import (
  BertTokenizerFast,
)
import config
import numpy as np
import re
from collections import defaultdict
from fuzzywuzzy import fuzz

warnings.filterwarnings("ignore")
tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
model = torch.load('experiments/s_model/best_loss_model/model.bin', map_location=lambda storage, loc: storage)
# model = torch.load('experiments/s_model/last_model/model.bin')
CPredict = CompanyPredict(model, config, tokenizer)

blacklist_dir = './blacklist_data/'
split_dir = './split_data/'

class Logger(object):
    level_relations = {
        'info': logging.INFO,
        'error': logging.ERROR
    }
    def __init__(self, filename, level='debug', when='D', backCount=30, fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        if not self.logger.handlers:
            format_str = logging.Formatter(fmt)
            self.logger.setLevel(self.level_relations.get(level))
            th = handlers.TimedRotatingFileHandler(filename=filename, when=when, backupCount=backCount, encoding='utf-8')
            th.setFormatter(format_str)
            self.logger.addHandler(th)

def calculate_levenshtein_distance(text1, text2):
    m, n = len(text1), len(text2)
    dp = np.zeros((m + 1, n + 1))

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1

    return dp[m][n]

def get_result(target_text, split_file_path, item_num=10, weight = {"place": 1, "brand": 1, "trade": 1, "suffix": 1}):
    # target_text = '黄金自营有限公司'
    target_text = re.sub(r'[^\u4e00-\u9fffa-zA-Z0-9()（）"\'“”‘’]', ' ', target_text.lower())
    batch_data, batch_label_starts, tokens_ori = \
            CPredict.data_process([target_text])
    all_end, pred_tags, embedding = CPredict.predict(batch_data, batch_label_starts, tokens_ori, [target_text])

    target = {'text': target_text, 'label': all_end[0]}
    # target_embedding = embedding[0]
    results = []
    with open(split_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            l = json.loads(line.strip())
            results.append(l)

    candidates = cal_score(target, results, item_num, weight)

    return {'target': target, 'results': candidates}

def cal_score(target, results, item_num, weight):
    candidates = [] # defaultdict(list)
    target['text'] = target['text'].replace(" ", '')
    for result in results:
        text = result['text'].replace(' ', '')
        score = defaultdict(float)
        # print(item['text'], target['text'],  len(set(item['text']) & set(target['text'])))
        if len(set(text) & set(target['text'])) == 0:
            continue
        if len(set(text) & set(target['text'])) == len(target['text']):
            score['place'] = 100
            score['trade'] = 100
            score['brand'] = 100
            score['suffix'] = 100
            score['score'] = 400
            candidates.append([result['id'], text, result['label'], score, score['score']])
            continue
        
        score['score'] = 0
        for label in ['place', 'trade', 'brand', 'suffix']:
            score[label] = 0
            if target['label'][label] == [] and result['label'][label] == []:
                continue

            if target['label'][label] == [] and result['label'][label] != []:
                # score[item['text']][label] += sum([len(candi[0]) for candi in item['label'][label]])
                # score[item['text']]['score'] += score[item['text']][label]
                continue

            if target['label'][label] != [] and result['label'][label] == []:
                score[label] += 0 # sum([len(candi[0]) for candi in target['label'][label]])
            else:
                target_text = ''.join([target['text'][item[1]:item[2]] for item in target['label'][label]])
                candi = ''.join([text[item[1]:item[2]] for item in result['label'][label]])
                score[label] = fuzz.ratio(target_text, candi)# calculate_levenshtein_distance(target_text, candi)
            
            score[label] = score[label] * weight[label]

            score['score'] += score[label]

        # candidates[result['id']].append({'text': result['text'], 'label': result['label'], 'label_score': score, 'score': score['score']})

        candidates.append([result['id'], text, result['label'], score, score['score']])
        candidates = sorted(candidates, key=lambda item:item[-1], reverse=True)

    if len(candidates) < item_num:
        item_num = len(set([candidate[0] for candidate in candidates]))

    return_res = defaultdict(list)
    ids = set()
    i = 0
    while len(ids) < item_num and i < len(candidates):
        if candidates[i][0] not in ids:
            ids.add(candidates[i][0])
        return_res[candidates[i][0]].append({'text': candidates[i][1], 
                                             'label': candidates[i][2], 
                                             'label_score': candidates[i][3], 
                                             'score': candidates[i][4]})
        i += 1

    return return_res

if __name__ == "__main__":
    # split_flg, file_path = check_data()
    # if split_flg:
    #     split_blacklist(file_path)

    search_str = '农业机械化' # '哈尔滨黄金自营有限公司'
    file_path = './split_data/RECORD_LIST_SPLIT.txt'
    candidates = get_result(search_str, file_path)
    print(candidates)
