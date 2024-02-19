from apscheduler.schedulers.background import BackgroundScheduler, BlockingScheduler
import time 
from zhconv import convert
from search import Logger, CPredict
from tqdm import tqdm 
import re
import os
import json
from datetime import datetime, date, timedelta
# from predict import CompanyPredict

# scheduler = BackgroundScheduler()
blacklist_dir = './blacklist_data/'
split_dir = './split_data/'
today = date.today()
file_date = today + timedelta(days = -2)
file_date = file_date.strftime('%Y%m%d')

def is_chinese(string):
    for ch in string:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False

''' 检查.flg和.dat文件是否对应，找到最新的，检查通过的.dat文件
    返回： 是否需要重新对黑名单数据分词，已分词文件路径/黑名单数据路径
'''
def check_data(target_file_name = "HMD_SA29_RECORD_LIST_", split_file_name = "HMD_SA29_RECORD_LIST_SPLIT_"):
    file_dirs = os.listdir(blacklist_dir)
    del_dir = []
    for file_dir in file_dirs:
        if not file_dir.isnumeric():
            del_dir.append(file_dir)
    file_dirs = list(set(file_dirs) - set(del_dir))
    
    file_dirs.sort(reverse=True)
    for target_dir in file_dirs:
        dir_path = blacklist_dir + target_dir  # 日期目录
        Logger('./log/blacklist.log', level='info').logger.info('processing of dir: ' + dir_path)

        split_file = split_dir + target_dir + '/' + split_file_name + target_dir + '.txt'
        if os.path.exists(split_file):
            return False, split_file

        flg_file = dir_path + '/' + target_file_name + target_dir + '.flg'  # .flg文件路径
        if not os.path.exists(flg_file):
            Logger('./log/blacklist.log', level='error').logger.info(flg_file + ' not exist. File process failed')
            continue

        f = open(flg_file, 'r', encoding='utf-8')
        line = f.readline().strip().split(' ')
        file_name = line[0]
        file_size = line[1]
        file_lines = line[2]

        file_path = dir_path + '/' + file_name  # .dat文件路径
        if not os.path.exists(file_path):
            Logger('./log/blacklist.log', level='error').logger.info(file_path + ' not exist')
            continue

        if str(os.path.getsize(file_path)) != file_size:
            Logger('./log/blacklist.log', level='error').logger.info(file_path + ' file size is ' + str(os.path.getsize(file_path)) + ' != ' + file_size)
            continue
        
        with open(file_path) as f:
            for count, _ in enumerate(f, 1):
                pass
        if str(count) != file_lines:
            Logger('./log/blacklist.log', level='error').logger.info(file_path + ' file lines is ' + str(count) + ' != ' + file_lines)
            continue
        
        return True, file_path

    Logger('./log/blacklist.log', level='error').logger.info('no valid files')

    return False, False

def split_blacklist(file_path, split_file_name = "HMD_SA29_RECORD_LIST_SPLIT_"):
    Logger('./log/blacklist.log', level='info').logger.info('processing of file: ' + file_path)

    company_ids = []
    company_names = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            l = line.strip().split('@!@')
            l_id = l[0]
            l_name = convert(l[1].strip().replace("“",'"').replace("”",'"'), 'zh-cn')
            if not is_chinese(l_name):
                continue
            l_name = re.sub(r'[^\u4e00-\u9fffa-zA-Z0-9()（）]', ' ', l_name.lower())
            company_ids.append(l_id)
            company_names.append(l_name)

    Logger('./log/blacklist.log', level='info').logger.info('processing of company_name... ...')
    results = []
    part = 128
    iter_num = int(len(company_names) / part) + 1
    embeddings = []
    for i in tqdm(range(iter_num)):
        start = i * part 
        if i == iter_num -1:
            end = len(company_names)
        else:
            end = (i + 1) * part
        batch_data, batch_label_starts, sentences_ori = \
            CPredict.data_process(company_names[start:end])
        all_end, pred_tags, _ = CPredict.predict(batch_data, batch_label_starts, sentences_ori, company_names[start:end])
        results.extend(all_end)
        # embeddings.extend(embedding)

    results_ori = []
    for i in range(len(results)):
        results_ori.append({'id': company_ids[i], 'text': company_names[i], 'label': results[i]})
    
    if not os.path.exists(split_dir + file_path[-12:-4]):
        os.mkdir(split_dir + file_path[-12:-4])

    split_file_path = split_dir + file_path[-12:-4] + '/' + split_file_name + file_path[-12:-4] + '.txt'

    with open(split_file_path, 'w', encoding='utf-8') as f:
        for res in results_ori:
            f.write(json.dumps(res, ensure_ascii=False) + '\n')

    with open(split_dir + split_file_name[:-1] + '.txt', 'w', encoding='utf-8') as f:
        for res in results_ori:
            f.write(json.dumps(res, ensure_ascii=False) + '\n')

    Logger('./log/blacklist.log', level='info').logger.info(split_file_path + ' processed.')

def update_split():
    Logger('./log/blacklist.log', level='info').logger.info('split file update start.')
    split_flg, file_path = check_data()
    if split_flg:
        split_blacklist(file_path)
    else:
        Logger('./log/blacklist.log', level='info').logger.info('use of historical files: ' + file_path)


if __name__ == '__main__':
    split_flg, file_path = check_data()
    if split_flg:
        split_blacklist(file_path)
#     scheduler.add_job(split_blacklist, 'cron', second="*/5", hour='*', id="update_blacklist_split")
#     print('start')
#     scheduler.start()
#     while (True):
#         print('------')
#         time.sleep(5)