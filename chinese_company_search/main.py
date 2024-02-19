from fastapi import FastAPI
from typing import Optional
from search import *
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
from update import update_split
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
'''
BackgroundScheduler 不会阻塞主进程，需要主程序有内容在运行，如果主程序结束了，后台也停止，不会挂在后台继续运行
BlockingScheduler 阻塞主进程，主进程停止，等定时任务结束再继续运行
'''

scheduler = BackgroundScheduler()

app = FastAPI()

class Params(BaseModel):
    search_str: str
    item_num: Optional[int] = 10
    place: Optional[int] = 1
    brand: Optional[int] = 2
    trade: Optional[int] = 1.5
    suffix: Optional[int] = 1
    

@app.on_event("startup")
def startup_event():
    scheduler.add_job(update_split, 'cron', hour=6, id="update_blacklist_split")
    scheduler.start()

@app.get("/")
def read_root():
    return {'hello': 'world'}

@app.post("/search/")
def read_time(params: Params):
    file_path = './split_data/RECORD_LIST_SPLIT.txt'
    params = jsonable_encoder(params)
    weight = {}
    weight['place'] = params['place']
    weight['brand'] = params['brand']
    weight['trade'] = params['trade']
    weight['suffix'] = params['suffix']
    candidates = get_result(params['search_str'], file_path, params['item_num'], weight)

    # print(candidates[:10])
    return {'msg': '查询成功', 'code': 200, 'results': candidates}

# if __name__ == '__main__':
#     update_split()

#     search_str = '百得（苏州）电动工具有限公司'
#     file_path = './split_data/RECORD_LIST_SPLIT.txt'
#     candidates = get_result(search_str, file_path)
#     print(candidates[:10])

