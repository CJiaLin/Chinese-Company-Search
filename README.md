# Chinese-Company-Search
对公司名进行NER识别和标注后，实现在公司名单中搜索目标词

## Feature
1. 用已标注的训练样本进行NER标注训练：地名（place）、品牌名(brand)、行业词(trade)、公司名后缀词(suffix)
2. 对候选词列表进行NER标注
3. 对检索词进行NER识别标注
4. 增加是否完全包含检索词的判断
5. 字段相似度得分加权排序
6. 返回检索结果

## Data
公司名样本数据来源：https://github.com/wainshine/Company-Names-Corpus

基于jieba进行初步样本分词：https://github.com/shibing624/companynameparser

分词后存在很多错误，需要人工校对获取训练样本，标注清洗1500条数据用于训练。

**特殊标注**

1. 医院、学校、诊所、超市、研究院、设计院等标注为领域
2. 厂、店、服务社、商行、商店、中心等标注为性质
3. 如江苏银行、上海银行等，江苏、上海照常标注为地名，银行标注为领域


## NER Model Train
本项目采用bert-base-chinese进行NER微调训练

```bash
pip insatll -r requirements.txt
python train_main.py # 可直接开始训练模型
```
- data/目录下为标注好的训练样本
- 模型保存至experiments\s_model下
- config.py: 用于调整训练参数

## Chinese Company Search
`用于项目上线的API`

- blacklist_data/: 候选名单原始文件

- split_data/: 用于保存进行标注后的候选名单文件

- main.py: API启动入口

- predict.py: 调用训练好的模型对公司名进行标注

- search.py: 根据检索词在候选列表中检索相关公司

- update.py: 用于定时对每日更新的名单文件进行预处理
