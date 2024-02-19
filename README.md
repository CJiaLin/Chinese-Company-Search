# Chinese-Company-Search
对公司名进行NER识别和标注后，实现在公司名单中搜索目标词

## Feature
1. 对候选公司名文本进行NER标注训练：地名（place）、品牌名(brand)、行业词(trade)、公司名后缀词(suffix)
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


## NER Model
本项目采用bert-base-chinese进行NER微调训练





