# -*-coding:utf-8 -*-

'''
@File       : preprocess.py
@Author     : HW Shen
@Date       : 2020/9/25
@Desc       :
'''

import pandas as pd
import json

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_BASE = BASE_DIR + "/data/"
print(DATA_BASE)


def data_preprocess(corpus_file_name):
    """ 数据预处理 """

    print("===================Start Preprocess======================")
    df = pd.read_csv(DATA_BASE + corpus_file_name + ".csv")  # 读取源数据，将数据解析为时间格式
    # df["小时"] = df["time"].map(lambda x: int(x.strftime("%H")))  # 提取小时
    df = df.drop_duplicates()  # 去重
    print("Remove duplicate items completed! ")
    df = df.dropna(subset=["内容"])  # 删除 “评论内容” 空值行
    # df = df.dropna(subset=["gender"])  # 删除 “性别” 空值行
    print("Remove empty contents completed! ")
    # df.to_csv(corpus_file_name+".csv")  # 写入处理后的数据
    print("===================数据清洗完毕======================")

    return df


def get_phrases(corpus_file_name):
    """ 从excel/csv文件中提取相应的短语组合  """

    print("===================Start Withdraw======================")
    print(DATA_BASE + corpus_file_name + ".csv")
    df = pd.read_csv("../data/" + corpus_file_name + ".csv")  # 读取源数据
    df = df.fillna(" ")  # 用空格 " "替换 nan
    print("Replace NAN completed! ")
    # print(list(df["中性"]))
    pos = [ph.split("；") for ph in df["正向_细"]]
    neu = [ph.split("；") for ph in df["中性_细"]]
    neg = [ph.split("；") for ph in df["负向_细"]]

    pos_phrases, neu_phrases, neg_phrases = [], [], []
    for i in range(len(pos)):
        pos_phrases.extend(pos[i])
        neu_phrases.extend(neu[i])
        neg_phrases.extend(neg[i])

    with open(DATA_BASE + "neg_phrases.txt", "w", encoding="utf-8") as f:
        for ph in neg_phrases:
            if len(ph) > 1:
                f.write(ph + "\n")

    # all_phrases = pos_phrases + neu_phrases + neg_phrases
    # special_phrases = [line.strip() for line in open(DATA_BASE + "special_phrases.txt", encoding='utf-8').readlines()]
    # all_phrases = list(set(special_phrases + all_phrases))
    # # print(all_phrases)
    #
    # with open(DATA_BASE + "special_phrases.txt", "w", encoding="utf-8") as fw:
    #     for ph in all_phrases:
    #         if len(ph) > 1:
    #             fw.write(ph + "\n")
    print("===================Phrases saved in file======================")


def combine_phrases():
    """ 整合化妆品和护肤品的短语 """

    df1 = pd.read_csv(DATA_BASE + "skin_care_phrases" + ".csv")  # 读取源数据
    df2 = pd.read_csv(DATA_BASE + "makeup_phrases" + ".csv")  # 读取源数据
    df = pd.merge(df1, df2, left_on='护肤品', right_on='化妆品', how='outer')
    df = df.fillna(" ")  # 用空格 " "替换 nan
    df.to_csv("combined.csv")  # 写入处理后的数据


def get_specified_phrases():
    """ 按照不同分类获取词组 """

    lines = [line.strip() for line in open(DATA_BASE + "combined_phrases.csv", encoding='utf-8').readlines()]

    phrases_dic = {}
    pos_phrases_dic = {}
    neu_phrases_dic = {}
    neg_phrases_dic = {}
    for i, line in enumerate(lines):
        if i == 0: continue

        items = line.split(",")

        pos_phrases_dic[items[1]] = list(set(items[2].split("；")[:-1]))
        neu_phrases_dic[items[3]] = list(set(items[4].split("；")[:-1]))
        neg_phrases_dic[items[5]] = list(set(items[6].split("；")[:-1]))

    phrases_dic["pos"] = pos_phrases_dic
    phrases_dic["neu"] = neu_phrases_dic
    phrases_dic["neg"] = neg_phrases_dic
    # print(phrases_dic)

    with open(DATA_BASE + "combined_phrases_dict.json", "w", encoding="utf-8") as fw:
        fw.write(json.dumps(phrases_dic, ensure_ascii=False))


def get_raw_data_from_dureader(input_file_path):
    """ 从 dureader的json文件中提取源文本 """

    corpus = [json.loads(item) for item in open(input_file_path, encoding="utf-8").readlines()]
    raw_data = ""
    for item in corpus:
        for dic in item["documents"]:
            raw_data += dic["title"]
            raw_data += " ".join(dic["paragraphs"])

    return raw_data


def combine_dureader():
    file_path1 = BASE_DIR + "/data/dureader_raw/raw/trainset/zhidao.train.json"
    file_path2 = BASE_DIR + "/data/dureader_raw/raw/trainset/search.train.json"
    file_path3 = BASE_DIR + "/data/dureader_raw/raw/testset/zhidao.test.json"
    file_path4 = BASE_DIR + "/data/dureader_raw/raw/testset/search.test.json"
    file_path5 = BASE_DIR + "/data/dureader_raw/raw/devset/zhidao.dev.json"
    file_path6 = BASE_DIR + "/data/dureader_raw/raw/devset/search.dev.json"

    with open(file_path1, encoding="utf-8") as f1,\
        open(file_path2, encoding="utf-8") as f2,\
        open(file_path3, encoding="utf-8") as f3,\
        open(file_path4, encoding="utf-8") as f4,\
        open(file_path5, encoding="utf-8") as f5,\
        open(file_path6, encoding="utf-8") as f6:

        raw1 = f1.read()
        raw2 = f2.read()
        raw3 = f3.read()
        raw4 = f4.read()
        raw5 = f5.read()
        raw6 = f6.read()

    with open(BASE_DIR + "/data/raw_data.txt", "w", encoding="utf-8") as fw:
        fw.write(raw1+raw2+raw3+raw4+raw5+raw6)


if __name__ == '__main__':

    file_path = DATA_BASE + "/dureader_raw/raw/trainset/zhidao.train.json"
    raw_data = get_raw_data_from_dureader(file_path)
    print(raw_data)
