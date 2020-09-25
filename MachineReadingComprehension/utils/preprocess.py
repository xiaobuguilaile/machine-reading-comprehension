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
# print(DATA_BASE)


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
    #
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


if __name__ == '__main__':

    ### 更新源数据文件  ####
    # get_specified_phrases()
    get_phrases("combined_phrases")
    #######################

    # combine_phrases()

    # get_phrases("makeup_phrases")
    # get_phrases("skin_care_phrases")
    # pre_process_data("all_sku2")