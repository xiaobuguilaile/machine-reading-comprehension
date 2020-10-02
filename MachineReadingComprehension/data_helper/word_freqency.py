# -*-coding:utf-8 -*-

'''
@File       : word_freqency.py
@Author     : HW Shen
@Date       : 2020/9/25
@Desc       :
'''

from collections import Counter
import json
from loguru import logger
import re
import jieba_fast as jieba

from MachineReadingComprehension.utils.timer import time_count
from MachineReadingComprehension.utils.jieba_speed_up import chinese_word_segment

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class GenerateWordFreq(object):
    """ 词频统计 """

    def __init__(self, mode="normal"):

        self.mode = mode
        self.load_data()  # 加载所需数据集
        self.output_dic = {}  # 输出词频结果

    def load_data(self):

        # self.stopwords = [line.strip() for line in open(BASE_DIR + "/data/chinese_stopwords.txt", encoding="utf-8").readlines()]
        pass

    @time_count
    def get_words_freq(self, text:str):
        """ 词频生成主程序 """

        logger.info(" text length is : {} ".format(len(text)))

        # clean_text = re.findall('[\u4e00-\u9fa5A-Za-z0-9]+', text, re.S)  # 只保留中文，字母，数字，标点
        # clean_text = re.findall('[\u4e00-\u9fa5]+', text, re.S)  # 只保留中文
        # text = "".join(clean_text)

        # jieba_words = [word for word in self.jieba_cut(text) if word not in self.stopwords]  # jieba分词
        # jieba_words = [word for word in self.jieba_cut(text)]  # jieba分词
        jieba_words = jieba.lcut(text) # jieba分词
        self.count_dict = Counter(jieba_words)  # 词频统计
        # print("count_dict", self.count_dict)

        return self.count_dict

    def jieba_cut(self, text):

        return chinese_word_segment(text)


if __name__ == '__main__':

    # import pandas as pd
    # df = pd.read_csv(BASE_DIR + "/test/kwaishou_comment.csv")  # 读取源数据，将数据解析为时间格式
    # df = df.drop_duplicates()  # 去重
    # print("Remove duplicate items completed! ")
    # df = df.dropna(subset=["评论内容"])  # 删除 “评论内容” 空值行
    # text = " ".join(list(df["评论内容"]))

    # text = open(BASE_DIR + "/data/raw_data.txt", encoding="utf-8").read()
    # clean_text = re.findall('[\u4e00-\u9fa5]+', text, re.S)  # 只保留中文
    # text = "".join(clean_text)
    # with open(BASE_DIR + "/data/clean_data.txt", "w", encoding="utf-8") as fw:
    #     fw.write(text)

    # text = open(BASE_DIR + "/data/clean_data.txt", encoding="utf-8").read()
    # g = GenerateWordFreq()
    # count_dict = g.get_words_freq(text)

    seg_words = []
    fr = open(BASE_DIR + "/data/clean_data.txt", encoding="utf-8")
    content = fr.read(1028 * 32)
    i = 0
    while content:
        if i % 500 == 0: print(i)
        jieba_words = jieba.lcut(content)
        seg_words.extend(jieba_words)
        content = fr.read(1028 * 32)
        i += 1
        if i == 10000: break
    fr.close()

    count_dict = Counter(seg_words)
    # with open(BASE_DIR + "/data/words_frequency.json", "w", encoding="utf-8") as fw:
    #     fw.write(json.dumps(count_dict, ensure_ascii=False) + "\n")
    # content = json.loads(open(BASE_DIR + "/data/words_frequency.json", encoding="utf-8").read())

    new_dic = dict(sorted(count_dict.items(), key=lambda kv: kv[1], reverse=True))

    with open(BASE_DIR + "/data/soreted_words_frequency.json", "w", encoding="utf-8") as fw:
        fw.write(json.dumps(new_dic, ensure_ascii=False) + "\n")

