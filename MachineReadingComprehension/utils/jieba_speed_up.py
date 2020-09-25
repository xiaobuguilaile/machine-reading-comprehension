# -*-coding:utf-8 -*-

'''
@File       : jieba_speed_up.py
@Author     : HW Shen
@Date       : 2020/9/25
@Desc       :
'''

import pandas as pd
from multiprocessing import Pool, cpu_count
from MachineReadingComprehension.utils.timer import time_count
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import jieba_fast as jieba  # 代替原版jieba, 速度更快

NUMBER_OF_PROCESSES = cpu_count()  # CPU核数


def content_cut(text):
    """ 将文本按照CPU数量切分 """

    file_len = len(text)
    seg_len = int(file_len / NUMBER_OF_PROCESSES)
    str_list = [text[((i - 1) * seg_len):(i * seg_len - 1)] for i in range(1, NUMBER_OF_PROCESSES + 1, 1)]
    return str_list


def segment_accurate_mode(txt_string):
    """ 精确模式：
    我是上海交通大学学生
        => ['我', '是', '上海交通大学', '学生'] """
    return jieba.lcut(txt_string)


def segment_full_mode(txt_string):
    """ 全模式：
    我是上海交通大学学生
        => ['我', '是', '上海', '上海交通大学', '交通', '大学', '学学', '学生'] """
    return jieba.lcut(txt_string, cut_all=True)


def segment_search_mode(txt_string):
    """ 搜索模式：
    我是上海交通大学学生
        => ['我', '是', '上海', '交通', '大学', '上海交通大学', '学生'] """
    return jieba.lcut_for_search(txt_string)


@time_count
def chinese_word_segment(text:str, seg_mode='accurate'):
    """ 并行切词 """

    txt_list = content_cut(text)  # 将文本切分成多份

    pool = Pool(NUMBER_OF_PROCESSES)  # 开启线程池并行分词
    if seg_mode == 'accurate':
        words_list = pool.map(func=segment_accurate_mode, iterable=txt_list)
    elif seg_mode == 'full':
        words_list = pool.map(func=segment_full_mode, iterable=txt_list)
    else:
        words_list = pool.map(func=segment_search_mode, iterable=txt_list)
    pool.close()
    pool.join()

    cut_words = []
    for words in words_list:
        cut_words.extend(words)

    return cut_words


if __name__ == '__main__':

    df = pd.read_csv("../test/HXZ.csv")  # 读取源数据
    # df = pd.read_csv("../test/WMRJ.csv")  # 读取源数据
    df = df.fillna(" ")  # 用空格 " "替换 nan
    text = " ".join(list(df["title"])) + " ".join(list(df["content"]))

    jieba_words = chinese_word_segment(text, seg_mode='accurate')

    print(jieba_words[:100])

    # print(segment_accurate_mode("我是上海交通大学学生"))
    # print(segment_full_mode("我是上海交通大学学生"))
    # print(segment_search_mode("我是上海交通大学学生"))
