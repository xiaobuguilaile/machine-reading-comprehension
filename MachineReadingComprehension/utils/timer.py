# -*-coding:utf-8 -*-

'''
@File       : timer.py
@Author     : HW Shen
@Date       : 2020/9/25
@Desc       : 计时器
'''

import time
from loguru import logger


def time_count(func):

    def deco(*args, **kwargs):
        logger.info('\n函数：\033[32;1m{_funcname_}()\033[0m 开始运行：'.format(_funcname_=func.__name__))
        start_time = time.time()
        res = func(*args, **kwargs)
        end_time = time.time()
        logger.info('函数: \033[32;1m{_funcname_}()\033[0m 运行了 {_time_}秒'
              .format(_funcname_=func.__name__, _time_=(end_time - start_time)))
        return res

    return deco

