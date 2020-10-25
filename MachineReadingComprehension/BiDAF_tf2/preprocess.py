# -*-coding:utf-8 -*-

'''
@File       : preprocess.py
@Author     : HW Shen
@Date       : 2020/9/27
@Desc       : 预处理实现类
'''

import numpy as np
import MachineReadingComprehension.BiDAF_tf2.data_io as pio
import os
import nltk
from bert_serving.client import BertClient

BASE_DIR = ""

GLOVE_FILE_PATH = "data/glove.6B.50d.txt"

# 查看服务端的端口号 5555，5556，ip为本机，时间为10000s, 为了防止一直等待
bc = BertClient(ip='localhost', check_version=False, port=5555, port_out=5556, check_length=False, timeout=10000)


class Preprocessor:

    def __init__(self, datasets_fp, max_length=384, stride=128):

        self.datasets_fp = datasets_fp
        self.max_length = max_length
        self.max_clen = 100  # context最大长度
        self.max_qlen = 100  # query最大长度
        self.max_char_len=16
        self.stride = stride
        self.charset = set()
        self.build_charset()
        self.embeddings_index = {}
        self.embedding_matrix = []
        self.word_list = []
        self.load_glove(GLOVE_FILE_PATH)
        self.build_wordset()

    def build_charset(self):
        for fp in self.datasets_fp:
            self.charset |= self.dataset_char_info(fp)

        self.charset = sorted(list(self.charset))
        self.charset = ['[PAD]', '[CLS]', '[SEP]'] + self.charset + ['[UNK]']
        idx = list(range(len(self.charset)))
        self.ch2id = dict(zip(self.charset, idx))
        self.id2ch = dict(zip(idx, self.charset))
        print(self.ch2id, self.id2ch)
        
    def build_wordset(self):
        idx = list(range(len(self.word_list)))
        self.w2id = dict(zip(self.word_list, idx))
        self.id2w = dict(zip(idx, self.word_list))

    def dataset_char_info(self, inn):
        charset = set()
        dataset = pio.load(inn)

        for _, context, question, answer, _ in self.iter_cqa(dataset):
            charset |= set(context) | set(question) | set(answer)
            # self.max_clen = max(self.max_clen, len(context))
            # self.max_qlen = max(self.max_clen, len(question))

        return charset

    def iter_cqa(self, dataset):
        for data in dataset['data']:
            for paragraph in data['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    qid = qa['id']
                    question = qa['question']
                    for answer in qa['answers']:
                        text = answer['text']
                        answer_start = answer['answer_start']
                        yield qid, context, question, text, answer_start

    def char_encode(self, context, question):
        q_seg_list = self.seg_text(question)
        c_seg_list = self.seg_text(context)
        question_encode = self.convert2id_char(max_char_len=self.max_char_len, begin=True, end=True, word_list=q_seg_list)
        print(question_encode)
        left_length = self.max_length - len(question_encode)
        context_encode = self.convert2id_char(max_char_len=self.max_char_len,maxlen=left_length, end=True, word_list=c_seg_list)
        cq_encode = question_encode + context_encode

        return cq_encode

    def word_encode(self, context, question):
        q_seg_list = self.seg_text(question)
        c_seg_list = self.seg_text(context)
        question_encode = self.convert2id_word(begin=True, end=True, word_list=q_seg_list)
        left_length = self.max_length - len(question_encode)
        context_encode = self.convert2id_word(maxlen=left_length, end=True, word_list=c_seg_list)
        cq_encode = question_encode + context_encode

        assert len(cq_encode) == self.max_length

        return cq_encode

    def convert2id_char(self, max_char_len=None, maxlen=None, begin=False, end=False, word_list = []):
        char_list = []
        char_list = [[self.get_id_char('[CLS]')] + [self.get_id_char('[PAD]')] * (max_char_len-1)] * begin + char_list
        for word in word_list:
            ch = [ch for ch in word]
            if max_char_len is not None:
                ch = ch[:max_char_len]

            ids = list(map(self.get_id_char, ch))
            while len(ids) < max_char_len:
                ids.append(self.get_id_char('[PAD]'))
            char_list.append(np.array(ids))

        if maxlen is not None:
            char_list = char_list[:maxlen - 1 * end]
            # char_list += [[self.get_id_char('[SEP]')]] * end
            char_list += [[self.get_id_char('[PAD]')] * max_char_len] * (maxlen - len(char_list))
        # else:
        #     char_list += [[self.get_id_char('[SEP]')]] * end

        return char_list

    def convert2id_word(self, maxlen=None, begin=False, end=False, word_list=[]):
        ch = [ch for ch in word_list]
        ch = ['cls'] * begin + ch

        if maxlen is not None:
            ch = ch[:maxlen - 1 * end]
            # ch += ['sep'] * end
            ch += ['pad'] * (maxlen - len(ch))
        # else:
        #     ch += ['sep'] * end

        ids = list(map(self.get_id_word, ch))

        return ids

    def get_id_char(self, ch):
        return self.ch2id.get(ch, self.ch2id['[UNK]'])

    def get_id_word(self, ch):
        return self.w2id.get(ch, self.w2id['unk'])

    def seg_text(self, text):
        words = [word.lower() for word in nltk.word_tokenize(text)]
        return words

    def load_glove(self, glove_file_path):
        with open(glove_file_path, encoding='utf-8') as fr:
            for line in fr:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, sep=' ')
                self.embeddings_index[word] = coefs
                self.word_list.append(word)
                self.embedding_matrix.append(coefs)

    def encode(self, context, question):
        question_encode = self.convert2id(question, begin=True, end=True)
        left_length = self.max_length - len(question_encode)
        context_encode = self.convert2id(context, maxlen=left_length, end=True)
        cq_encode = question_encode + context_encode

        assert len(cq_encode) == self.max_length

        return cq_encode

    def convert2id(self, sent, maxlen=None, begin=False, end=False):
        ch = [ch for ch in sent]
        ch = ['[CLS]'] * begin + ch

        if maxlen is not None:
            ch = ch[:maxlen - 1 * end]
            ch += ['[SEP]'] * end
            ch += ['[PAD]'] * (maxlen - len(ch))
        else:
            ch += ['[SEP]'] * end

        ids = list(map(self.get_id, ch))

        return ids

    def get_id(self, ch):
        return self.ch2id.get(ch, self.ch2id['[UNK]'])

    def get_dataset(self, ds_fp):
        cs, qs, be = [], [], []
        for _, c, q, b, e in self.get_data(ds_fp):
            cs.append(c)
            qs.append(q)
            be.append((b, e))
        return map(np.array, (cs, qs, be))

    def get_data(self, ds_fp):
        dataset = pio.load(ds_fp)
        for qid, context, question, text, answer_start in self.iter_cqa(dataset):
            cids = self.get_sent_ids(context, self.max_clen)
            qids = self.get_sent_ids(question, self.max_qlen)
            b, e = answer_start, answer_start + len(text)
            if e >= len(cids):
                b = e = 0
            yield qid, cids, qids, b, e

    def get_sent_ids(self, sent, maxlen):
        return self.convert2id(sent, maxlen=maxlen, end=True)

    def get_bert_data(self, ds_fp):

        dataset = pio.load(ds_fp)
        for qid, context, question, text, answer_start in self.iter_cqa(dataset):
            # cids = self.get_sent_ids(context, self.max_clen)
            # qids = self.get_sent_ids(question, self.max_qlen)
            # b, e = answer_start, answer_start + len(text)
            cids = bc.encode([context[:self.max_clen]])[0]
            qids = bc.encode([question[:self.max_qlen]])[0]
            b, e = answer_start, answer_start + len(text)
            yield qid, cids, qids, b, e

    def bert_feature(self, dataset_fp):
        # 获取bert的特征值
        cs, qs, be = [], [], []  # 初始化
        for _, c, q, b, e in self.get_bert_data(dataset_fp):
            cs.append(c)
            qs.append(q)
            be.append((b, e))
        return map(np.array, (cs, qs, be))


if __name__ == '__main__':
    p = Preprocessor([
        './data/squad/train-v1.1.json',
        './data/squad/dev-v1.1.json',
        './data/squad/dev-v1.1.json'
    ])
    print(p.encode('modern stone statue of Mary', 'To whom did the Virgin Mary '))