# machine-reading-comprehension (MRC) 机器阅读理解

 - 基于 pytorch 实现的 “机器阅读理解” 任务，通过训练 bert-wwm, bert-wwm-ext, robert-wwm 等多种深度学习模型，集成代码的方式
实现多模型的投票方案，给出阅读理解的最终结果。

 - 机器阅读理解 (Machine Reading Comprehension) 是指让机器阅读文本，然后回答和阅读内容相关的问题。阅读理解是自然语言处理和人工智能领域的重要前沿课题，对于提升机器的智能水平、使机器具有持续知识获取的能力等具有重要价值，近年来受到学术界和工业界的广泛关注。
Machine Reading Comprehension (MRC) enables computers to read the text and answer to the questions related to the reading text. It is one of the most important research topics in the area of natural language processing and artificial intelligence. It is of great value to improve the machine intelligence and enable the machine to acquire knowledge. It has gained a lot of attention from both academia and industry in recent years.

## 数据：来自于2个开源数据集

 - SQuAD数据集 https://rajpurkar.github.io/SQuAD-explorer/
 
 - DuReader数据集 http://ai.baidu.com/broad/download?dataset=dureader
 （2020语言与智能技术竞赛：机器阅读理解任务 
 https://aistudio.baidu.com/aistudio/competition/detail/28）

## 项目文件夹

 - transformers：huggingface的开源代码；
 - models：bert, roberta等的 预训练模型；
 - data： 用于存放数据文件，包括原始数据，词表, 词向量文模型，预处理数据等；
 - utils：用于存放数据预处理，计时器，jieba分词加速器等工具；
 - output：输出bert等模型的模型文件（保存.ckpt的模型参数文件）
 - results: 输出结果文件

### Reference

  - Shanshan Liu, Xin Zhang, Sheng Zhang, et al. Neural Machine Reading Comprehension: Methods and Trends
https://arxiv.org/pdf/1907.01118.pdf


### To be continue...
