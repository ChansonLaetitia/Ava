#coding=utf8
'''
Created on 2019年5月6日

@author: 82114
'''


import torch
#import random
import re
import os
import unicodedata
from io import open
import itertools
from vocabulary import vocabulary

MAX_LENGTH = 10  # 句子最大长度是10个词(包括EOS等特殊词)
MIN_COUNT = 3    # 阈值为3

PAD_token= 0
SOS_token= 1
EOS_token= 2

corpus_name = "cornell movie-dialogs corpus"
corpus = os.path.join("C:\\Users\\82114\\eclipse-workspace\\chatbot\\data\\", corpus_name)
datafile = os.path.join(corpus, "formatted_movie_lines.txt")

class DataPreprocessing(object):
    
    def __init__(self):
        # Load/Assemble voc and pairs
        #save_dir = os.path.join("data", "save")
        self.voc, self.pairs = self.loadPrepareData(corpus, corpus_name, datafile)
        
        '''
        # 输出一些句对
        print("\npairs:")
        for pair in pairs[:10]:
            print(pair)
        '''
        
        # 实际进行处理
        self.pairs = self.trimRareWords(self.voc, self.pairs, MIN_COUNT)
        
    
    # 把Unicode字符串变成ASCII
    # 参考https://stackoverflow.com/a/518232/2809427
    def unicodeToAscii(self,s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )
     
    def normalizeString(self,s):
        # 变成小写、去掉前后空格，然后unicode变成ascii
        s = self.unicodeToAscii(s.lower().strip())
        
        # 在标点前增加空格，这样把标点当成一个词
        s = re.sub(r"([.!?])", r" \1", s)
        
        # 字母和标点之外的字符都变成空格
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        
        # 因为把不用的字符都变成空格，所以可能存在多个连续空格
        # 下面的正则替换把多个空格变成一个空格，最后去掉前后空格
        s = re.sub(r"\s+", r" ", s).strip()
        return s
    
    # 读取问答句对并且返回Voc词典对象 
    def readVocs(self,datafile, corpus_name):
        print("Reading lines...")
        
        # 文件每行读取到list lines中。 
        lines = open(datafile, encoding='utf-8').\
            read().strip().split('\n')
            
        # 每行用tab切分成问答两个句子，然后调用normalizeString函数进行处理。
        pairs = [[self.normalizeString(s) for s in l.split('\t')] for l in lines]
        voc = vocabulary.Voc(corpus_name)
        return voc, pairs
    
    def filterPair(self,p): 
        return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH
    
    # 过滤太长的句对 
    def filterPairs(self,pairs):
        return [pair for pair in pairs if self.filterPair(pair)]
    
    # 使用上面的函数进行处理，返回Voc对象和句对的list 
    def loadPrepareData(self,corpus, corpus_name, datafile):
        print("Start preparing training data ...")
        voc, pairs = self.readVocs(datafile, corpus_name)
        print("Read {!s} sentence pairs".format(len(pairs)))
        pairs = self.filterPairs(pairs)
        print("Trimmed to {!s} sentence pairs".format(len(pairs)))
        print("Counting words...")
        for pair in pairs:
            voc.addSentence(pair[0])
            voc.addSentence(pair[1])
        print("Counted words:", voc.num_words)
        return voc, pairs
    
    def trimRareWords(self,voc, pairs, MIN_COUNT):
        # 去掉voc中频次小于3的词 
        voc.trim(MIN_COUNT)
        # 保留的句对 
        keep_pairs = []
        for pair in pairs:
            input_sentence = pair[0]
            output_sentence = pair[1]
            keep_input = True
            keep_output = True
            # 检查问题
            for word in input_sentence.split(' '):
                if word not in voc.word2index:
                    keep_input = False
                    break
            # 检查答案
            for word in output_sentence.split(' '):
                if word not in voc.word2index:
                    keep_output = False
                    break
    
            # 如果问题和答案都只包含高频词，我们才保留这个句对
            if keep_input and keep_output:
                keep_pairs.append(pair)
    
        print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
        return keep_pairs
    
    # 把句子的词变成ID
    def indexesFromSentence(self,voc, sentence):
        return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]
    
    # l是多个长度不同句子(list)，使用zip_longest padding成定长，长度为最长句子的长度。
    def zeroPadding(self,l, fillvalue=PAD_token):
        return list(itertools.zip_longest(*l, fillvalue=fillvalue))
    
    # l是二维的padding后的list
    # 返回m和l的大小一样，如果某个位置是padding，那么值为0，否则为1
    def binaryMatrix(self,l, value=PAD_token):
        m = []
        for i, seq in enumerate(l):
            m.append([])
            for token in seq:
                if token == PAD_token:
                    m[i].append(0)
                else:
                    m[i].append(1)
        return m
    
    # 把输入句子变成ID，然后再padding，同时返回lengths这个list，标识实际长度。
    # 返回的padVar是一个LongTensor，shape是(batch, max_length)，
    # lengths是一个list，长度为(batch,)，表示每个句子的实际长度。
    def inputVar(self,l, voc):
        indexes_batch = [self.indexesFromSentence(voc, sentence) for sentence in l]
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        padList = self.zeroPadding(indexes_batch)
        #print (type(padList))
        padVar = torch.LongTensor(padList)
        return padVar, lengths
    
    # 对输出句子进行padding，然后用binaryMatrix得到每个位置是padding(0)还是非padding，
    # 同时返回最大最长句子的长度(也就是padding后的长度)
    # 返回值padVar是LongTensor，shape是(batch, max_target_length)
    # mask是ByteTensor，shape也是(batch, max_target_length)
    def outputVar(self,l, voc):
        indexes_batch = [self.indexesFromSentence(voc, sentence) for sentence in l]
        max_target_len = max([len(indexes) for indexes in indexes_batch])
        padList = self.zeroPadding(indexes_batch)
        mask = self.binaryMatrix(padList)
        mask = torch.ByteTensor(mask)
        padVar = torch.LongTensor(padList)
        return padVar, mask, max_target_len
    
    # 处理一个batch的pair句对 
    def batch2TrainData(self,voc, pair_batch):
        # 按照句子的长度(词数)排序
        pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
        input_batch, output_batch = [], []
        for pair in pair_batch:
            input_batch.append(pair[0])
            output_batch.append(pair[1])
        inp, lengths = self.inputVar(input_batch, voc)
        output, mask, max_target_len = self.outputVar(output_batch, voc)
        return inp, lengths, output, mask, max_target_len


