#coding=utf8
'''
Created on 2019年5月6日

@author: 82114
'''

import csv
import os
import codecs
from io import open

corpus_name = "cornell movie-dialogs corpus"
corpus = os.path.join("C:\\Users\\82114\\eclipse-workspace\\chatbot\\data\\", corpus_name)

class DataExtract(object):
    def __init__(self):
        # 定义新的文件 
        datafile = os.path.join(corpus, "formatted_movie_lines.txt")
        
        delimiter = '\t'
        # 对分隔符delimiter进行decode，这里对tab进行decode结果并没有变
        delimiter = str(codecs.decode(delimiter, "unicode_escape"))
        
        # 初始化dict lines，list conversations以及前面我们介绍过的field的id数组。
        lines = {}
        conversations = []
        MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
        MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]
        
        # 首先使用loadLines函数处理movie_lines.txt 
        print("\nProcessing corpus...")
        lines = self.loadLines(os.path.join(corpus, "movie_lines.txt"), MOVIE_LINES_FIELDS)
        # 接着使用loadConversations处理上一步的结果，得到conversations
        print("\nLoading conversations...")
        conversations = self.loadConversations(os.path.join(corpus, "movie_conversations.txt"),
                                          lines, MOVIE_CONVERSATIONS_FIELDS)
        
        # 输出到一个新的csv文件
        print("\nWriting newly formatted file...")
        with open(datafile, 'w', encoding='utf-8') as outputfile:
            writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
            # 使用extractSentencePairs从conversations里抽取句对。
            for pair in self.extractSentencePairs(conversations):
                writer.writerow(pair)
        
        # 输出一些行用于检查 
        print("\nSample lines from file:")
        self.printLines(datafile)
        
    def printLines(self,file, n=10):
        with open(file, 'rb') as datafile:
            lines = datafile.readlines()
        for line in lines[:n]:
            print(line)
    
    #printLines(os.path.join(corpus, "movie_lines.txt"))
    
    # 把每一行都parse成一个dict，key是lineID、characterID、movieID、character和text
    # 分别代表这一行的ID、人物ID、电影ID，人物名称和文本。
    # 最终输出一个dict，key是lineID，value是一个dict。
    # value这个dict的key是lineID、characterID、movieID、character和text
    def loadLines(self,fileName, fields):
        lines = {}
        with open(fileName, 'r', encoding='iso-8859-1') as f:
            for line in f:
                values = line.split(" +++$+++ ")
                # 抽取fields
                lineObj = {}
                for i, field in enumerate(fields):
                    lineObj[field] = values[i]
                lines[lineObj['lineID']] = lineObj
        return lines
    
    
    # 根据movie_conversations.txt文件和上输出的lines，把utterance组成对话。
    # 最终输出一个list，这个list的每一个元素都是一个dict，key分别是character1ID、character2ID、movieID和utteranceIDs。
    # 分别表示这对话的第一个人物的ID，第二个的ID，电影的ID以及它包含的utteranceIDs
    # 最后根据lines，还给每一行的dict增加一个key为lines，其value是个list，包含所有utterance(上面得到的lines的value)
    def loadConversations(self,fileName, lines, fields):
        conversations = []
        with open(fileName, 'r', encoding='iso-8859-1') as f:
            for line in f:
                values = line.split(" +++$+++ ")
                # 抽取fields
                convObj = {}
                for i, field in enumerate(fields):
                    convObj[field] = values[i]
                # convObj["utteranceIDs"]是一个字符串，形如['L198', 'L199']
                # 我们用eval把这个字符串变成一个字符串的list。
                lineIds = eval(convObj["utteranceIDs"])
                # 根据lineIds构造一个数组，根据lineId去lines里检索出存储utterance对象。
                convObj["lines"] = []
                for lineId in lineIds:
                    convObj["lines"].append(lines[lineId])
                conversations.append(convObj)
        return conversations
    
    
    # 从对话中抽取句对 
    # 假设一段对话包含s1,s2,s3,s4这4个utterance
    # 那么会返回3个句对：s1-s2,s2-s3和s3-s4。
    def extractSentencePairs(self,conversations):
        qa_pairs = []
        for conversation in conversations:
            # 遍历对话中的每一个句子，忽略最后一个句子，因为没有答案。
            for i in range(len(conversation["lines"]) - 1): 
                inputLine = conversation["lines"][i]["text"].strip()
                targetLine = conversation["lines"][i+1]["text"].strip()
                # 如果有空的句子就去掉 
                if inputLine and targetLine:
                    qa_pairs.append([inputLine, targetLine])
        return qa_pairs
    
    
