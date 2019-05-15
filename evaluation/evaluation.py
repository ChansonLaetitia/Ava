#coding=utf8
'''
Created on 2019年5月7日

@author: 82114
'''
import torch
from models import greedysearchdecoder as gs

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

MAX_LENGTH = 10  # 句子最大长度是10个词(包括EOS等特殊词)
#MIN_COUNT = 3    # 阈值为3

class Evaluation(object):
    def __init__(self,dp,encoder, decoder):
        self.dp = dp
        self.voc = dp.voc
        
        # 进入eval模式，从而去掉dropout。 
        print("Starting Evaluating!")
        encoder.eval()
        decoder.eval()
        
        # 构造searcher对象 
        self.searcher = gs.GreedySearchDecoder(encoder, decoder)
        
        # 测试
        self.evaluateInput()
        
    
    def evaluate(self, sentence, max_length=MAX_LENGTH):
        
        ### 把输入的一个batch句子变成id
        indexes_batch = [self.dp.indexesFromSentence(self.voc, sentence)]
        # 创建lengths tensor
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        # 转置 
        input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
        # 放到合适的设备上(比如GPU)
        input_batch = input_batch.to(device)
        lengths = lengths.to(device)
        # 用searcher解码
        tokens, scores = self.searcher(input_batch, lengths, max_length)
        # ID变成词。
        decoded_words = [self.voc.index2word[token.item()] for token in tokens]
        return decoded_words
    
    
    def evaluateInput(self):
        input_sentence = ''
        while(1):
            try:
                # 得到用户终端的输入
                input_sentence = input('> ')
                
                # 是否退出
                if input_sentence == 'q' or input_sentence == 'quit': break
                
                # 句子归一化
                input_sentence = self.dp.normalizeString(input_sentence)
                
                # 生成响应Evaluate sentence
                output_words = self.evaluate(input_sentence)
                
                # 去掉EOS后面的内容
                words = []
                for word in output_words:
                    if word == 'EOS':
                        break
                    elif word != 'PAD':
                        words.append(word)
                print('Bot:', ' '.join(words))
    
            except KeyError:
                print("Error: Encountered unknown word.")
                
                