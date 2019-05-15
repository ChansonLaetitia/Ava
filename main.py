#coding=utf8

'''
Created on 2019年5月6日

@author: 82114
'''

from preprocessing import datapreprocessing
from train import train
from evaluation import evaluation
            
if __name__ == '__main__':
    #de = dataextract.DataExtract()
    dp = datapreprocessing.DataPreprocessing()
    tr = train.Train(dp)
    ev = evaluation.Evaluation(dp,tr.encoder,tr.decoder)
    
    
