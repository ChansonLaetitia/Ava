#coding=utf8
import torch
import torch.nn as nn
from torch import optim
import random
import os
from models import decoder as de
from models import encoder as en


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

MAX_LENGTH = 10  # 句子最大长度是10个词(包括EOS等特殊词)

PAD_token= 0
SOS_token= 1
EOS_token= 2
corpus_name = "cornell movie-dialogs corpus"
save_dir = os.path.join("C:\\Users\\82114\\ML\\chatbot\\data\\", "save")

class Train(object):
    def __init__(self,dp):
        self.dp = dp
        voc = dp.voc
        pairs = dp.pairs
        # 配置模型
        model_name = 'cb_model'
        attn_model = 'dot'
        #attn_model = 'general'
        #attn_model = 'concat'
        self.hidden_size = 500
        encoder_n_layers = 2
        decoder_n_layers = 2
        dropout = 0.1
        batch_size = 64
        
        # 从哪个checkpoint恢复，如果是None，那么从头开始训练。
        loadFilename = None
        #checkpoint_iter = 5000
          
        
        # 如果loadFilename不空，则从中加载模型 
        if loadFilename:
            # 如果训练和加载是一条机器，那么直接加载 
            self.checkpoint = torch.load(loadFilename)
            # 否则比如checkpoint是在GPU上得到的，但是我们现在又用CPU来训练或者测试，那么注释掉下面的代码
            #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
            encoder_sd = self.checkpoint['en']
            decoder_sd = self.checkpoint['de']
            encoder_optimizer_sd = self.checkpoint['en_opt']
            decoder_optimizer_sd = self.checkpoint['de_opt']
            embedding_sd = self.checkpoint['embedding']
            voc.__dict__ = self.checkpoint['voc_dict']
        
        print('Building encoder and decoder ...')
        # 初始化word embedding
        embedding = nn.Embedding(voc.num_words, self.hidden_size)
        if loadFilename:
            embedding.load_state_dict(embedding_sd)
        # 初始化encoder和decoder模型
        self.encoder = en.EncoderRNN(self.hidden_size, embedding, encoder_n_layers, dropout)
        self.decoder = de.LuongAttnDecoderRNN(attn_model, embedding, self.hidden_size, voc.num_words, decoder_n_layers, dropout)
        if loadFilename:
            self.encoder.load_state_dict(encoder_sd)
            self.decoder.load_state_dict(decoder_sd)
        # 使用合适的设备
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
        print('Models built and ready to go!')
        
        # 配置训练的超参数和优化器 
        clip = 50.0
        self.teacher_forcing_ratio = 1.0
        learning_rate = 0.0001
        decoder_learning_ratio = 5.0
        n_iteration = 20
        print_every = 1
        save_every = 10
        
        # 设置进入训练模式，从而开启dropout 
        self.encoder.train()
        self.decoder.train()
        
        # 初始化优化器 
        print('Building optimizers ...')
        encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
        if loadFilename:
            encoder_optimizer.load_state_dict(encoder_optimizer_sd)
            decoder_optimizer.load_state_dict(decoder_optimizer_sd)
        
        # 开始训练
        print("Starting Training!")
        self.trainIters(model_name, voc, pairs, self.encoder, self.decoder, encoder_optimizer, decoder_optimizer,
                   embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
                   print_every, save_every, clip, corpus_name, loadFilename)
        
        '''
        # 进入eval模式，从而去掉dropout。 
        encoder.eval()
        decoder.eval()
        
        # 构造searcher对象 
        searcher = gs.GreedySearchDecoder(encoder, decoder)
        
        # 测试
        ev.Evaluation(self.dp,encoder, decoder, searcher, voc).evaluateInput()
        '''
    
    def maskNLLLoss(self,inp, target, mask):
        # 计算实际的词的个数，因为padding是0，非padding是1，因此sum就可以得到词的个数
        nTotal = mask.sum()
        
        crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
        loss = crossEntropy.masked_select(mask).mean()
        #loss = loss.to(device)
        return loss, nTotal.item()
    
    def train(self,input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding,
              encoder_optimizer, decoder_optimizer, batch_size, clip, max_length=MAX_LENGTH):
    
        # 梯度清空
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
    
        # 设置device，从而支持GPU，当然如果没有GPU也能工作。
        input_variable = input_variable.to(device)
        lengths = lengths.to(device)
        target_variable = target_variable.to(device)
        mask = mask.to(device)
    
        # 初始化变量
        loss = 0
        print_losses = []
        n_totals = 0
    
        # encoder的Forward计算
        encoder_outputs, encoder_hidden = encoder(input_variable, lengths)
    
        # Decoder的初始输入是SOS，我们需要构造(1, batch)的输入，表示第一个时刻batch个输入。
        decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
        decoder_input = decoder_input.to(device)
    
        # 注意：Encoder是双向的，而Decoder是单向的，因此从下往上取n_layers个
        decoder_hidden = encoder_hidden[:decoder.n_layers]
    
        # 确定是否teacher forcing
        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
    
        # 一次处理一个时刻 
        if use_teacher_forcing:
            for t in range(max_target_len):
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                # Teacher forcing: 下一个时刻的输入是当前正确答案
                decoder_input = target_variable[t].view(1, -1)
                # 计算累计的loss
                mask_loss, nTotal = self.maskNLLLoss(decoder_output, target_variable[t], mask[t])
                loss += mask_loss
                print_losses.append(mask_loss.item() * nTotal)
                n_totals += nTotal
        else:
            for t in range(max_target_len):
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                # 不是teacher forcing: 下一个时刻的输入是当前模型预测概率最高的值
                _, topi = decoder_output.topk(1)
                decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
                decoder_input = decoder_input.to(device)
                # 计算累计的loss
                mask_loss, nTotal = self.maskNLLLoss(decoder_output, target_variable[t], mask[t])
                loss += mask_loss
                print_losses.append(mask_loss.item() * nTotal)
                n_totals += nTotal
    
        # 反向计算 
        loss.backward()
    
        # 对encoder和decoder进行梯度裁剪
        _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
        _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)
    
        # 更新参数
        encoder_optimizer.step()
        decoder_optimizer.step()
    
        return sum(print_losses) / n_totals
    
    def trainIters(self,model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size, print_every, save_every, clip, corpus_name, loadFilename):
    
        # 随机选择n_iteration个batch的数据(pair)
        training_batches = [self.dp.batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)])
                          for _ in range(n_iteration)]
    
        # 初始化
        print('Initializing ...')
        start_iteration = 1
        print_loss = 0
        if loadFilename:
            start_iteration = self.checkpoint['iteration'] + 1
    
        # 训练
        print("Training...")
        for iteration in range(start_iteration, n_iteration + 1):
            training_batch = training_batches[iteration - 1]
            
            input_variable, lengths, target_variable, mask, max_target_len = training_batch
    
            # 训练一个batch的数据
            loss = self.train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                         decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip)
            print_loss += loss
    
            # 进度
            if iteration % print_every == 0:
                print_loss_avg = print_loss / print_every
                print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, iteration / n_iteration * 100, print_loss_avg))
                print_loss = 0
    
            # 保存checkpoint
            if (iteration % save_every == 0):
                directory = os.path.join(save_dir, model_name, corpus_name, '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, self.hidden_size))
                #print (directory)
                if not os.path.exists(directory):
                    print ("Not Exist!")
                    os.makedirs(directory)
                    #print (directory)
                torch.save({
                    'iteration': iteration,
                    'en': encoder.state_dict(),
                    'de': decoder.state_dict(),
                    'en_opt': encoder_optimizer.state_dict(),
                    'de_opt': decoder_optimizer.state_dict(),
                    'loss': loss,
                    'voc_dict': voc.__dict__,
                    'embedding': embedding.state_dict()
                }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))
                
                
                