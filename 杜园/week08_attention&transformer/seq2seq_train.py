import json
import pickle
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from process_data import collate_fn,split_data
from EncoderDecoderAttenModel import Seq2seq
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os

if __name__ == '__main__':
    # os.chdir(os.getcwd() + "/homework")
    # device = torch.device('cuda')
    
    epochs = 10
    batch_size = 256
    
    emb_dim = 200
    hidden_dim = 250
    dropout = 0.5
    
    # 加载分词
    with open('./data/proc/train_in_words.json', 'r') as f:
        in_tokens = json.load(f)
    with open('./data/proc/train_out_words.json', 'r') as f:
        out_tokens = json.load(f)
        
    # 加载词典
    with open('./data/proc/vocabs.bin', 'rb') as f:
        vocabs = pickle.load(f)
        
    # 资源有限，获取部分数据训练
    # in_tokens,out_tokens = split_data(in_tokens, out_tokens, 5000)
    
     # 数据集
    dataset = list(zip(in_tokens, out_tokens))
    dataloader = DataLoader(dataset, batch_size, shuffle=True,collate_fn=collate_fn(vocabs))
    
    # for in_tokens_idxs, out_tokens_idxs, labels in dataloader:
    #     print(in_tokens_idxs.shape) # (128,25)
    #     print(out_tokens_idxs.shape) # (128,26)
    #     print(labels.shape) # (128,26)
    #     break
    
    # 定义模型 损失函数 优化器
    model = Seq2seq(len(vocabs), emb_dim, hidden_dim, dropout)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimzer = torch.optim.Adam(model.parameters(), lr=0.05)
    
    # model.to(device)
    writer = SummaryWriter()
    
    # 训练数据
    train_loss_cnt = 0
    for epoch in range(epochs):
        model.train()
        tbar = tqdm(dataloader)
        for in_tokens_idxs, out_tokens_idxs, labels in tbar:
            # in_tokens_idxs = in_tokens_idxs.to(device)
            # out_tokens_idxs = out_tokens_idxs.to(device)
            # labels = labels.to(device)
            # 前向运算
            logits = model(in_tokens_idxs, out_tokens_idxs)
            # 转化前：logits:(batch_size,seq_len,vocab_size) labels(batch_size, vocab_size)
            # 转化后：logits:(batch_size*seq_len,vocab_size) labels(batch_size*vocab_size)
            loss = loss_fn(logits.view(-1,logits.size(-1)), labels.view(-1))
                         
            # 反向传播
            optimzer.zero_grad()
            loss.backward()
            optimzer.step()
            
            tbar.set_description(f'Epoch {epoch+1} Loss {loss.item():.4f}')
            writer.add_scalar('Loss/train', loss.item(), train_loss_cnt)
            train_loss_cnt += 1
    
    torch.save(model.state_dict(), './data/model/seq2seq_state.bin')    
    