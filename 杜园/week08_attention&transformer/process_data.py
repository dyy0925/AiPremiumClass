import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import pickle
import json
import numpy as np
import os as os

# 训练集处理
# 行处理-分词、转为对应词典下标

def readFile(fileName, is_out):
    data_array = []
    with open(fileName, 'r', encoding='utf-8') as f:
        all_lines = f.read().split('\n')
        for line in all_lines:
            if line.strip() == '':
                continue
                        
            # 分词
            if is_out:
                data_array.append(line.strip().split())
            else:
                data_array.append(line.strip().split())
            
    return data_array


def getVocab(fileName):
    with open(fileName, encoding='utf-8') as f:
        all_words = f.read().split("\n") 
        tokens = ['<pad>'] + [tk for tk in all_words if tk != '']
        vocabs = {tk:i for i,tk in enumerate(tokens)}
    return vocabs


def collate_fn(vocabs):
    def getVector(batch_data):
        in_tokens_idxs = []
        out_tokens_idxs = []
        labels = []
        for in_tokens, out_tokens in batch_data:
            in_idx = [vocabs['<s>']] + [vocabs[word] for word in in_tokens] + [vocabs['</s>']]
            out_idx = [vocabs['<s>']] + [vocabs[word] for word in out_tokens] + [vocabs['</s>']]
            
            in_tokens_idxs.append(torch.tensor(in_idx))
            out_tokens_idxs.append(torch.tensor(out_idx[:-1]))
            labels.append(torch.tensor(out_idx[1:]))
        
        in_tokens_idxs = pad_sequence(in_tokens_idxs, batch_first=True)
        out_tokens_idxs = pad_sequence(out_tokens_idxs, batch_first=True)
        labels = pad_sequence(labels, batch_first=True)
        
        return in_tokens_idxs, out_tokens_idxs, labels
    return getVector

def split_data(in_tokens, out_tokens, chunk_size):
    random_idxs = np.random.randint(0, len(in_tokens), chunk_size)
    selected_in_tokens = [in_tokens[i] for i in random_idxs]
    selected_out_tokens = [out_tokens[i] for i in random_idxs]
    return selected_in_tokens, selected_out_tokens

if __name__ == '__main__':
    batch_size = 126
    # os.chdir(os.getcwd() + "/homework")

    # 训练分词
    train_in_tokens = readFile('./data/couplet/train/in.txt', False)
    train_out_tokens = readFile('./data/couplet/train/out.txt', True)
    # 测试分词
    # test_in_tokens = readFile('./data/couplet/test/in.txt', False)
    # test_out_tokens = readFile('./data/couplet/test/out.txt', True)
    
    # 获取词典
    vocabs = getVocab('./data/couplet/vocabs')
    
    # 数据集
    # dataset = list(zip(train_in_tokens, train_out_tokens))
    # dataloader = DataLoader(dataset, batch_size, shuffle=True,collate_fn=collate_fn)
    
    # 保存词典
    with open('./data/proc/vocabs.bin', 'wb') as f:
        pickle.dump(vocabs, f)
        
    # 保存分词
    with open('./data/proc/train_in_words.json', 'w', encoding='utf-8') as f:
        json.dump(train_in_tokens, f)
    with open('./data/proc/train_out_words.json', 'w', encoding='utf-8') as f:
        json.dump(train_out_tokens, f)
        
    # with open('./data/proc/test_in_words.json', 'w', encoding='utf-8') as f:
    #     json.dump(test_in_tokens, f)
    # with open('./data/proc/test_out_words.json', 'w', encoding='utf-8') as f:
    #     json.dump(test_out_tokens, f)