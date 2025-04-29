import json
import pickle
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from process_data import collate_fn,split_data
from EncoderDecoderAttenModel import Seq2seq
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os


if __name__ == "__main__":
    
    # 加载训练好的模型
    model_state = torch.load('./data/model/seq2seq_state.bin', weights_only=False)
    
    # 加载分词
    with open('./data/proc/test_in_words.json', 'r') as f:
        in_tokens = json.load(f)
    with open('./data/proc/test_out_words.json', 'r') as f:
        out_tokens = json.load(f)
        
    # 加载词典(汉字：索引)
    with open('./data/proc/vocabs.bin', 'rb') as f:
        vocabs = pickle.load(f)
    
    # (索引：汉字)
    vocabs_cn = [(i, t) for i, t in vocabs.items()]
    
    # 模型
    model = Seq2seq(
        input_dim=len(vocabs),
        emb_dim=200,
        hidden_dim=250,
        dropout = 0.5
    )
    model.load_state_dict(model_state)
        
    # 随机抽取测试样本
    random_idxs = np.random.randint(0, len(in_tokens))
    one_in_tokens = in_tokens[random_idxs]
    one_out_tokens = out_tokens[random_idxs]
    
    # 获取样本对应词典的下标
    in_idxs = [vocabs[token] for token in one_in_tokens]
    out_idxs = [vocabs[token] for token in one_out_tokens] 
    
    # 预测
    with torch.no_grad():
        hidden_out, output = model.encoder(in_idxs)
        
        dec_input = torch.tensor([[vocabs['<s>']]])
        
        dec_tokens = []
        while True:
            if len(dec_tokens) > 50:
                break
            
            # logits:(batch_size,seq_len,vocab_size) = (1, 1, vocab_size)
            logits, l_hidden = model.decoder(dec_input, hidden_out, output)
            
            # next_token (1, 1)
            next_token = torch.argmax(logits, dim=-1)
            # 结束直接跳出
            if vocabs_cn[next_token.squeeze().item()] == '</s>':
                break
            
            dec_tokens.append(next_token.squeeze().item())
            dec_input = next_token
            hidden_state = hidden_state.view(1, -1)
            
    print(f"上联：{"".join(one_in_tokens)}")
    print(f"下联：{"".join([vocabs_cn[tk] for tk in dec_tokens])}")
    print("真实下联：", "".join(one_out_tokens))