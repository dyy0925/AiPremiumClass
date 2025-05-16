import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

class PositionEncoding(nn.Module):
    def __init__(self, emb_size, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.emb_size = emb_size
        
    def forward(self, token_embedding):
        seq_len = token_embedding.size(1)  # 获取序列长度
        position = torch.arange(seq_len, dtype=torch.float, device=token_embedding.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.emb_size, 2, dtype=torch.float, device=token_embedding.device) * 
                             (-math.log(10000.0) / self.emb_size))
        
        pos_embedding = torch.zeros_like(token_embedding)
        pos_embedding[:, :, 0::2] = torch.sin(position * div_term)
        pos_embedding[:, :, 1::2] = torch.cos(position * div_term)
        
        return self.dropout(token_embedding + pos_embedding)

class Seq2SeqTransformer(nn.Module):
    
    def __init__(self, emb_size, nhead, num_encoder_layers, num_decoder_layers, dropout,
                 enc_voc_size, dec_voc_size):
        super().__init__()
        self.transformer = nn.Transformer(d_model=emb_size,
                                          nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=emb_size * 4,
                                          dropout=dropout,
                                          batch_first=True)
        self.enc_emb = nn.Embedding(enc_voc_size, emb_size)
        self.dec_emb = nn.Embedding(dec_voc_size, emb_size)
        self.PositionEmbedding = PositionEncoding(emb_size=emb_size, 
                                                  dropout=dropout)
        self.predict = nn.Linear(emb_size, dec_voc_size)
    
    def forward(self, src, tgt, src_mask, tgt_mask,src_key_padding_mask, tgt_key_padding_mask,
                memory_key_padding_mask=None):
        enc_embedding = self.enc_emb(src)
        dec_embedding = self.dec_emb(tgt)
        
        enc_p_emb = self.PositionEmbedding(enc_embedding)
        dec_p_emb = self.PositionEmbedding(dec_embedding)
        
        outs = self.transformer(enc_p_emb, dec_p_emb, src_mask, tgt_mask, None,
                         src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask)
        return self.predict(outs)

    def encoder(self, src, src_mask):
        src_emb = self.enc_emb(src)
        src_p_emb = self.PositionEmbedding(src_emb)
        return self.transformer.encoder(src_p_emb, src_mask)
    
    def decoder(self, tgt, tgt_mask, memory):
        tgt_emb = self.dec_emb(tgt)
        tgt_p_emb = self.PositionEmbedding(tgt_emb)
        return self.transformer.decoder(tgt_p_emb, memory, tgt_mask)


def dealData(wordList):
    # 构建encoder输入模版  decoder输入模版
    enc_tokens, dec_tokens = [],[]
    for i in range(1, len(wordList)):
        enc = wordList[:i]
        dec = ['<s>'] + wordList[i:] + ['</s>']
        
        enc_tokens.append(enc)
        dec_tokens.append(dec)
    return enc_tokens, dec_tokens
    
def get_proc(vocab):
    def batch_proc(data):
        enc_ids, dec_ids, labels = [],[],[]
        for enc_data,dec_data in data:
            enc_tk_idx = [vocab['<s>']] + [vocab[tk] for tk in enc_data] + [vocab['</s>']]
            dec_tk_idx = [vocab['<s>']] + [vocab[tk] for tk in dec_data] + [vocab['</s>']]
            
            enc_ids.append(torch.tensor(enc_tk_idx))
            dec_ids.append(torch.tensor(dec_tk_idx[:-1]))
            labels.append(torch.tensor(dec_tk_idx[1:]))
            
        enc_input = pad_sequence(enc_ids, batch_first=True)
        dec_input = pad_sequence(dec_ids, batch_first=True)
        target = pad_sequence(labels, batch_first=True)
        
        return enc_input,dec_input,target
    
    return batch_proc

def gennerate_square_subsequent_mask(dim):
    mt = torch.ones((dim,dim))
    # 上三角为0，下三角为1
    mask = torch.tril(mt)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def createMask(src, tgt):
    src_seq_len = src.shape[1]  # 修改为src.shape[1]以适应batch_first=True
    tgt_seq_len = tgt.shape[1]  # 修改为tgt.shape[1]以适应batch_first=True
    
    tgt_mask = gennerate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len)).type(torch.bool)  # 修正src_mask形状
    
    # 填充项取值为True
    src_padding_mask = (src == 0)  # 不再需要transpose，因为batch_first=True
    tgt_padding_mask = (tgt == 0)  # 不再需要transpose，因为batch_first=True
    memory_key_padding_mask = src_padding_mask  # 添加memory_key_padding_mask
    
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask
        
    
if __name__ == '__main__':
    # device = torch.device('cuda')
    
    # 例如：input-人  output-生得以须尽欢，莫使金樽空对月
    corpus = '人生得以须尽欢，莫使金樽空对月'
    tokens = list(corpus)
    
    # 1、数据预处理
    enc_tokens, dec_tokens = dealData(tokens)
    
    # 2、构建词典
    vocabs = {tk:i for i,tk in enumerate(['<pad>'] + ['<s>','</s>'] + tokens)}
    
    # 3、构建小批次数据集
    dataset = list(zip(enc_tokens, dec_tokens))
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True, collate_fn=get_proc(vocabs))
    
    # for ins,out,tg in dataloader:
    #     print(ins.shape)
    #     print(out.shape)
    #     print(tg.shape)
        
    
    EMB_SIZE = 512
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3
    VOC_SIZE = len(vocabs)
    LR = 0.0001
    EPOCHS = 10
    
    # 4、定义模型
    model = Seq2SeqTransformer(emb_size=EMB_SIZE,
                               nhead=4,
                               num_encoder_layers=NUM_ENCODER_LAYERS,
                               num_decoder_layers=NUM_DECODER_LAYERS,
                               dropout=0.1,
                               enc_voc_size=VOC_SIZE,
                               dec_voc_size=VOC_SIZE)
    # model.to(device)
    
    # 5、定义损失函数、优化器
    loss_fn = nn.CrossEntropyLoss()
    optimzer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # 6、训练模型
    for epoch in range(EPOCHS):
        model.train()
        loss_total = 0
        tbar = tqdm(dataloader)
        for enc_input, dec_input, label in tbar:
            # enc_input = enc_input.to(device)
            # dec_input = dec_input.to(device)
            # targets = targets.to(device)
            
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask = createMask(enc_input, dec_input)
            # 前向传播
            logits = model(enc_input, dec_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
            # 展平为 [batch_size * seq_len, vocab_size] 和 [batch_size * seq_len]
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), label.view(-1))

            # 反向传播
            optimzer.zero_grad()
            loss.backward()
            optimzer.step()
            
            loss_total += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {loss_total/len(dataloader)}")
        
torch.save(model.state_dict(), 'transformer.bin')