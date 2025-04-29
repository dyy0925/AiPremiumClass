import torch.nn as nn
import torch

class Encoder(nn.Module):
    
    # input_dim == max_seq_len
    def __init__(self, input_dim, emb_dim, hidden_dim, dropout):
        super().__init__()
        self.embbed = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(input_size=emb_dim, hidden_size=hidden_dim,
                          batch_first=True, bidirectional=True, dropout=dropout)
        
    def forward(self, seq_tokens):
        # seq_tokens(batch_size, seq_len)
        embbed = self.embbed(seq_tokens)
        # embbed(batch_size, seq_len,emb_dim)
        output, l_hidden = self.rnn(embbed)
        # output(batch_size, seq_len, hidden_dim*2)  l_hidden(2, batch_size, hidden_dim)
        hidden_out = torch.cat((l_hidden[0], l_hidden[1]), dim=-1)
        # hidden_out(batch_size, hidden_dim*2)
        return hidden_out, output
    

class Decoder(nn.Module):
    
    def __init__(self, input_dim, emb_dim, hidden_dim, dropout):
        super().__init__()
        self.embbed = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(input_size=emb_dim, hidden_size=hidden_dim * 2, batch_first=True,
                          dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, input_dim)
        self.attention = Attention()
        self.fc_a = nn.Linear(hidden_dim * 4, hidden_dim * 2)
        
    # enc_hidden_state(batch_size, hidden_dim*2)
    # enc_output(batch_size, seq_len, hidden_dim*2)
        
    def forward(self,seq_tokens, enc_hidden_state, enc_output):
        # seq_tokens(batch_size, seq_len)
        embbed = self.embbed(seq_tokens)
        # embbed(batch_size, seq_len, emb_dim)
        
        # GRU-hidden_state(num_layers * num_directions, batch_size, hidden_size*2)
        # enc_hidden_state.unsqueeze(0) 增加一个维度
        # l_hidden(1, batch_size, hidden_size*2)
        dec_output, l_hidden = self.rnn(embbed, enc_hidden_state.unsqueeze(0))
        
        # Attention -- begin
        # dec_output(batch_size, seq_len, hidden_size*2)
        # enc_output(batch_size, seq_len, hidden_size*2)
        # 关联性权重：内积-归一化-归一化后的和en_output做内积
        c_t = self.attention(enc_output, dec_output)
        # c_t (batch_size, seq_len, hidden_size * 2)
        # 拼接
        cat_output = torch.cat((c_t, dec_output), dim=-1)
        # cat_output(batch_size,hidden_size*4)
        # 线性运算
        a_fc = self.fc_a(cat_output)
        # a_fc(batch_size, hidden_size*4, hidden_size*2)
        # 非线性运算
        out = torch.tanh(a_fc)
        # out: (batch_size,seq_len,hidden_size * 2) 和输入的一致
        # Attention -- end
        
        logits = self.fc(out)
        # logits:(batch_size,seq_len,vocab_size)
        return logits, l_hidden

    
class Attention(nn.Module):
    
    def __init__(self):
        super().__init__()
        
    def forward(self, enc_ouput, dec_output):
        # 内积
        a_t = torch.bmm(enc_ouput, dec_output.permute(0, 2, 1))
        # 归一化
        a_t = torch.softmax(a_t, dim=1)
        # 归一化后的和enc做内积
        c_t = torch.bmm(a_t.permute(0, 2, 1), enc_ouput)
        return c_t
        
    
class Seq2seq(nn.Module):
    
    def __init__(self, input_dim, emb_dim, hidden_dim, dropout):
        super().__init__()
        self.encoder = Encoder(input_dim, emb_dim, hidden_dim, dropout)
        self.decoder = Decoder(input_dim, emb_dim, hidden_dim, dropout)
        
    def forward(self, in_seq_tokens, out_seq_tokens):
        hidden_out, output= self.encoder(in_seq_tokens)
        logits, _ = self.decoder(out_seq_tokens, hidden_out, output)
        return logits
        