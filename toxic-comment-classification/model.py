import torch
import torch.nn as nn
# import torch.nn.functional as F


class BiGRU(nn.Module):
    def __init__(self, emb_dim, hidden_size, n_layers, num_classes, dropout, weights):
        super(BiGRU, self).__init__()
        self.emb = nn.Embedding.from_pretrained(torch.FloatTensor(weights), freeze=False)
        self.spatial_dropout1d = nn.Dropout2d(p=dropout)
        self.gru1 = nn.GRU(emb_dim, hidden_size, n_layers, bidirectional=True, batch_first=True)
        self.gru2 = nn.GRU(hidden_size*2, hidden_size, n_layers, bidirectional=True, batch_first=True)
        self.maxpooling = nn.AdaptiveMaxPool1d(1)
        self.linear = nn.Linear(4*hidden_size, num_classes)
        # self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        
    def forward(self, input_var, input_len):
        embeded = self.emb(input_var) # b x l x emb_dim
        embeded = self.spatial_dropout1d(embeded.permute(0,2,1)).permute(0,2,1)
        # embeded = self.dropout1(embeded)
        
        total_length = embeded.size(1)
        
        packed1 = torch.nn.utils.rnn.pack_padded_sequence(embeded, input_len, batch_first=True)
        self.gru1.flatten_parameters()
        rnn1, hidden1 = self.gru1(packed1)
        rnn1, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn1, batch_first=True, total_length=total_length) # b x l x 2hs
        
        packed2 = torch.nn.utils.rnn.pack_padded_sequence(rnn1, input_len, batch_first=True)
        self.gru2.flatten_parameters()
        rnn2, hidden2 = self.gru2(packed2)
        rnn2, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn2, batch_first=True, total_length=total_length) # b x l x 2hs
        
        out = torch.cat((rnn1, rnn2), 2) # b x l x 4hs
        out = out.permute(0, 2, 1) # b x 4hs x l
        out = self.maxpooling(out).squeeze(2) # b x 4hs x 1 -> b x 4hs
        out = self.dropout2(out)
        out = self.linear(out)
        return out