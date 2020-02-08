import torch
from torch.utils.data import Dataset

class TCdata(Dataset):
    def __init__(self, src, tgt):
        super(TCdata, self).__init__()
        self.src = src
        self.tgt = tgt
        
    def __len__(self):
        return len(self.src)
    
    def __getitem__(self, i):
        return self.src[i], self.tgt[i]
    
def collate_func(seqs, pad_token=0):
    seq_lens = [len(seq) for seq in seqs]
    max_len = max(seq_lens)
    seqs = [seq + [pad_token] * (max_len - len(seq)) for seq in seqs]
    return torch.LongTensor(seqs), torch.LongTensor(seq_lens)    
    
def pair_collate_func(inps):
    pairs = sorted(inps, key=lambda p: len(p[0]), reverse=True)
    seqs, tgt = zip(*pairs)
    seqs, seq_lens = collate_func(seqs)
    return seqs, seq_lens, torch.FloatTensor(tgt) 