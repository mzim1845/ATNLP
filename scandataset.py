import os
from torch.utils.data import Dataset
import torch

class SCANDataset(Dataset):
    def __init__(
        self, 
        data,
        src_vocab, 
        tgt_vocab, 
        max_len=50
    ):
        self.max_len = max_len
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx]
        src = sample['src']
        tgt = sample['tgt']

        src = [self.src_vocab[token] for token in src]
        tgt = [self.tgt_vocab[token] for token in tgt]
        
        assert len(src) <= self.max_len, f"A sequence exceeded max_len of {self.max_len}"

        src = [self.src_vocab['<BOS>']] + src + [self.src_vocab['<EOS>']]
        src += [self.src_vocab['<PAD>']] * (self.max_len - len(src))

        tgt = [self.tgt_vocab['<BOS>']] + tgt + [self.tgt_vocab['<EOS>']]
        tgt += [self.tgt_vocab['<PAD>']] * (self.max_len - len(tgt))

        sample = {
            'src': torch.tensor(src, dtype=torch.long),
            'tgt': torch.tensor(tgt, dtype=torch.long)
        }

        return sample
