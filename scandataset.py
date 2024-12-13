"""
    The SCANDataset class is a custom pytorch Dataset designed to handle the SCAN dataset,
    allowing for efficient loading and preprocessing of input-output pairs.
"""
import os
from torch.utils.data import Dataset
import torch

class SCANDataset(Dataset):
    """
        The class initializes with parameters for data directory, vocabularies, split type,
        maximum sequence length, and padding indices, and provides methods for retrieving
        tokenized and padded sequences as tensors.
    """
    def __init__(
        self, 
        file_path,
        src_vocab, 
        tgt_vocab, 
        max_len=50, 
        transform=None
    ):
        """
        Args:
            root_dir (string): Directory with the SCAN dataset.
            split (string): Which split to use (e.g., 'simple_split', 'length_split').
            max_len (int): Maximum sequence length for input and output.
            src_pad_idx (int): Padding token index for input sequences.
            tgt_pad_idx (int): Padding token index for output sequences.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.transform = transform
        self.max_len = max_len
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

        # Parse src-tgt pairs from all files
        self.data = []
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            src, tgt = line.strip().split(" OUT: ")
            src = src.replace("IN: ", "").split()
            tgt = tgt.split()
            # transforming the input and output into tokenized sequences
            self.data.append({"src": src, "tgt": tgt})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx]
        src = sample['src']
        tgt = sample['tgt']

        # replacing tokens with their corresponding indices in the vocab
        src = [self.src_vocab[token] for token in src]
        tgt = [self.tgt_vocab[token] for token in tgt]
        
        assert len(src) <= self.max_len, f"A sequence exceeded max_len of {self.max_len}"

        # special tokens
        src = [self.src_vocab['<BOS>']] + src + [self.src_vocab['<EOS>']]
        src += [self.src_vocab['<PAD>']] * (self.max_len - len(src))

        tgt = [self.tgt_vocab['<BOS>']] + tgt + [self.tgt_vocab['<EOS>']]
        tgt += [self.tgt_vocab['<PAD>']] * (self.max_len - len(tgt))

        sample = {
            'src': torch.tensor(src, dtype=torch.long),
            'tgt': torch.tensor(tgt, dtype=torch.long)
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
