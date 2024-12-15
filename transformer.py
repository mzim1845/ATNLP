import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super().__init__()

        assert emb_dim % num_heads == 0, "MultiHeadAttention: Embedding dimension must be divisible by number of heads."

        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads

        # emb_dim = num_heads * self.head_dim
        self.query_linear = nn.Linear(emb_dim, emb_dim)
        self.key_linear = nn.Linear(emb_dim, emb_dim)
        self.value_linear = nn.Linear(emb_dim, emb_dim)

        self.out_linear = nn.Linear(emb_dim, emb_dim)

    def forward(self, query, key, value, mask=None):

        assert query.ndim == 3, "MultiHeadAttention: Expected input to have shape (batch_size, seq_len, emb_dim)"
        assert key.ndim == 3, "MultiHeadAttention: Expected input to have shape (batch_size, seq_len, emb_dim)"
        assert value.ndim == 3, "MultiHeadAttention: Expected input to have shape (batch_size, seq_len, emb_dim)"

        assert key.shape == value.shape, "MultiHeadAttention: Expected key and value to have same shape"

        batch_size, seq_len, _ = query.size()

        Q = self.query_linear(query)
        K = self.key_linear(key)
        V = self.value_linear(value)

        def split_heads(x):
            x = x.view(batch_size, -1, self.num_heads, self.head_dim)
            return x.permute(0, 2, 1, 3) # (batch_size, num_heads, seq_len, head_dim)

        Q = split_heads(Q)
        K = split_heads(K)
        V = split_heads(V)

        # QK^T / sqrt(head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # (batch_size, num_heads, seq_len, seq_len)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.emb_dim)

        return self.out_linear(attn_output)

class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout, forward_dim):
        super().__init__()

        self.mha = MultiHeadAttention(emb_dim, num_heads)
        self.dropout = nn.Dropout(dropout)
        self.layernorm1 = nn.LayerNorm(emb_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(emb_dim, eps=1e-6)

        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, forward_dim),
            nn.ReLU(),
            nn.Linear(forward_dim, emb_dim)
        )

    def forward(self, query, key, value, mask):
        
        assert key.shape == value.shape, "TransformerBlock: Expected key and value to have same shape"

        mha_output = self.mha(query, key, value, mask)  # (batch_size, seq_len, emb_dim)
        skip_connection1 = self.dropout(mha_output) + query
        normed1 = self.layernorm1(skip_connection1)

        ffn_output = self.ffn(normed1)  # (batch_size, seq_len, emb_dim)
        skip_connection2 = self.dropout(ffn_output) + normed1
        normed2 = self.layernorm2(skip_connection2)

        return normed2
    
def get_sinusoid_table(max_len, emb_dim):
    def get_angle(pos, i, emb_dim):
        return pos / 10000 ** ((2 * (i // 2)) / emb_dim)

    sinusoid_table = torch.zeros(max_len, emb_dim)
    for pos in range(max_len):
        for i in range(emb_dim):
            if i % 2 == 0:
                sinusoid_table[pos, i] = math.sin(get_angle(pos, i, emb_dim))
            else:
                sinusoid_table[pos, i] = math.cos(get_angle(pos, i, emb_dim))
    return sinusoid_table

class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim,
        num_layers,
        num_heads,
        forward_dim,
        dropout,
        max_len,
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, emb_dim)

        sinusoid_table = get_sinusoid_table(max_len + 1, emb_dim)  # Shift by +1 for [PAD]
        self.position_embedding = nn.Embedding.from_pretrained(sinusoid_table, freeze=True)

        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(emb_dim, num_heads, dropout, forward_dim)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask):
        token_embeddings = self.token_embedding(x)  # (batch_size, seq_len, emb_dim)

        positions = torch.arange(1, x.size(1) + 1, device=x.device).unsqueeze(0).repeat(x.size(0), 1)  # (batch_size, seq_len)
        position_embeddings = self.position_embedding(positions)

        x = self.dropout(token_embeddings + position_embeddings)

        for layer in self.layers:
            x = layer(x, x, x, mask)

        return x

class DecoderBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, forward_dim, dropout):
        super().__init__()

        self.mha = MultiHeadAttention(emb_dim, num_heads)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(emb_dim, eps=1e-6)

        self.transformer_block = TransformerBlock(emb_dim, num_heads, dropout, forward_dim)

    def forward(self, x, value, key, src_mask, tgt_mask):

        assert key.shape == value.shape, "DecoderBlock: Expected key and value to have same shape"

        mha_output = self.mha(x, x, x, tgt_mask)  # (batch_size, tgt_seq_len, emb_dim)
        skip_connection = self.dropout(mha_output) + x
        normed = self.layernorm(skip_connection)

        return self.transformer_block(normed, key, value, src_mask)

class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim,
        num_layers,
        num_heads,
        forward_dim,
        dropout,
        max_len
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        self.position_embedding = nn.Embedding(max_len, emb_dim)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            DecoderBlock(emb_dim, num_heads, forward_dim, dropout)
            for _ in range(num_layers)
        ])

        self.output_layer = nn.Linear(emb_dim, vocab_size)

    def forward(self, x, encoder_out, src_mask, tgt_mask):
        token_embeddings = self.token_embedding(x)  # (batch_size, tgt_seq_len, emb_dim)

        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0).repeat(x.size(0), 1)  # (batch_size, tgt_seq_len)
        position_embeddings = self.position_embedding(positions)  # (batch_size, tgt_seq_len, emb_dim)

        x = self.dropout(token_embeddings + position_embeddings)

        for layer in self.layers:
            x = layer(x, encoder_out, encoder_out, src_mask, tgt_mask)

        return self.output_layer(x)  # (batch_size, tgt_seq_len, vocab_size)

class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        src_pad_idx,
        tgt_pad_idx,
        emb_dim=512,
        num_layers=6,
        num_heads=8,
        forward_dim=2048,
        dropout=0.0,
        max_len=128,
    ):
        super().__init__()

        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            emb_dim=emb_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            forward_dim=forward_dim,
            dropout=dropout,
            max_len=max_len,
        )

        self.decoder = Decoder(
            vocab_size=tgt_vocab_size,
            emb_dim=emb_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            forward_dim=forward_dim,
            dropout=dropout,
            max_len=max_len,
        )

    def create_src_mask(self, src):
        device = src.device

        # (batch_size, 1, 1, src_seq_len)
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        return src_mask.to(device)

    def create_tgt_mask(self, tgt):
        device = tgt.device
        batch_size, tgt_len = tgt.shape
        
        tgt_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2)

        tgt_mask = tgt_mask * torch.tril(torch.ones((tgt_len, tgt_len))).expand(
            batch_size, 1, tgt_len, tgt_len
        ).to(device)

        return tgt_mask

    def forward(self, src, tgt):
        src_mask = self.create_src_mask(src)
        tgt_mask = self.create_tgt_mask(tgt)

        encoder_out = self.encoder(src, src_mask)

        return self.decoder(tgt, encoder_out, src_mask, tgt_mask)

    def predict(self, src, start_token, end_token, max_len=128):
        batch_size = src.size(0)
        
        src_mask = self.create_src_mask(src)
        encoder_out = self.encoder(src, src_mask)
        
        tgt = torch.full((batch_size, max_len), self.tgt_pad_idx, dtype=torch.long, device=src.device)
        tgt[:, 0] = start_token

        active_sequences = torch.ones(batch_size, dtype=torch.bool, device=src.device)

        for i in range(1, max_len):
            tgt_mask = self.create_tgt_mask(tgt[:, :i])
            decoder_out = self.decoder(tgt[:, :i], encoder_out, src_mask, tgt_mask)

            next_token_logits = decoder_out[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)

            tgt[active_sequences, i] = next_token[active_sequences]
        
            active_sequences = active_sequences & (next_token != end_token)
            
            if not active_sequences.any():
                break

        return tgt
    
    def oracle_len_predict(self, src, start_token, end_token, oracle_len, max_len=128):
        batch_size = src.size(0)
        
        special_tokens = [start_token, end_token, self.tgt_pad_idx]
        
        src_mask = self.create_src_mask(src)
        encoder_out = self.encoder(src, src_mask)
        
        tgt = torch.full((batch_size, max_len), self.tgt_pad_idx, dtype=torch.long, device=src.device)
        tgt[:, 0] = start_token

        for i in range(1, max_len):
            tgt_mask = self.create_tgt_mask(tgt[:, :i])
            decoder_out = self.decoder(tgt[:, :i], encoder_out, src_mask, tgt_mask)

            next_token_logits = decoder_out[:, -1, :]

            sorted_indices = torch.argsort(next_token_logits, dim=-1, descending=True)
            next_token = sorted_indices[:, 0]
            
            for b in range(batch_size):
                if i == oracle_len[b] - 1:
                    next_token[b] = end_token
                elif i < oracle_len[b]:
                    valid_token_found = False
                    for candidate in sorted_indices[b]:
                        if candidate not in special_tokens:
                            next_token[b] = candidate
                            valid_token_found = True
                            break
                    if not valid_token_found:
                        next_token[b] = end_token
                        assert False
                else:
                    next_token[b] = self.tgt_pad_idx
            
            tgt[:, i] = next_token
            
            if (next_token == end_token).all():
                break

        return tgt
