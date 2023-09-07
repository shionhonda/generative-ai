"""
Copied and modified from https://github.com/karpathy/nanoGPT/blob/master/model.py
Copyright (c) 2022 Andrej Karpathy
https://github.com/karpathy/nanoGPT/blob/master/LICENSE
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class GPT(nn.Module):
    def __init__(
        self,
        block_size: int,
        vocab_size: int,
        n_layer: int,
        n_head: int,
        n_embd: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.backbone = nn.ModuleList(
            [
                GPTBlock(n_head=n_head, n_embd=n_embd, dropout=dropout)
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight  # weight tying

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        n_params -= self.wpe.weight.numel()
        return n_params

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> (torch.Tensor, torch.Tensor | None):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        tok_emb = self.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.wpe(pos)  # position embeddings of shape (t, n_embd)
        x = self.drop(tok_emb + pos_emb)
        for block in self.backbone:
            x = block(x)
        x = self.ln_f(x)  # (b, t, n_embd)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        idx_eot: int = 1,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx if idx.size(1) <= self.block_size else idx[:, -self.block_size :]
            )
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if idx_next == idx_eot:
                break
        return idx


class GPTBlock(nn.Module):
    def __init__(self, n_head: int, n_embd: int, dropout: float) -> None:
        super().__init__()
        self.prenorm_attn = nn.LayerNorm(n_embd)
        self.attn = CausalMultiHeadSelfAttention(
            n_head=n_head, n_embd=n_embd, dropout=dropout
        )
        self.prenorm_mlp = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd=n_embd, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.prenorm_attn(x))
        x = x + self.mlp(self.prenorm_mlp(x))
        return x


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, n_head: int, n_embd: int, dropout: float) -> None:
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (
            B,
            T,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # using Flash Attention CUDA kernels for efficiency
        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=True,
        )
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float) -> None:
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
