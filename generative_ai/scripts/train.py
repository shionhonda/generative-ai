from generative_ai.models import GPT
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from contextlib import nullcontext
from datasets import load_dataset
from tokenizers import Tokenizer
from pathlib import Path
from dataclasses import dataclass

CWD = Path(__file__).parent


@dataclass
class CFG:
    # 51M params
    block_size = 64
    n_layer = 8
    n_head = 8
    n_embd = 512
    dropout = 0.0
    batch_size = 128
    lr = 1e-3
    weight_decay = 0.00


class GPTDataset(Dataset):
    def __init__(self, tokenizer: Tokenizer, max_length: int) -> None:
        super().__init__()
        self.data = load_dataset("bookcorpus", split="train")  # 1.0B tokens
        tokenizer.enable_padding(length=max_length + 1)  # add buffer for a single token
        tokenizer.enable_truncation(max_length=max_length)
        self.tokenizer = tokenizer

    def __getitem__(self, index) -> (torch.Tensor, torch.Tensor):
        sentence = self.data[index]["text"]
        input_ids = self.tokenizer.encode(sentence).ids
        x = torch.LongTensor(input_ids[:-1])
        y = torch.LongTensor(input_ids[1:])
        return x, y

    def __len__(self) -> int:
        return len(self.data)


def main() -> None:
    # system settings
    device = "cpu"
    dtype = (
        "bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float16"
    )
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    device_type = "cuda" if "cuda" in device else "cpu"
    cast_ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )

    tokenizer = Tokenizer.from_file(f"{CWD.parent}/artifacts/tokenizer.json")

    model = GPT(
        block_size=CFG.block_size,
        vocab_size=tokenizer.get_vocab_size(),
        n_layer=CFG.n_layer,
        n_head=CFG.n_head,
        n_embd=CFG.n_embd,
        dropout=CFG.dropout,
    )
    model.to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )
    if device_type == "cuda":
        model = torch.compile(model)

    dataset = GPTDataset(tokenizer=tokenizer, max_length=CFG.block_size)
    dataloader = DataLoader(dataset=dataset, batch_size=CFG.batch_size, shuffle=True)

    pbar = tqdm(dataloader)
    for step, (X, Y) in enumerate(pbar):
        if step % 10 == 0:
            idx = model.generate(torch.LongTensor([[0]]), max_new_tokens=8)
            sentence = tokenizer.decode(idx.squeeze().detach().cpu().tolist())
            print(sentence)

        X, Y = X.to(device), Y.to(device)
        with cast_ctx:
            _, loss = model(X, Y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        pbar.set_postfix_str(f"loss={loss.item():.2E}")


if __name__ == "__main__":
    main()
