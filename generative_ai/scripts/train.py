from generative_ai.models import GPT
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from contextlib import nullcontext
from datasets import load_dataset
from tokenizers import Tokenizer
from pathlib import Path

CWD = Path(__file__).parent


class GPTDataset(Dataset):
    def __init__(self, tokenizer: Tokenizer, max_length: int) -> None:
        super().__init__()
        self.data = load_dataset("bookcorpus", split="train")
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

    # 51M params
    # model = GPT(
    #     block_size=512, vocab_size=50304, n_layer=8, n_head=8, n_embd=512, dropout=0.1
    # )
    # 5.3M params
    model = GPT(
        block_size=256,
        vocab_size=tokenizer.get_vocab_size(),
        n_layer=4,
        n_head=4,
        n_embd=256,
        dropout=0.1,
    )
    model.to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))
    optimizer = torch.optim.AdamW(model.parameters(), kr=1e-4, weight_decay=0.02)
    if device_type == "cuda":
        model = torch.compile(model)

    dataset = GPTDataset(tokenizer=tokenizer, max_length=256)
    dataloader = DataLoader(dataset=dataset, batch_size=24, shuffle=True)

    pbar = tqdm(dataloader)
    for step, (X, Y) in enumerate(pbar):
        # if step % 20 == 0:
        # idx = model.generate(torch.LongTensor([[78]]), max_new_tokens=7)
        # sentence = tokenizer.decode(idx.squeeze().detach().cpu().tolist())
        # print(sentence)

        print(tokenizer.decode(X[0].detach().cpu().tolist()))
        print(tokenizer.decode(Y[0].detach().cpu().tolist()))
        print(tokenizer.decode(X[1].detach().cpu().tolist()))
        print(tokenizer.decode(Y[1].detach().cpu().tolist()))

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
