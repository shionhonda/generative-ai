from generative_ai.models import GPT
from train import CFG
import torch
from tokenizers import Tokenizer
from pathlib import Path

CWD = Path(__file__).parent


def main() -> None:
    # system settings
    device = "cpu"
    tokenizer = Tokenizer.from_file(f"{CWD.parent}/artifacts/tokenizer.json")
    model = GPT(
        block_size=CFG.block_size,
        vocab_size=tokenizer.get_vocab_size(),
        n_layer=CFG.n_layer,
        n_head=CFG.n_head,
        n_embd=CFG.n_embd,
        dropout=CFG.dropout,
    )

    ckpt = torch.load(f"{CWD.parent}/artifacts/ckpt_step15000.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    model.to(device)
    idx = model.generate(
        torch.LongTensor([[0, 6251, 221, 288]]).to(device),  # prompt: life is about
        max_new_tokens=60,
    )
    sentence = tokenizer.decode(idx.squeeze().detach().cpu().tolist())
    print(sentence)


if __name__ == "__main__":
    main()
