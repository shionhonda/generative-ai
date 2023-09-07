from generative_ai.models import GPT
from train import CFG
import torch
from tokenizers import Tokenizer
from pathlib import Path
import argparse

CWD = Path(__file__).parent
DEVICE = "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="Arguments for generation")
    parser.add_argument("-m", "--model", required=True, type=str)
    parser.add_argument("-p", "--prompt", default="", type=str)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(f"{CWD.parent}/artifacts/tokenizer.json")
    model = load_model(args.model, tokenizer.get_vocab_size())

    input_ids = tokenizer.encode("<|startoftext|>" + args.prompt).ids
    x = model.generate(
        torch.LongTensor([input_ids]).to(DEVICE),
        max_new_tokens=60,
    )
    sentence = tokenizer.decode(x.squeeze().detach().cpu().tolist())
    print(sentence)


def load_model(model_path: str, vocab_size: int) -> torch.nn.Module:
    model = GPT(
        block_size=CFG.block_size,
        vocab_size=vocab_size,
        n_layer=CFG.n_layer,
        n_head=CFG.n_head,
        n_embd=CFG.n_embd,
        dropout=CFG.dropout,
    )

    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)
    return model


if __name__ == "__main__":
    main()
