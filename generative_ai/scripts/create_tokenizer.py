from datasets import load_dataset
from tokenizers import models, pre_tokenizers, processors, trainers, Tokenizer, decoders
from pathlib import Path

CWD = Path(__file__).parent


def main():
    dataset = load_dataset("bookcorpus", split="train")

    def batch_iterator(batch_size=1000):
        for i in range(0, len(dataset), batch_size):
            yield dataset[i : i + batch_size]["text"]

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=50304,
        special_tokens=["<|startoftext|>", "<|endoftext|>", "<|padding|>"],
    )

    tokenizer.train_from_iterator(
        batch_iterator(), trainer=trainer, length=len(dataset)
    )
    tokenizer.save(f"{CWD.parent}/artifacts/tokenizer.json")


if __name__ == "__main__":
    main()
