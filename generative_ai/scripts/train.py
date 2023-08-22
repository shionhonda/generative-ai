from generative_ai.models import GPT


gpt = GPT(
    block_size=512, vocab_size=50304, n_layer=8, n_head=8, n_embd=512, dropout=0.1
)
