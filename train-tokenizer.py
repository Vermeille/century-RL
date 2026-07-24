from tokenizers import ByteLevelBPETokenizer, Regex  # type: ignore[import-not-found]
from tokenizers.pre_tokenizers import Split  # type: ignore[import-not-found]

with open("game.txt", "r") as f:
    games = f.read().replace("\n", "")

tokenizer = ByteLevelBPETokenizer()
tokenizer.pre_tokenizer = Split(
    pattern=Regex(r"[\s\n]+|->|@|[AHVR]\d+|\d"),
    behavior="isolated",
)
print([x[0] for x in tokenizer.pre_tokenizer.pre_tokenize_str(games[:2000])])
tokenizer.train_from_iterator(
    [games],
    vocab_size=256 + 70,
    min_frequency=6000,
    special_tokens=["<pad>", "_Him", "_Moves", "_Me", "_Board"],
)
print(tokenizer.encode(games[:2000]).tokens)
print(len(tokenizer.encode(games[:2000]).tokens))
print(sorted(list(tokenizer.get_vocab().items()), key=lambda x: x[1]))
tokenizer.save("tokenizer.json")
