
![[Pasted image 20260715140456.png]]

![[Pasted image 20260715140507.png]]

![[Pasted image 20260715140525.png]]


![[Pasted image 20260715140559.png]]

![[Pasted image 20260715140612.png]]

![[Pasted image 20260715140626.png]]

![[Pasted image 20260715140649.png]]


## Concept

Before a language model can process text, it needs a vocabulary: a bidirectional mapping between characters (or tokens) and integers. The model works with integers internally, so we need `encode` to convert text to numbers and `decode` to convert numbers back to text.

The construction process is:

1. **Extract unique characters** from the training text.
2. **Sort them** alphabetically for deterministic ordering.
3. **Build `stoi`** (string-to-integer): assign each character a unique index starting from 0.
4. **Build `itos`** (integer-to-string): the reverse mapping.

This is character-level tokenization. The vocabulary size equals the number of unique characters in the training data, typically 50-100 for English text. Compare this to BPE (50,000+ tokens) or word-level (100,000+ tokens). Character-level vocabularies produce much longer sequences but never encounter out-of-vocabulary tokens (as long as the character appeared in training).

The `encode`/`decode` functions must be inverses: `decode(encode(text)) == text`. This round-trip property is essential. If you cannot perfectly reconstruct the original text, the model cannot learn the correct input-output mapping


## Key Takeaways

- A character-level vocabulary is the simplest tokenization approach, with vocabulary size equal to the number of unique characters in the training data.
- The `stoi`/`itos` pair enables lossless round-trip conversion between text and integer sequences, which is a hard requirement for any tokenizer.
- Sorting the unique characters ensures deterministic ID assignment. Without sorting, the same text could produce different vocabularies across runs.
