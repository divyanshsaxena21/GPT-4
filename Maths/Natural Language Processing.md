

Word embeddings give each token a dense vector, but before you can look up an embedding you need to decide what counts as a "token." That's **tokenization**: splitting raw text into pieces and assigning each piece an integer ID. This is the very first step in every NLP pipeline: text goes in, a sequence of integers comes out, and those integers index into an embedding table. Here you'll build a simple word-level tokenizer from scratch.

Each unique word gets a unique integer ID (sorted lexicographically, starting at 1). Shorter sentences are padded with 0s so the output is a rectangular tensor of shape $(2N,T)$, where $N$ is the number of samples per class and $T$ is the max sentence length.

![[Pasted image 20260712211142.png]]


![[Pasted image 20260712211202.png]]

![[Pasted image 20260712211232.png]]



![[Pasted image 20260712211245.png]]




## Concept

Neural networks cannot process raw text. They need numbers. NLP preprocessing converts strings into numerical tensors through a pipeline:

1. **Tokenization**: Split text into individual tokens (words in this case, though production systems use subword tokenizers like BPE).
2. **Vocabulary construction**: Collect all unique words, sort them, and assign each a unique integer ID starting from 1. We reserve 0 for padding.
3. **Encoding**: Replace each word with its integer ID.
4. **Padding**: Sentences have different lengths, but tensors must be rectangular. Shorter sequences are padded with zeros on the right so all sequences in a batch have the same length.

Why start IDs at 1? Because 0 is reserved as the **padding token**. Later, the model can use a mask to ignore these padding positions, so they do not affect attention scores or loss computation.

PyTorch provides `nn.utils.rnn.pad_sequence` to handle step 4. With `batch_first=True`, it returns a tensor of shape $(N,T)$ where $N$ is the number of sentences and $T$ is the length of the longest sentence.



## Key Takeaways

- NLP preprocessing converts variable-length strings into fixed-size numerical tensors that neural networks can process.
- Vocabulary IDs start at 1 so that 0 serves as a padding token, allowing models to mask and ignore padding positions.
- Sorting the vocabulary ensures deterministic ID assignment, which is critical for reproducibility and testing.




