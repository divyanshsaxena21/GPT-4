

Your vocabulary encoder turns text into a long sequence of integers. Now you need to slice that sequence into training examples. The key insight: a language model's training data is just the same sequence shifted by one position. If the input is `[t1, t2, t3]`, the target is `[t2, t3, t4]`. The model learns to predict each next token given the tokens before it. This loader takes **pre-tokenized integer sequences** and produces batches of these (input, target) pairs, ready for training.


![[Pasted image 20260718141706.png]]

![[Pasted image 20260718141725.png]]

![[Pasted image 20260718141848.png]]

![[Pasted image 20260718142017.png]]


## Concept

A GPT data loader creates batches of input-target pairs from a flat sequence of token IDs. The model learns by next-token prediction: given a context of tokens, predict what comes next. So we need overlapping windows where the target is the input shifted by one position.

Given an encoded text (a 1D tensor of token IDs), a **context length** $C$, and a **batch size** $B$:

1. **Sample** BB random starting positions from range $[0,len(data)−C)$.
2. For each position $i$:
    ## Input and Target Sequences

- **Input (`x`)**: Tokens `i` to `i + C - 1` (length `C`).
- **Target (`y`)**: Tokens `i + 1` to `i + C` (length `C`).

> [!note]
> The target sequence is the input sequence shifted **one token forward**, which is the standard setup for next-token prediction in language models.

The target is the input shifted right by one. At every position, the model is asked: "given everything up to here, what comes next?" This means a single context window of length $C$ actually provides $C$ training examples. A batch of $B$ windows provides $B×C$ training examples, which is highly efficient.

Random starting positions ensure the model sees different slices of the training data each iteration, acting as a form of data augmentation for language modeling.



## Common Pitfalls

### Off-by-One Error in the Random Range

If you sample the starting index from:

```text
[0, len(data))
```

instead of:

```text
[0, len(data) - C)
```

then starting positions near the end of `data` can cause an **index-out-of-bounds** error when extracting the target window.

> [!warning]
> The target sequence is shifted by one token relative to the input. Therefore, you must leave enough room for both the input window (`C` tokens) **and** the final target token.


## Key Takeaways

- The data loader creates input-target pairs by shifting the input window by one position, which is the standard setup for next-token prediction.
- Random starting positions ensure the model sees diverse training data each iteration, preventing overfitting to a fixed set of windows.
- Each context window of length $C$ provides $C$ independent training examples, making language model training data-efficient.





