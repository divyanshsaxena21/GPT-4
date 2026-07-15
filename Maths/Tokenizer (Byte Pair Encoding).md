
In the NLP Intro problem, we tokenized text by splitting on spaces (one word, one token). But what happens with rare words like "unforgettable" or code like `getElementById`? The model has never seen them before! **Byte Pair Encoding** (BPE) solves this by learning a vocabulary of subword tokens. GPT, LLaMA, and most modern LLMs use BPE tokenization.


## GPT Pipeline

The GPT pipeline consists of five main stages:

```text
Tokenize → Data → Model → Train → Generate
```

### 1. Tokenize

Convert raw text into smaller units called **tokens** that the model can process.

Example:

```
"Hello world"
```

becomes:

```
["Hello", "world"]
```

These tokens are then converted into numerical IDs that can be used by the neural network.

---

### 2. Data

Prepare and organize tokenized text into training examples.

This stage includes:

- Collecting large text datasets.
- Cleaning and filtering data.
- Creating input-target pairs for next-token prediction.

Example:

Input:

```
The cat sat on the
```

Target:

```
mat
```

The model learns to predict the next token given previous tokens.

---

### 3. Model

The transformer model processes token sequences using:

- Token embeddings
- Positional embeddings
- Multi-Head Self-Attention
- Feed-Forward Networks
- Layer Normalization
- Residual connections

The model transforms token representations into probability distributions over possible next tokens.

---

### 4. Train

During training, the model learns by repeatedly predicting the next token.

The process involves:

1. Forward pass:
   - Input tokens are passed through the transformer.
   - The model predicts probabilities for the next token.

2. Loss calculation:
   - Compare predictions with the actual next tokens.

3. Backpropagation:
   - Compute gradients.
   - Update model parameters using an optimizer.

The goal is to minimize the prediction error over a large dataset.

---

### 5. Generate

During inference, the trained model generates text autoregressively:

1. Start with a prompt.
2. Predict the next token.
3. Append the predicted token to the sequence.
4. Repeat until completion.

Example:

```
Input:
"The capital of France is"

Model predicts:
"Paris"
```

> [!summary]
> A GPT model follows a pipeline:
>
> **Tokenize → Data → Model → Train → Generate**
>
> Text is converted into tokens, used to train a transformer model through next-token prediction, and finally used to generate new text one token at a time.


Implement the BPE **training** algorithm:

1. Split the corpus into individual characters
2. Count the frequency of every adjacent pair of tokens
3. Merge the most frequent pair into a new token (break ties by choosing the lexicographically smallest pair)
4. Replace all non-overlapping occurrences of that pair (scanning left to right)
5. Repeat for `num_merges` iterations


![[Pasted image 20260715135319.png]]

![[Pasted image 20260715135455.png]]

![[Pasted image 20260715135528.png]]



## Concept

Byte Pair Encoding (BPE) is the tokenization algorithm used by GPT-2, GPT-3, and most modern language models. It finds the sweet spot between character-level tokenization (every character is a token, so sequences are very long) and word-level tokenization (every word is a token, so rare words are out-of-vocabulary).

The algorithm starts with individual characters as tokens and iteratively merges the most frequent adjacent pair:

1. Count the frequency of every adjacent token pair.
2. Find the most frequent pair (break ties lexicographically).
3. Merge all non-overlapping occurrences of that pair into a single new token.
4. Repeat for a specified number of merges.

Each merge creates a new token. Common words like "the" quickly become single tokens after a few merges (t+h, th+e). Rare words like "pneumonoultramicroscopicsilicovolcanoconiosis" stay as subword pieces. This is exactly what you want: common patterns get compressed, rare patterns decompose into known parts.

The merge table is the tokenizer's learned vocabulary. GPT-2 uses about 50,000 merges, producing a vocabulary of 50,257 tokens. The encoding process replays the merges in order to tokenize any input text.


## Key Takeaways

- BPE learns subword tokens by greedily merging the most frequent adjacent pairs, achieving a vocabulary that compresses common patterns while decomposing rare words into known pieces.
- Non-overlapping left-to-right merging ensures deterministic, reproducible results.
- BPE eliminates out-of-vocabulary problems: any input can be encoded as a sequence of subword tokens, even words the model has never seen before.
