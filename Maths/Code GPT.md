
# Transformer Forward Pass

The forward pass computes:

$h_0 = \text{TokenEmbed}(x) + \text{PosEmbed}(\text{positions})$

$h_i = \text{TransformerBlock}_i(h_{i-1}), \quad \text{for } i = 1, \ldots, N$

$\text{logits} = h_N \cdot W_{\text{vocab}}$

## Explanation

1. **Token Embeddings**
   - Convert each input token `x` into a dense embedding vector using the token embedding layer.

2. **Positional Embeddings**
   - Add positional embeddings to the token embeddings so the model knows the order of tokens in the sequence.

3. **Transformer Blocks**
   - Pass the resulting hidden representation through `N` transformer blocks:
     $h_i = \text{TransformerBlock}_i(h_{i-1})$

4. **Vocabulary Projection**
   - Project the final hidden states into the vocabulary space:
	  $\text{logits} = h_N \cdot W_{\text{vocab}}$

## `forward()` Output

The provided starter code already implements `TransformerBlock`.

Your `forward()` method should return a tensor with shape:

```text
(batch_size, context_length, vocab_size)
```

where:

- `batch_size` = number of input sequences.
- `context_length` = number of tokens in each sequence.
- `vocab_size` = size of the model's vocabulary.

Each vector along the last dimension contains the **unnormalized logits** for predicting the **next token** at that position.


![[Pasted image 20260718220250.png]]

![[Pasted image 20260718220439.png]]


![[Pasted image 20260718220601.png]]

![[Pasted image 20260718220734.png]]

![[Pasted image 20260718220856.png]]

# GPT (Generative Pre-trained Transformer)

## Concept

GPT (Generative Pre-trained Transformer) assembles everything from the course into a complete language model.

The architecture is:

1. **Token Embeddings**
   - Map each input token ID to a dense vector of size $d_{\text{model}}$ using `nn.Embedding`.

2. **Position Embeddings**
   - Add learned position vectors using a second `nn.Embedding`.
   - Unlike the sinusoidal encoding introduced earlier, GPT uses **learned positional embeddings**.

3. **$N$ Transformer Blocks**
   - Each block applies:
     - Multi-Head Self-Attention (for inter-token communication)
     - A Feed-Forward Network (for per-token computation)
   - These are connected using **residual connections** and **layer normalization**.

4. **Final Layer Normalization**
   - Stabilizes the output of the last transformer block before prediction.

5. **Vocabulary Projection**
   - A linear layer maps from $d_{\text{model}}$ to the vocabulary size, producing **logits** (raw, unnormalized scores) for every possible next token.

## Next-Token Prediction

At each position $t$, the model outputs logits over the vocabulary, predicting the token that should appear at position $t + 1$.

### During Training

- `cross_entropy` internally applies the **softmax** function.
- The model is trained to maximize the probability of the correct next token.

### During Generation

- Apply **softmax** to the logits yourself.
- Sample (or choose) the next token.
- Append it to the sequence.
- Repeat the process autoregressively.

## Causal Masking

The attention layers use **causal masking**, ensuring that position $t$ can only attend to tokens from positions

$$
0, 1, \ldots, t.
$$

This prevents the model from seeing future tokens during prediction, making autoregressive generation possible.

## Scaling GPT

One of the most remarkable properties of the GPT architecture is that it scales extremely well. The architecture remains identical; only the model dimensions increase.

| Model | Hidden Size ($d_{\text{model}}$) | Transformer Blocks | Attention Heads |
|-------|-------------------------------:|-------------------:|----------------:|
| GPT-2 Small | $768$ | $12$ | $12$ |
| GPT-3 | $12288$ | $96$ | $96$ |

The core architecture does not change—only the values of:

- $d_{\text{model}}$
- Number of transformer blocks ($N$)
- Number of attention heads



## Key Takeaways

- GPT composes token embeddings, position embeddings, a stack of transformer blocks (each with $W^O$ output projection in multi-head attention), final normalization, and a vocabulary projection into raw logits.
- Learned position embeddings (rather than sinusoidal) let the model discover its own positional representation during training.
- The same architecture scales from tiny models (this problem) to GPT-3 (175 billion parameters) by increasing the model dimension, number of blocks, and number of heads.
