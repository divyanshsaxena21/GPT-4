
You know how to embed words and encode their positions. But the Bag of Words model you built for sentiment analysis throws away word order; it just averages all embeddings. Real language understanding requires knowing which words relate to which. This is the core mechanism: **self-attention** is what makes transformers work. It's how ChatGPT decides that in "The animal didn't cross the street because **it** was too tired," the word "it" refers to "animal" and not "street."

![[Pasted image 20260713202548.png]]

![[Pasted image 20260713202612.png]]

![[Pasted image 20260713205923.png]]


![[Pasted image 20260713205943.png]]

![[Pasted image 20260713210004.png]]

![[Pasted image 20260713210022.png]]

The core computation is:

## Scaled Dot-Product Attention

The core computation of the attention mechanism involves creating three matrices from the input \(X\):

- **Query (Q)**
- **Key (K)**
- **Value (V)**

These are computed using learned weight matrices:

$Q = XW_Q$

$K = XW_K$

$V = XW_V$

The attention output is then calculated as:

$\text{Attention}(Q, K, V)=\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$

### Why divide by $(\sqrt{d_k})$?

The scaling factor $(\sqrt{d_k})$ prevents the dot-product values from becoming excessively large as the dimensionality of the key vectors increases. Without this scaling, the softmax function can become too peaked, leading to vanishing gradients and unstable training.

### Causal Mask (GPT-style Models)

For **autoregressive transformer models** (such as GPT), each token should only attend to:

- Itself
- Previous tokens

To enforce this, a **causal mask** is applied before the softmax operation.

- Future positions are assigned a value of **\(-\infty\)**.
- After applying softmax, these positions receive a probability of **0**, preventing attention to future tokens.

The masked attention equation becomes:

$\text{Attention}(Q, K, V)=\text{softmax}\left(\frac{QK^T + \text{Mask}}{\sqrt{d_k}}\right)V$

where

$\text{Mask}_{ij}=\begin{cases} 0, & j \le i \\-\infty, & j > i \end{cases}$

> [!note]
> The causal mask ensures that during both training and inference, a token cannot access information from future tokens, preserving the autoregressive property of GPT-style language models.



![Scaled dot-product attention](https://neetcode.io/assets/ml/attention-matrix-calc.png)



In a **causal** (decoder-only) transformer like GPT, a lower-triangular mask is applied before softmax: future positions are set to −∞−∞, making their softmax weights zero. This ensures each token can only attend to itself and earlier tokens, which is essential for autoregressive generation where the model predicts one token at a time.


## Key Takeaways

- **Self-attention** computes pairwise similarity between all tokens using the dot product:

	$QK^T$

  These similarity scores are then used to compute a **weighted average of the value vectors $((V))$**, allowing each token to gather relevant information from other tokens in the sequence.

- **Scaling by $(\sqrt{d_k})$ is essential.** Without this scaling, the dot-product values become very large, causing the softmax distribution to become overly sharp. As a result, a single token may receive almost all the attention weight, making training unstable and preventing the model from learning smooth attention patterns.

- **Causal masking** ensures **autoregressive behavior**. The output for token \(t\) depends only on tokens from position \(0\) through \(t\). Future tokens are masked out before applying softmax, making left-to-right text generation possible in GPT-style models.

> [!summary]
> - Self-attention measures token similarity using $(QK^T)$.
> - The resulting attention weights determine how much each value vector contributes to the output.
> - Scaling by $(\sqrt{d_k})$ stabilizes training by preventing excessively large attention scores.
> - Causal masking blocks access to future tokens, enabling autoregressive text generation.

