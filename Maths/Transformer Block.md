
The **transformer block** is the repeating building block of every large language model. GPT-2 stacks 12 of them, GPT-3 stacks 96. Each block applies two sub-layers with residual connections and layer normalization:

## Transformer Block (Pre-Layer Normalization)

A transformer block uses **residual connections** around both the Multi-Head Attention and Feed-Forward Network.

The computations are:

$$
x = x + \text{MultiHeadAttention}(\text{LayerNorm}(x))
$$

$$
x = x + \text{FeedForward}(\text{LayerNorm}(x))
$$

### Explanation

1. Apply **Layer Normalization** to the input.
2. Pass the normalized input through the **Multi-Head Attention** layer.
3. Add the attention output back to the original input using a **residual (skip) connection**.
4. Apply **Layer Normalization** again.
5. Pass the result through the **Feed-Forward Network (FFN)**.
6. Add the FFN output back using another **residual connection**.

> [!note]
> This is known as the **Pre-Layer Normalization (Pre-LN)** transformer architecture because Layer Normalization is applied **before** each sublayer. Most modern transformer models (including GPT-2, GPT-3, LLaMA, and others) use this design because it improves training stability, especially for very deep networks.


## Pre-Norm Transformer Architecture

The equations for a **Pre-Norm Transformer Block** are:

$$
x = x + \text{MultiHeadAttention}(\text{LayerNorm}(x))
$$

$$
x = x + \text{FeedForward}(\text{LayerNorm}(x))
$$

Unlike the original **Post-Norm** architecture introduced in *Attention Is All You Need*, the **Pre-Norm** architecture applies **Layer Normalization before each sub-layer**. This design has become the standard in modern transformer models because it provides significantly more stable training, especially for deep networks.

---

## Residual Connections

Residual (or **skip**) connections are one of the key reasons deep transformers can be trained successfully.

A residual connection has the form:

$$
x + f(x)
$$

where:

- \(x\) is the input.
- \(f(x)\) is the output of a sub-layer (such as Multi-Head Attention or the Feed-Forward Network).

Without residual connections, gradients must propagate through dozens (or even hundreds) of layers, causing them to shrink exponentially. The skip connection provides a direct path for gradients during backpropagation, making optimization much easier.

Another way to think about residual connections is that the sub-layer only needs to learn **how the output differs from the input**, rather than learning the entire transformation from scratch.

> [!tip]
> Residual connections improve gradient flow, speed up convergence, and enable very deep transformer architectures.

---

## Feed-Forward Network (FFN)

After the attention layer, each token is processed independently by a **Feed-Forward Network (FFN)**.

The FFN is a two-layer multilayer perceptron (MLP):

$$
\text{FFN}(x)
=
\text{Dropout}
\left(
\text{ReLU}(xW_1+b_1)W_2+b_2
\right)
$$

### How It Works

1. Apply a linear transformation using \(W_1\).
2. Increase the feature dimension (typically by **4×**).
3. Apply the **ReLU** activation function.
4. Project the features back to the original model dimension using \(W_2\).
5. Apply dropout for regularization.

For example, in **GPT-2**:

- Input dimension: **768**
- Hidden dimension: **3072** (4× expansion)
- Output dimension: **768**

This temporary expansion gives the network more capacity to learn complex transformations for each token.

### Role of the FFN

The transformer block separates computation into two complementary parts:

- **Multi-Head Attention**
  - Enables communication **between different tokens**.
  - Learns contextual relationships across the sequence.

- **Feed-Forward Network**
  - Operates **independently on each token**.
  - Learns richer feature representations for individual token embeddings.

> [!summary]
> - **Pre-Norm** applies Layer Normalization before each sub-layer, improving training stability.
> - **Residual connections** (\(x + f(x)\)) provide a direct path for gradients, enabling very deep transformers.
> - The **Feed-Forward Network (FFN)** is a two-layer MLP applied independently to every token.
> - The FFN temporarily expands the hidden dimension (typically by **4×**) before projecting back to the original model dimension, giving the model greater representational capacity.



## Key Takeaways

- A transformer block combines attention (inter-token communication) with an FFN (per-token computation), connected by residual paths and layer normalization.
- Residual connections enable deep networks by providing gradient highways that bypass the sub-layers.
- The 4x expansion in the FFN gives the network additional parameter capacity. In GPT-2, the FFN contains two-thirds of each block's parameters.



![[Pasted image 20260714212514.png]]


![[Pasted image 20260714213040.png]]

![[Pasted image 20260714213132.png]]

![[Pasted image 20260714213148.png]]

![[Pasted image 20260714213216.png]]

![[Pasted image 20260714213234.png]]

![[Pasted image 20260714213246.png]]

