


![[Pasted image 20260714210953.png]]


![[Pasted image 20260714211012.png]]


![[Pasted image 20260714211201.png]]


A single attention head can only focus on one type of relationship, but language has many simultaneous patterns (subject-verb, adjective-noun, pronoun-reference). **Multi-headed attention** runs several attention heads in parallel, each learning different patterns, then combines them. GPT-3 uses 96 heads working simultaneously.


## Multi-Head Attention

Multi-Head Attention applies multiple attention mechanisms in parallel and then combines their outputs.

The overall computation is:

$\text{MultiHead}(X) =\text{Concat}(\text{head}_1,\ldots,\text{head}_h)W_O$

where each attention head is computed as:

$\text{head}_i=\text{Attention}\left(XW_Q^{(i)},XW_K^{(i)},XW_V^{(i)}\right)$

Here:

- $(W_Q^{(i)})$ is the **query projection matrix** for head $(i)$.
- $(W_K^{(i)})$ is the **key projection matrix** for head $(i)$.
- $(W_V^{(i)})$ is the **value projection matrix** for head $(i)$.
- $(W_O)$ is the **learned output projection matrix** that combines the outputs of all attention heads into a single representation.

> [!note]
> Each attention head learns to focus on different aspects of the input sequence, allowing the model to capture multiple types of relationships simultaneously.

Each head independently learns which tokens to attend to. Their outputs are concatenated and passed through a final linear projection that mixes information across heads.



## Concept: Multi-Head Attention

A **single attention head** can learn only one type of relationship between tokens. **Multi-Head Attention** overcomes this limitation by running several attention heads in parallel, each with its own **Query (Q)**, **Key (K)**, and **Value (V)** projection matrices. This allows each head to specialize in capturing different patterns in the input.

### Head Dimensions

Suppose:

- Total attention dimension = \(d\)
- Number of attention heads = \(h\)

Then each head operates on a smaller dimension:

$\frac{d}{h}$

This means the model **does not increase the overall computation**. Instead, it **splits the same attention budget across multiple heads**.

### Multi-Head Attention Process

1. Create **\(h\)** attention heads, each operating on a feature dimension of $(\frac{d}{h})$.
2. Run scaled dot-product attention independently for each head using its own \(Q\), \(K\), and \(V\) projections.
3. Concatenate the outputs of all heads along the feature dimension.
4. Apply a learned output projection matrix \(W_O\) to combine the information from all heads.

### Why Does Multi-Head Attention Help?

Different attention heads learn different types of relationships within the input sequence.

In practice, researchers have observed that different heads specialize in tasks such as:

- **Syntactic relationships**
  - Attending to the previous word.
  - Identifying grammatical dependencies.

- **Semantic relationships**
  - Attending to the subject or object of a sentence.
  - Capturing long-range contextual meaning.

- **Positional relationships**
  - Focusing on nearby tokens.
  - Learning local context and ordering.

A **single attention head** would need to balance all of these relationships simultaneously, whereas **multiple heads** can specialize independently.

### Output Projection

After concatenating the outputs from all heads, a learned linear projection is applied:

$W_O \in \mathbb{R}^{d \times d}$

This projection learns how to combine information from the different attention heads into a unified representation.

The final output has shape:

$(B,\;T,\;d)$

where:

- \(B\) = Batch size
- \(T\) = Sequence length
- \(d\) = Model (embedding) dimension

> [!summary]
> Multi-Head Attention divides the model's attention capacity across multiple specialized heads. Each head captures different relationships in the input sequence, and their outputs are combined using a learned projection matrix \(W_O\). The final output has the same shape as the input, making Multi-Head Attention a drop-in replacement for single-head attention while providing much richer representations.


