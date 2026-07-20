

Your GPT generates text, but it's slow. Each new token recomputes attention for ALL previous tokens. KV-Cache fixes this.

![[Pasted image 20260720194705.png]]

![[Pasted image 20260720194725.png]]

![[Pasted image 20260720194801.png]]

![[Pasted image 20260720194829.png]]


**The Problem: Quadratic Attention**

During autoregressive generation, each new token runs a full forward pass. Inside each attention layer, the model computes Q, K, V from the _entire_ context:

```python
# Without cache: token 10 recomputes K, V for tokens 1-9
q = W_q @ context      # (1, 10, d)
k = W_k @ context      # (1, 10, d)  <-- recomputed from scratch!
v = W_v @ context      # (1, 10, d)  <-- recomputed from scratch!
scores = q @ k.T        # (1, 10, 10)
output = softmax(scores) @ v
```

For each new token, the total work is O(N2)O(N2) because Q, K, V are computed for all NN tokens, and the attention matrix is N×NN×N. Generating 100 tokens means recomputing K, V a total of 1+2+3+⋯+100=50501+2+3+⋯+100=5050 times across all steps.

**The Fix: Cache K and V**

The key insight: K and V for previous tokens **don't change** between generation steps. Only the new token's K and V need computing. Instead of recomputing all NN key and value vectors, you store (cache) the ones from previous steps and append only the new token's K and V.

This turns the attention computation from O(N2)O(N2) per step to O(N)O(N): the new query only needs to attend over the growing cache, not recompute everything.

**Worked Example -- Cache Growing Step by Step:**

|Step|New token|Cache K shape|Work|
|---|---|---|---|
|1|"The"|(1, 1, d)|1|
|2|" cat"|(1, 2, d)|2|
|3|" sat"|(1, 3, d)|3|
|4|" on"|(1, 4, d)|4|

Without cache: 12+22+32+42=3012+22+32+42=30 operations. With cache: 1+2+3+4=101+2+3+4=10.



## Concept

During autoregressive generation, each new token requires attending to all previous tokens. Without caching, this means recomputing K and V for the entire context at every step -- quadratic total work. KV-Cache stores the K and V tensors from previous steps and only computes new K, V for the current token, then concatenates.

The key insight: K and V for previous tokens do not change between generation steps. Only Q needs to be computed fresh for the new token. The cache turns an $O(N^2)$ total cost (across $N$ generation steps) into $O(N)$ per step.

## Common Pitfalls

### Concatenating Along the Wrong Dimension

K and V have shape `(batch, seq_len, model_dim)`. The cache grows along `dim=1` (sequence length). Concatenating along `dim=0` (batch) or `dim=2` (model_dim) would corrupt the tensors.
### Recomputing K, V for the Full Context

The whole point of the cache is to avoid recomputing K, V for previous tokens. If you project the full context through `k_proj` and `v_proj` at each step, you lose the speedup entirely.

## Key Takeaways

- KV-Cache stores K and V from previous tokens so they don't need to be recomputed. Only the new token's K and V are computed and appended.
- The cache grows by one entry per generation step via `torch.cat` along the sequence dimension.
- No causal mask is needed in the cached forward pass: the cache naturally contains only past tokens, enforcing causality by construction.
- The memory cost is linear in sequence length ($O(N⋅d)$ per layer), which is why long context windows are expensive. This motivates optimizations like Grouped Query Attention (fewer KV heads = smaller cache).




