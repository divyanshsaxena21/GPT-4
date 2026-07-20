
Your multi-head attention gives every head its own K and V projections. But during inference with KV-Cache, storing K and V for all heads is expensive. Grouped Query Attention (GQA) shares K and V across groups of heads, achieving the same quality with a fraction of the memory.


**The Spectrum of Attention**

Standard Multi-Head Attention (MHA) uses (h) independent KV heads -- one per query head. Multi-Query Attention (MQA) goes to the other extreme: a single shared KV head for all queries. GQA is the middle ground with (g) KV heads, where (1 < g < h):

|Variant|KV Heads|Cache Size|Quality|
|---|---|---|---|
|MHA|(h)|Baseline|Best|
|GQA|(g)|(g/h) of MHA|Near-MHA|
|MQA|1|(1/h) of MHA|Slight drop|

When `num_kv_heads == num_heads`, GQA reduces to standard MHA. When `num_kv_heads == 1`, it becomes MQA.

**How It Works**

Each KV head serves a group of (h / g) query heads. You project Q with (h) heads but K and V with only (g) heads. Then you expand the (g) KV heads to match the (h) query heads by repeating each KV head within its group. After expansion, attention proceeds exactly as standard MHA: scaled dot-product with causal masking.

For example, with 8 query heads and 2 KV heads, each KV head is shared by 4 query heads. The expansion repeats each of the 2 KV head vectors 4 times to produce 8 matching vectors.

**Real Models Using GQA:**

- **Llama 2 70B**: 64 query heads, 8 KV heads (8x memory reduction)
- **Llama 3**: GQA across all model sizes
- **Mistral 7B**: 32 query heads, 8 KV heads
- **Gemma**: GQA as default attention

![[Pasted image 20260720195517.png]]

![[Pasted image 20260720195647.png]]

![[Pasted image 20260720195710.png]]

![[Pasted image 20260720195743.png]]


## Concept

Standard Multi-Head Attention gives every query head its own K and V projections. During inference with KV-Cache, all those K and V tensors must be stored, and the memory cost scales linearly with the number of heads. Grouped Query Attention reduces this by sharing K, V across groups of query heads.

With hh query heads and gg KV heads, each KV head serves h/gh/g query heads. The key operation is `repeat_interleave`: it expands the $g$ KV heads to $h$ by repeating each one $h/g$ times, making the shapes match for standard attention math. When $g=h$, GQA is identical to MHA. When $g=1$, it becomes Multi-Query Attention.


## Common Pitfalls

### Using `repeat` Instead of `repeat_interleave`

`repeat` tiles the entire tensor, while `repeat_interleave` repeats each element individually. With 2 KV heads and 4 query heads, you want `[KV0, KV0, KV1, KV1]`, not `[KV0, KV1, KV0, KV1]`.

### Forgetting the Causal Mask

GQA is used in decoder-only models (GPT, Llama) where tokens must not attend to future positions. Without the lower-triangular mask, the model breaks autoregressive generation.

### Wrong Reshape Order

The view/transpose sequence matters. Q has `num_heads` heads, but K and V have `num_kv_heads` heads. Using `num_heads` for the K reshape would produce incorrect head dimensions.


## Key Takeaways

- GQA shares K and V across groups of query heads, reducing KV-Cache memory by a factor of `num_heads / num_kv_heads` with minimal quality loss.
- `repeat_interleave` is the key operation: it expands each KV head to serve its assigned group of query heads, making the shapes compatible for standard attention.
- GQA generalizes both MHA ($g=h$) and MQA ($g=1$). Most production models use an intermediate value (e.g., Llama 2 70B uses 64 query heads with 8 KV heads).
- The attention computation after expansion is identical to standard MHA. The savings come entirely from the smaller projection layers and smaller cache.




