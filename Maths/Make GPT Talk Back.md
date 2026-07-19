

Your GPT can train. Now make it speak. You started with gradient descent and a quadratic function. You built neurons, taught them to learn with backprop, stacked them into networks, learned PyTorch, processed language, implemented attention, assembled a transformer. And now your GPT is about to generate text. The same autoregressive loop you'll implement here is what runs every time someone talks to ChatGPT: predict the next token, append it, repeat.

The generation loop is autoregressive:

# GPT Text Generation Loop

## Concept

GPT generates text **one token at a time** using an autoregressive process.

At each iteration, the model predicts the probability distribution for the **next token**, samples one token, appends it to the current context, and repeats the process.

---

## Generation Steps

### 1. Run the Forward Pass

Feed the current context into the model:

```python
logits = model(context)
```

The output has shape

$$
(1,\;\text{seq\_len},\;\text{vocab\_size})
$$

where

- $1$ = batch size
- $\text{seq\_len}$ = current sequence length
- $\text{vocab\_size}$ = number of possible output tokens

Each position contains logits for predicting the **next token**.

---

### 2. Select the Last Position

Only the logits from the **last token** are used for generation:

```python
logits = logits[:, -1, :]
```

This produces a tensor of shape

$$
(1,\;\text{vocab\_size})
$$

These logits represent the model's prediction for the next token after the current sequence.

---

### 3. Convert Logits to Probabilities

Apply the softmax function:

```python
probs = torch.softmax(logits, dim=-1)
```

Mathematically,

$$
P(x_i)
=
\frac{e^{z_i}}
{\sum_j e^{z_j}}
$$

where

- $z_i$ is the logit for token $i$
- $P(x_i)$ is the probability of selecting token $i$

The resulting probability distribution sums to $1$.

---

### 4. Sample the Next Token

Draw a token according to the probability distribution:

```python
next_token = torch.multinomial(probs, num_samples=1)
```

Unlike greedy decoding (which always chooses the most probable token), sampling allows the model to generate more varied and creative text.

---

### 5. Append the Token

Append the sampled token to the existing context:

```python
context = torch.cat([context, next_token], dim=1)
```

The generated sequence grows by one token.

---

### 6. Crop the Context

GPT can only attend to the previous `context_length` tokens.

If the sequence becomes longer than this limit, keep only the most recent tokens:

```python
context = context[:, -context_length:]
```

This creates a sliding context window while continuing generation.

---

### 7. Repeat

Repeat the entire process until the desired number of tokens has been generated.

The generation loop is therefore

$$
\text{Forward Pass}
\rightarrow
\text{Last Logits}
\rightarrow
\text{Softmax}
\rightarrow
\text{Sample}
\rightarrow
\text{Append}
\rightarrow
\text{Crop Context}
\rightarrow
\text{Repeat}
$$

---

# Context Window

The provided GPT model has been trained on **Drake lyrics**.

Because the model has a fixed context length, it cannot attend to tokens that are more than `context_length` positions in the past.

At every generation step, older tokens are discarded, and only the most recent `context_length` tokens are retained.

---

# Output

After generating the required number of tokens:

1. Convert the token IDs back into text.
2. Return the **entire generated sequence** as a string.




![[Pasted image 20260719203636.png]]

![[Pasted image 20260719203702.png]]

![[Pasted image 20260719203718.png]]

![[Pasted image 20260719203736.png]]



## Concept

Text generation with GPT is an autoregressive loop: the model generates one token at a time, appends it to the context, and repeats. This is the inference-time procedure that turns a trained language model into a text generator.

The generation loop:

1. **Crop** the context to the model's maximum context length (if it has grown too long).
2. **Forward pass**: Feed the context through the model to get probabilities at every position.
3. **Extract last position**: Only the final position's distribution matters, since it predicts the next token.
4. **Sample**: Draw a token from the distribution using `torch.multinomial`.
5. **Append**: Add the sampled token to the context.
6. **Decode**: Convert the token ID to a character.
7. Repeat for the desired number of new characters.

**Sampling vs. greedy decoding**: Taking the argmax (most probable token) every time produces deterministic but repetitive text. Sampling from the full distribution introduces variety. In production, temperature scaling $(probs=softmax(logits/T))$ and top-k/top-p filtering control the creativity-coherence tradeoff.

The context window is finite. Once the context exceeds the maximum length $C$, earlier tokens are cropped off. The model's "memory" is limited to the most recent $C$ tokens. This is why GPT models have a context length limit (1024 for GPT-2, 8192+ for modern models).


## Common Pitfalls

### Forgetting to Crop the Context

Without cropping, the context grows past the model's maximum length, causing position embedding index errors.

### Using Argmax Instead of Sampling

Greedy decoding (argmax) produces deterministic but often repetitive text. The problem expects sampling with `torch.multinomial`.

### Taking Logits from All Positions Instead of the Last

The model outputs logits at every position, but only the last position predicts the next token. Using logits from other positions gives you predictions for tokens that already exist.


## Key Takeaways

- Autoregressive generation produces text one token at a time, feeding each prediction back as input to produce the next one.
- The context window limits the model's "memory" to the most recent $C$ tokens. Older tokens are cropped and forgotten.
- Sampling from the distribution (rather than taking argmax) produces more diverse and natural text, and forms the basis for more advanced decoding strategies like temperature scaling and top-p sampling.

