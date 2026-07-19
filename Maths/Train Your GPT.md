
The driver provides a `MiniGPT` model (returns raw logits) and a 1D integer data tensor. Implement the training loop.

**Steps per epoch:**

1. Set `torch.manual_seed(epoch)` for reproducible batch sampling
2. Sample random start indices with `torch.randint`
3. Build `X` (input) and `Y` (target) where `Y = X` shifted right by 1
4. Forward pass: `logits = model(X)` with shape `(batch_size, context_length, vocab_size)`
5. Reshape for cross-entropy: `logits` to `(B*T, C)`, `Y` to `(B*T)`
6. `loss = F.cross_entropy(logits_flat, targets_flat)`
7. `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`

Use `torch.optim.AdamW` as the optimizer.


![[Pasted image 20260719202508.png]]

![[Pasted image 20260719202521.png]]

![[Pasted image 20260719202535.png]]

![[Pasted image 20260719202555.png]]

![[Pasted image 20260719202606.png]]


# GPT Training Loop

## Concept

Training a GPT model follows the standard deep learning training loop, with details specific to **autoregressive language modeling**.

Each training epoch consists of the following steps.

## 1. Sample a Batch

Randomly select starting positions in the training corpus and construct:

- **Input sequence:** the current tokens
- **Target sequence:** the input shifted one position to the left

For example,

```text
Input :  [The, cat, sat]
Target:  [cat, sat, on]
```

The model learns to predict the next token at every position.

---

## 2. Forward Pass

Feed the input tokens through the model.

The output is a tensor of shape

$$
(B,\;T,\;V)
$$

where

- $B$ = batch size
- $T$ = context length (sequence length)
- $V$ = vocabulary size

Each position contains logits (unnormalized scores) over the entire vocabulary.

---

## 3. Compute Loss

Before applying cross-entropy, reshape the tensors:

- Logits:

$$
(B,\;T,\;V)
\;\rightarrow\;
(B \cdot T,\;V)
$$

- Targets:

$$
(B,\;T)
\;\rightarrow\;
(B \cdot T)
$$

Then compute

$$
\text{loss}
=
\text{CrossEntropy}(\text{logits},\;\text{targets})
$$

This treats every token position as an independent classification problem.

### Why Flatten?

PyTorch's `cross_entropy()` expects:

- Logits of shape

$$
(\text{samples},\;\text{classes})
$$

- Targets of shape

$$
(\text{samples})
$$

A batch containing $B$ sequences, each of length $T$, contains

$$
B \times T
$$

prediction positions.

Flattening converts the tensors into exactly the required format:

- Logits:

$$
(B \times T,\;V)
$$

- Targets:

$$
(B \times T)
$$

---

## 4. Backward Pass

Compute gradients for every trainable parameter:

```python
loss.backward()
```

Backpropagation automatically applies the chain rule through the entire computation graph.

---

## 5. Update Parameters

Update the model parameters using the optimizer:

```python
optimizer.step()
```

For GPT models, the optimizer is typically **AdamW**.

---

# AdamW Optimizer

AdamW is the standard optimizer for Transformer models.

It extends **Adam** by applying **weight decay** correctly.

Instead of incorporating L2 regularization into the gradient update (as Adam does), AdamW applies weight decay **directly to the model weights**, improving generalization.

The **"W"** stands for **Weight Decay**.

---

# Initial Loss

Suppose the vocabulary contains $V$ tokens.

An untrained model assigns nearly uniform probability to every token:

$$
P(\text{token})
=
\frac{1}{V}
$$

The expected cross-entropy loss is therefore approximately

$$
\ln(V)
$$

since

$$
-\ln\left(\frac{1}{V}\right)
=
\ln(V)
$$

As training progresses, the model assigns higher probability to the correct next token, causing the loss to decrease.

---

# Summary

Each training iteration follows the pipeline

$$
\text{Sample Batch}
\rightarrow
\text{Forward Pass}
\rightarrow
\text{Compute Loss}
\rightarrow
\text{Backward Pass}
\rightarrow
\text{Optimizer Step}
$$

The goal is to minimize the cross-entropy loss so that the model becomes increasingly accurate at predicting the next token in a sequence.

## Common Pitfalls

### Forgetting optimizer.zero_grad()

Without zeroing gradients, they accumulate across epochs. The updates become the sum of all previous gradients, causing erratic training.

### Wrong Reshape for Cross-Entropy

The logits must be $(B⋅T,V)$ and targets must be $(B⋅T)$. Getting the view dimensions wrong produces incorrect loss values or runtime errors.


## Key Takeaways

- Training GPT uses the universal loop: sample batch, forward, loss, backward, step. The only language-model-specific part is the reshape for cross-entropy.
- Cross-entropy loss treats each position independently: a batch of $B$ sequences of length $T$ yields $B×T$ classification examples.
- Setting `torch.manual_seed(epoch)` gives each epoch a different but reproducible batch, balancing randomness with reproducibility.



