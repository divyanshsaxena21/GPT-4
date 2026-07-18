

The Data Loader works with pre-tokenized integers. This problem goes one step earlier, starting from **raw text**. When people say “ChatGPT was trained on the internet,” what actually happens? A text corpus gets sliced into overlapping _(context, next_token)_ pairs using a sliding window.

## Input–Target Window Construction

For a document of length `L` and a context length `C`, any valid starting position

```text
i ∈ [0, L - C - 1]
```

produces the following input–target pair:

```text
Xᵢ = [tokenᵢ, ..., tokenᵢ₊₍C₋₁₎]
Yᵢ = [tokenᵢ₊₁, ..., tokenᵢ₊C]
```

Or, written more explicitly:

- **Input (`Xᵢ`)**: `token[i]` → `token[i + C - 1]`
- **Target (`Yᵢ`)**: `token[i + 1]` → `token[i + C]`

> [!note]
> The target sequence is simply the input sequence shifted **one token to the left**, enabling the model to learn **next-token prediction**.


![[Pasted image 20260718142830.png]]

![[Pasted image 20260718142900.png]]

![[Pasted image 20260718142919.png]]

![[Pasted image 20260718142959.png]]


# Dataset Loader (Raw Text Version)

## Concept

The GPT data loader typically works with **pre-encoded integer token sequences**. However, before text can be encoded into integers, it must first be **tokenized** and organized into training examples.

This dataset loader combines **tokenization** and **batch creation** into a single pipeline, operating directly on **raw text** instead of integer IDs.

---

## Pipeline

1. **Tokenize the raw text**
   - Split the text on whitespace.
   - This produces a list of **word tokens**.

2. **Sample random starting indices**
   - Use PyTorch's random number generator.
   - Sample only from **valid starting positions**.

3. **Create input–target pairs**
   - For each starting index `i`:
     - **Input:** Words from position `i` to `i + C - 1`
     - **Target:** Words from position `i + 1` to `i + C`

---

## Input–Target Construction

For each sampled index `i`:

```text
Input  = words[i : i + C]
Target = words[i + 1 : i + C + 1]
```

or equivalently,

```text
Xᵢ = [wordᵢ, ..., wordᵢ₊₍C₋₁₎]
Yᵢ = [wordᵢ₊₁, ..., wordᵢ₊C]
```

The target sequence is simply the input sequence shifted **one word forward**.

---

## Difference from the Integer Data Loader

| Integer Data Loader | Raw Text Data Loader |
|---------------------|----------------------|
| Operates on integer token IDs | Operates on string tokens (words) |
| Returns PyTorch tensors | Returns Python lists of words |
| Used during model training | Useful for debugging and understanding the data pipeline |

> [!note]
> Because this loader returns **actual words**, you can print batches and verify that the input–target pairs are constructed correctly before introducing tokenization and numerical encoding.

---

## Context Length (`C`)

The **context length (`C`)** determines how many previous words the model can use to predict the next word.

For a context length of `C`:

- **Input length:** `C` words
- **Target length:** `C` words
- **Target** is the **input shifted by one position**

### Example (`C = 4`)

```text
Input  : [The, cat, sat, on]
Target : [cat, sat, on, the]
```

Here, the model learns to predict:

- `cat` from `The`
- `sat` from `The cat`
- `on` from `The cat sat`
- `the` from `The cat sat on`

---

## Trade-off of Context Length

The choice of context length affects both model capability and computational cost.

- **Larger `C`**
  - More previous words are available as context.
  - Better long-range understanding.
  - Higher memory usage.
  - More computation.

- **Smaller `C`**
  - Faster training.
  - Lower memory requirements.
  - Less contextual information.

> [!important]
> Self-attention has a computational complexity of **O(C²)**. Doubling the context length roughly **quadruples** the attention computation, making long contexts significantly more expensive.


## Common Pitfalls

### Using Python's random Instead of torch.randint

The problem uses `torch.manual_seed(0)` for reproducibility. Using `random.randint` instead produces different indices and fails the test cases.



## Key Takeaways

- A dataset loader combines tokenization and batch creation, going from raw text to training pairs in one step.
- Word-level tokenization by splitting on whitespace is the simplest approach, though it cannot handle subword patterns or unseen words like BPE can.
- Using `torch.manual_seed` ensures reproducible batches. This is critical for debugging and testing, and it is the pattern used in the GPT training loop.



A single Wikipedia article can produce thousands of these training examples.
