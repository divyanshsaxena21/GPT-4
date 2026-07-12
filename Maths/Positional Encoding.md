

Transformers process all tokens in parallel. Unlike RNNs, they don't read left-to-right. Without positional information, the sentence "dog bites man" would be identical to "man bites dog." **Positional encoding** solves this by injecting position information into the embeddings using sine and cosine waves at different frequencies.


![[Pasted image 20260712202052.png]]



![[Pasted image 20260712202123.png]]

***Low dims -> fast oscillations
High dims -> slow oscillations***


![[Pasted image 20260712202226.png]]


![[Pasted image 20260712202253.png]]


![[Pasted image 20260712202318.png]]



# Concept: Sinusoidal Positional Encoding in Transformers

## Why Positional Encoding?

Transformers process **all tokens in parallel**, unlike RNNs, which process tokens **sequentially**. This parallelism makes transformers significantly faster, but it also means the model has **no inherent understanding of token order**.

Without positional information, the following sentences would produce identical representations:

- "the cat sat on the mat"
- "mat the on sat cat the"

To solve this, transformers add **positional encodings** to the input embeddings, allowing the model to distinguish between different token positions.

---

## Sinusoidal Positional Encoding

For a token at position `pos` and embedding dimension `i`, the positional encoding is defined as:

$PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d}}\right)$


$PE(pos, 2i + 1) = \cos\left(\frac{pos}{10000^{2i/d}}\right)$


where:

- `pos` = token position in the sequence
- `i` = embedding dimension index
- `d` = embedding size (model dimension)

---

## How It Works

- **Even-indexed dimensions (`2i`)** use the **sine** function.
- **Odd-indexed dimensions (`2i + 1`)** use the **cosine** function.

The denominator:

\[
10000^{2i/d}
\]

creates a **range of frequencies** across embedding dimensions:

- **Dimension 0** oscillates very rapidly (wavelength ≈ \(2\pi\)).
- Higher dimensions oscillate progressively more slowly.
- The final dimensions have extremely long wavelengths (≈ \(10000 \times 2\pi\)).

This produces a unique positional signature for every token.

---

## Intuition

Think of positional encoding like a **binary clock**:

- The **seconds hand** spins very quickly.
- The **minutes hand** spins more slowly.
- The **hours hand** moves even more slowly.

Similarly, each embedding dimension acts like a clock with a different frequency. Every token position receives a unique combination of values across these frequencies, creating a distinctive "timestamp."

---

## Why This Is Useful

One of the key properties of sinusoidal encodings is that the encoding for position `pos + k` can be expressed as a **linear function** of the encoding at position `pos`.

This makes it easy for the attention mechanism to learn **relative positional relationships**, such as:

- the next token,
- tokens five positions away,
- or longer-range dependencies,

without requiring the model to explicitly memorize every possible position.

---

## Key Takeaways

- Transformers process tokens in parallel, so they need explicit positional information.
- Sinusoidal positional encoding adds a fixed, unique vector to each token position.
- Sine is used for even dimensions, cosine for odd dimensions.
- Different dimensions encode different frequencies, ranging from fast to slow oscillations.
- The mathematical structure allows the model to easily infer relative positions, improving attention over sequences.


