
![[Pasted image 20260712194956.png]]

![[Pasted image 20260712195031.png]]

***PARALLEL Arrows = Same Relation***



![[Pasted image 20260712195318.png]]


An **embedding table** is simply a matrix of shape `(vocab_size, embed_dim)`. To look up the embedding for a token ID, you index into the matrix: `embedding[token_id]`.



## Concept

Word embeddings map discrete tokens (words, characters, or subwords) to dense vectors of real numbers. This solves a fundamental problem: neural networks need numerical inputs, but text is categorical.

The naive approach is **one-hot encoding**: a vocabulary of $V$ words becomes $V$-dimensional vectors with a single 1 and the rest 0s. But this is wasteful (GPT-2's vocabulary has 50,257 tokens) and treats all words as equally different. "King" and "queen" are just as far apart as "king" and "banana."

An embedding matrix $E$ has shape $(V,d)$, where each row is a $d$-dimensional vector for one token. Looking up token $i$ is just selecting row $i$ from E:

embed(token_ids)=E[token_ids]

This is equivalent to multiplying a one-hot vector by $E$, but indexing is $O(1)$ per token versus $O(V)$ for the matrix multiply. The embedding vectors are learned parameters: during training, backpropagation updates only the rows that were looked up in each batch.

The magic happens during training. Words used in similar contexts develop similar embedding vectors. "King" and "queen" end up close together. "Paris" and "France" develop a similar offset as "Tokyo" and "Japan." These relationships emerge automatically from the training data.


## Key Takeaways

- Embedding lookup is just array indexing, making it $O(1)$ per token. This is far more efficient than the mathematically equivalent one-hot matrix multiply.
- Embedding vectors are learned parameters that capture semantic relationships. Similar words develop similar vectors during training.
- The embedding matrix is one of the largest components in language models: GPT-2's is $50,257×768=38$ million parameters.

