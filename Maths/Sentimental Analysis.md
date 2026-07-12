
The embedding layer maps each token ID to a learned 16-dimensional vector. Averaging across the sequence collapses variable-length input into a fixed-size representation. The linear layer + sigmoid outputs a probability between 0 (negative) and 1 (positive).



![[Pasted image 20260712211813.png]]


![[Pasted image 20260712211836.png]]

![[Pasted image 20260712211923.png]]



![[Pasted image 20260712211939.png]]



## Concept

Sentiment analysis classifies text as positive or negative. This problem uses a simple but effective **embedding bag** architecture:

1. **Embed** each word using `nn.Embedding`, producing a matrix of shape $(B,T,d)$ where $B$ is batch size, $T$ is sequence length, and dd is embedding dimension.
2. **Average** all word embeddings in each sentence across the sequence dimension (`dim=1`), producing a single vector of shape $(B,d)$ per sentence.
3. **Project** this averaged vector to a single score using `nn.Linear`.
4. **Sigmoid** converts the score to a probability.

The averaging step is the key design choice. It creates a fixed-size representation from variable-length input without any recurrence or attention. This makes it a **bag-of-words** model: word order is thrown away. "I love not this" and "I not love this" produce the same representation.

Despite ignoring word order, bag-of-words models work surprisingly well for sentiment because individual words carry strong signals. "Amazing," "terrible," "boring," and "fantastic" are strong sentiment indicators regardless of position. The embedding layer learns these associations: positive words develop vectors that point in one direction, negative words point in another, and the linear layer learns to separate them.


## Key Takeaways

- Averaging word embeddings is the simplest way to create fixed-size sentence representations from variable-length inputs.
- Bag-of-words models ignore word order but work well when individual words carry strong signals, as in sentiment analysis.
- The embedding layer learns task-specific representations during training, so "great" and "excellent" develop similar vectors without any manual feature engineering.

