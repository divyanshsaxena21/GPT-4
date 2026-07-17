

Your vocabulary encoder turns text into a long sequence of integers. Now you need to slice that sequence into training examples. The key insight: a language model's training data is just the same sequence shifted by one position. If the input is `[t1, t2, t3]`, the target is `[t2, t3, t4]`. The model learns to predict each next token given the tokens before it. This loader takes **pre-tokenized integer sequences** and produces batches of these (input, target) pairs, ready for training.


