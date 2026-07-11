


![[Pasted image 20260711144440.png]]

![[Pasted image 20260711144456.png]]


![[Pasted image 20260711144901.png]]


**Greedy Tokenization Algorithm:**  
Starting from the left of the text, find the longest substring that exists in the vocabulary. Consume that token, then repeat from where you left off. If no match is found, consume a single character.

1. **tokenize_numbers(numbers, vocab)** - Tokenize each number (as a string) using the vocabulary. Return a list of token lists showing how each number gets split. This reveals the inconsistency: consecutive numbers like 2249, 2250, 2251 can be tokenized as `["22","49"]`, `["225","0"]`, `["225","1"]` - completely different structures for numbers just 1 apart.
    
2. **count_tokens(text, vocab)** - Count the total number of tokens produced by greedy tokenization. Text that matches multi-character vocabulary entries uses fewer tokens.
    
3. **fertility_score(text, vocab)** - Compute the tokens-per-word ratio (called "fertility"). Split the text by spaces to get words. Fertility = token_count / word_count. A score of 1.0 means one token per word (ideal). English in GPT-4 averages ~1.3. Some languages score 3-6x higher, meaning they cost more per word and get less context. Round to 4 decimal places.

## Concept

Greedy tokenization starts at position 0, finds the longest substring in the vocabulary, consumes it as a token, then repeats from the next position. This deterministic process creates surprising behaviors: consecutive numbers get completely different token structures, and languages with fewer vocabulary entries need more tokens per word.

The three functions expose these behaviors: `tokenize_numbers` shows the inconsistency, `count_tokens` measures total token usage, and `fertility_score` computes the tokens-per-word ratio that determines how efficient a language is for the model.



## Key Takeaways

- Greedy left-to-right longest match is simple but creates non-obvious token boundaries. Consecutive numbers can have completely different token structures.
- Fertility (tokens per word) measures how efficient a language is for a given tokenizer. English averages around 1.3 in GPT-4; some languages are 3-6x higher.
- The shared `_greedy_tokenize` helper avoids code duplication: all three methods use the same tokenization logic and just differ in what they compute from the result.
- Real BPE tokenizers use merge-order priority rather than simple longest match, but the edge cases are similar.
