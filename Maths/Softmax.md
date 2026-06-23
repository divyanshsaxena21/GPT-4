
Softmax Activation Function transforms a vector of numbers into a probability distribution, where each value represents the likelihood of a particular class. It is especially important for multi-class classification problems.

- Each output value lies between 0 and 1.
- The sum of all output values equals 1.

This property makes Softmax ideal for scenarios where each output neuron represents the probability of a distinct class.

## Softmax Function

For a given vector $z$ = $[ z_1, z_2, ......, z_n]$ the Softmax function is defined as:

$\sigma(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}$

where:

- $e^{z_i}$: Exponentiation of the input value.
- $\sum_{j=1}^{n} e^{z_j}$: Sum of all exponentiated values, used to normalize the outputs.

 Each output $\sigma(z_i)$: Probability assigned to class \(i\).

### Key Properties

- ***Normalization:*** Converts logits into a probability distribution where the sum equals 1.
- ***Exponentiation:*** Amplifies larger values making the model’s confidence more pronounced.
- ***Differentiable:*** Enables gradient-based optimization during backpropagation.
- ***Probabilistic Interpretation:*** Makes output easier to interpret as class likelihoods.


## How Softmax Activation Function Works

Softmax converts a vector of raw scores into a probability distribution.

- ***Input Scores:*** Take the raw output vector from the model. These values can be any real numbers.
- ***Exponentiate:*** Apply $e^x$ to make every value positive and amplify differences.
- ***Sum of exponentials:** Compute the normalising constant $Z=∑e^x'$
- ***Normalize:*** Divide each exponent by Z to get probabilities $$
p_i = \frac{e^{x_i'}}{Z}
$$
- ***Output (Probabilities):*** Final probability vector can be used with argmax to pick the predicted class.


