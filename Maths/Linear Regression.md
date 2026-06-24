
# ==Forward== 

Linear Regression assumes a **linear** relationship in the data: no exponents, no logarithms, just weighted sums.

For example, imagine predicting Uber prices from three features: time of day, distance in miles, and estimated duration. The model assumes:

$price(x,y,z)=w1⋅x+w2⋅y+w3⋅z$

Training means finding the values of w1,w2,w3w1​,w2​,w3​ that make predictions closest to the real prices in the dataset. This is exactly what gradient descent does, and it's the same process used to train GPT, just with millions more weights.


# ==Training==

The training loop ties together everything from this series: forward pass, loss computation, and gradient descent.

At each iteration, weights are updated by subtracting the gradient scaled by the learning rate:

$W←W−α⋅∇_WL$

where $∇_WL$ is the gradient of the loss with respect to each weight (provided via `get_derivative()`), and $α$ = 0.01 is already set as `self.learning_rate`


Training a machine learning model means finding the weights that minimize the error between the model's predictions and the actual answers. This is done iteratively through gradient descent.

At each iteration, the training loop performs three steps:

1. **Forward pass**: Compute predictions using the current weights: $\hat{Y}= X⋅W$
2. **Compute gradients**: Calculate how much each weight contributes to the error (the derivative of the loss with respect to each weight)
3. **Update weights**: Nudge each weight in the direction that reduces the error: $W←W−α⋅∇L$ where $α$ is the learning rate

The learning rate αα controls how big each step is. Too large and you overshoot the minimum; too small and training takes forever. This loop is the foundation of how _every_
neural network is trained, from linear regression to GPT.



## Concept

Training linear regression means finding the weight vector ww that minimizes the MSE loss. The forward pass gives us predictions, the loss tells us how bad they are, and the gradient tells us how to adjust each weight.

The MSE loss is:

$L = \frac{1}{N} \sum_{i=1}^{N} \left(\hat{y}_i - y_i\right)^2$


The partial derivative with respect to weight $w_j$ is:

$\frac{\partial L}{\partial w_j}= \frac{-2}{N} \sum_{i=1}^{N} \left( y_i - \hat{y}_i \right)\cdot x_{i,j}$



In plain terms: the gradient for weight $j$ is the average of (the error at each sample times that sample's $j$-th feature). This makes sense intuitively: if a feature is large and the error is large, that weight needs a big update.

This can be computed as a single dot product: $\frac{-2}{N} (y - \hat{y})^T$ $X_j$​ where $X_j$​ is the $j$-th column of $X$. Each weight is updated independently using the gradient descent rule. This is called **batch gradient descent** because we use all $N$ samples for each update.