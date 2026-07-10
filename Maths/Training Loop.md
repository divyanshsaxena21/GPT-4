
![[Pasted image 20260709202815.png]]


# Linear Regression Model

## Model

The prediction equation is:

$$
\hat{y} = Xw + b
$$

where:

- $\hat{y}$ = predicted values
- $X$ = input feature matrix
- $w$ = weight vector
- $b$ = bias term

---

# Mean Squared Error (MSE) Loss

The loss function is:

$$
L = \frac{1}{n}\sum_{i=1}^{n}\left(\hat{y}_i - y_i\right)^2
$$

where:

- $n$ = number of training samples
- $\hat{y}_i$ = predicted value for the $i^{th}$ sample
- $y_i$ = actual value for the $i^{th}$ sample

---

# Gradients

## Gradient with respect to the weights

$$
\frac{\partial L}{\partial w}
=
\frac{2}{n}X^T(\hat{y}-y)
$$

## Gradient with respect to the bias

$$
\frac{\partial L}{\partial b}
=
\frac{2}{n}\sum_{i=1}^{n}\left(\hat{y}_i-y_i\right)
$$

---

# Gradient Descent Update Rule

Update the parameters using the learning rate $\alpha$:

## Weight update

$$
w = w - \alpha \frac{\partial L}{\partial w}
$$

## Bias update

$$
b = b - \alpha \frac{\partial L}{\partial b}
$$

# Concept


# Training Loop

A **training loop** is the engine that drives learning in machine learning. It repeatedly performs four fundamental steps:

1. **Forward pass**
2. **Loss computation**
3. **Gradient calculation (Backpropagation)**
4. **Weight update**

Every neural network follows this pattern, from **linear regression** to large language models like **GPT**.

---

# Training Loop for Linear Regression

## 1. Forward Pass

Compute the model's predictions:

$$
\hat{y} = Xw + b
$$

where:

- $X$ = input feature matrix
- $w$ = weight vector
- $b$ = bias term
- $\hat{y}$ = predicted values

---

## 2. Loss Computation

Use the **Mean Squared Error (MSE)** loss:

$$
L = \frac{1}{N}\sum_{i=1}^{N}\left(\hat{y}_i - y_i\right)^2
$$

where:

- $N$ = number of training samples
- $\hat{y}_i$ = predicted value
- $y_i$ = actual value

---

## 3. Gradient Calculation

### Gradient with respect to the weights

$$
\frac{\partial L}{\partial w}
=
\frac{2}{N}X^T(\hat{y}-y)
$$

### Gradient with respect to the bias

$$
\frac{\partial L}{\partial b}
=
\frac{2}{N}\sum_{i=1}^{N}\left(\hat{y}_i-y_i\right)
$$

---

## 4. Parameter Update

Update the parameters using the learning rate $\alpha$:

### Weight Update

$$
w \leftarrow w - \alpha \frac{\partial L}{\partial w}
$$

### Bias Update

$$
b \leftarrow b - \alpha \frac{\partial L}{\partial b}
$$

---

# Understanding the Vectorized Gradient

The gradient

$$
\frac{2}{N}X^T(\hat{y}-y)
$$

is the **vectorized form** of the weight gradient.

Instead of computing each weight's gradient separately using individual dot products, the matrix multiplication

$$
X^T(\hat{y}-y)
$$

computes **all weight gradients simultaneously**.

This is one of the main advantages of representing machine learning computations with matrices—it is both cleaner and significantly faster.

---

# Epochs

The training loop repeats for a fixed number of **epochs**.

An **epoch** is one complete pass through the entire training dataset.

With each epoch:

- Predictions are computed.
- Loss is calculated.
- Gradients are computed.
- Parameters are updated.

As training progresses, the weights and bias gradually converge toward values that minimize the training loss.

---

# Connection to PyTorch

PyTorch follows the same four-step training process, but it automates the gradient computation and parameter updates.

A typical PyTorch training loop looks like:

```python
# Forward pass
predictions = model(X)

# Compute loss
loss = loss_fn(predictions, y)

# Compute gradients
loss.backward()

# Update parameters
optimizer.step()

# Clear gradients
optimizer.zero_grad()
```

Here:

- `loss.backward()` computes all gradients automatically using **automatic differentiation (Autograd)**.
- `optimizer.step()` updates the model parameters using the chosen optimization algorithm (such as Gradient Descent or Adam).
- `optimizer.zero_grad()` clears previously accumulated gradients before the next iteration.

