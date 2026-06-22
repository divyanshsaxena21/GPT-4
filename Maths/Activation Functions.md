

## Sigmoid Activation Function — (Logistic function)


- A Sigmoid function is a mathematical function having a characteristic “S”-shaped curve or sigmoid curve which ranges between 0 and 1, therefore it is used for models where we need to predict the probability as an output.

$$
f(x) = \frac{1}{1 + e^{-x}}
$$

![[Pasted image 20260622231643.png]]


## ReLU - Rectified Linear Unit

- ReLu is the most used activation function in CNN and ANN which ranges from zero to infinity.[0,∞)

$$
\text{ReLU}(x)=
\begin{cases}
x & \text{if } x > 0 \\
0 & \text{if } x \le 0
\end{cases}
$$

![[Pasted image 20260622231845.png]]


## Leaky ReLU

- **Leaky ReLU** is an activation function used in neural networks designed to solve the "dying ReLU" problem.

$$
\text{Leaky ReLU}(x)=
\begin{cases}
x & \text{if } x > 0 \\
ax & \text{if } x \le 0
\end{cases}
$$

![[Pasted image 20260622232126.png]]

