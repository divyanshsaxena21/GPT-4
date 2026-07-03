
A neuron takes a vector of inputs $x$, multiplies each by a corresponding weight $w$, adds a bias $b$, and passes the result through an activation function:

$\text{output} = \text{activation}\!\left(\sum_{i} w_i \cdot x_i + b\right)$


![[Pasted image 20260625190923-removebg-preview 1.jpg]]


A **neuron** (also called a perceptron) is the simplest unit in a neural network. It mimics a biological neuron: it receives signals (inputs), processes them (weighted sum + bias), and fires (activation function).


The **sigmoid** activation squishes any real number into the range (0,1): 
$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$


- The **ReLU** (Rectified Linear Unit) simply clips negative values to zero: $$ReLU(z)=max⁡(0,z)$$
