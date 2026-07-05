
Short for "backward propagation of error", backpropagation is an elegant method to calculate how changes to any of the weights or biases of a neural network will affect the accuracy of model predictions. It’s essential to the use of supervised learning, [semi-supervised learning](https://www.ibm.com/think/topics/semi-supervised-learning) or [self-supervised learning](https://www.ibm.com/think/topics/self-supervised-learning) to train neural networks.


These three interwoven processes—a loss function that tracks model error across different inputs, the backward propagation of that error to see how different parts of the network contribute to the error and the gradient descent algorithms that adjust model weights accordingly—are how deep learning models “learn.” As such, backpropagation is fundamental to training neural network models, from the most basic [multilayer perceptrons](https://www.ibm.com/docs/en/spss-statistics/saas?topic=networks-multilayer-perceptron) to the complex deep neural network architectures used for [generative AI](https://www.ibm.com/topics/generative-ai).


Training neural networks with backpropagation entails the following steps:

- **A _forward pass_**_,_ **making predictions on training data.**
- **A _loss function_ measures the error of the model’s predictions during that forward pass.**
- **_Backpropagation_ of error, or a _backward pass,_ to calculate the partial derivatives of the loss function.**
- **_Gradient descent,_ to update model weights.**


**"Loss function," "cost function" or "error function?"  

**It’s worth quickly noting that in some contexts, the terms _cost function_ or _error function_ are used in place of _loss function_, with “cost” or “error” replacing “loss.”

Though some machine learning literature assigns unique nuance to each term, they’re generally interchangeable.1 An _objective function_ is a broader term for any such evaluation function that we want to either minimize or maximize. _Loss function, cost function_ or _error_ _function_ refer specifically to terms we want to minimize.


# Multi-Layer Backpropagation


**Architecture:** 
$x→Linear(W1,b1)→ReLU→Linear(W2,b2)→$$\hat{y}$

**Loss:** $L=1/n∑i($$\hat{y_i}$$−y_i)^2 (MSE)$


