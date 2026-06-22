- Gradient descent is an optimization algorithm which is commonly-used to train machine learning models and neural networks. It trains machine learning models by minimizing errors between predicted and actual results.

The goal of gradient descent is to minimize the cost function, or the error between predicted and actual y. In order to do this, it requires two data points—a direction and a learning rate. These factors determine the partial derivative calculations of future iterations, allowing it to gradually arrive at the local or global minimum (i.e. point of convergence).

- **[Learning rate](https://www.ibm.com/think/topics/learning-rate)** (also referred to as step size or the alpha) is the size of the steps that are taken to reach the minimum. This is typically a small value, and it is evaluated and updated based on the behavior of the cost function. High learning rates result in larger steps but risks overshooting the minimum. Conversely, a low learning rate has small step sizes. While it has the advantage of more precision, the number of iterations compromises overall efficiency as this takes more time and computations to reach the minimum.

- **The cost (or loss) function** measures the difference, or error, between actual y and predicted y at its current position. This improves the machine learning model’s efficacy by providing feedback to the model so that it can adjust the parameters to minimize the error and find the local or global minimum. ***It continuously iterates, moving along the direction of steepest descent (or the negative gradient) until the cost function is close to or at zero.*** At this point, the model will stop learning. Additionally, while the terms, cost function and [loss function](https://www.ibm.com/think/topics/loss-function), are considered synonymous, there is a slight difference between them. It’s worth noting that a loss function refers to the error of one training example, while a cost function calculates the average error across an entire training set.


## Types of Gradient Descent

### Batch Gradient Descent

- Batch gradient descent sums the error for each point in a training set, updating the model only after all training examples have been evaluated. This process referred to as a training epoch.
- While this batching provides computation efficiency, it can still have a long processing time for large training datasets as it still needs to store all of the data into memory. Batch gradient descent also usually produces a stable error gradient and convergence, but sometimes that convergence point isn’t the most ideal, finding the local minimum versus the global one.

### Stochastic Gradient Descent

- Stochastic gradient descent (SGD) runs a training epoch for each example within the dataset and it updates each training example's parameters one at a time. Since you only need to hold one training example, they are easier to store in memory. While these frequent updates can offer more detail and speed, it can result in losses in computational efficiency when compared to batch gradient descent. Its frequent updates can result in noisy gradients, but this can also be helpful in escaping the local minimum and finding the global one.

### Mini - Batch Gradient Descent

- Mini-batch gradient descent combines concepts from both batch gradient descent and stochastic gradient descent. It splits the training dataset into small batch sizes and performs updates on each of those batches. This approach strikes a balance between the computational efficiency of batch gradient descent and the speed of stochastic gradient descent.


## Challenges


### Local Minima and Saddle Points

- For convex problems, gradient descent can find the global minimum with ease, but as nonconvex problems emerge, gradient descent can struggle to find the global minimum, where the model achieves the best results.

- Recall that when the slope of the cost function is at or close to zero, the model stops learning. A few scenarios beyond the global minimum can also yield this slope, which are local minima and saddle points. Local minima mimic the shape of a global minimum, where the slope of the cost function increases on either side of the current point. However, with saddle points, the negative gradient only exists on one side of the point, reaching a local maximum on one side and a local minimum on the other. Its name inspired by that of a horse’s saddle.

- Noisy gradients can help the gradient escape local minimums and saddle points.


### Vanishing and Exploding Gradients

In deeper neural networks, particular [recurrent neural networks](https://www.ibm.com/think/topics/recurrent-neural-networks), we can also encounter two other problems when the model is trained with gradient descent and backpropagation.

- **Vanishing gradients:** This occurs when the gradient is too small. As we move backwards during backpropagation, the gradient continues to become smaller, causing the earlier layers in the network to learn more slowly than later layers. When this happens, the weight parameters update until they become insignificant—i.e. 0—resulting in an algorithm that is no longer learning.  


- **Exploding gradients:** This happens when the gradient is too large, creating an unstable model. In this case, the model weights will grow too large, and they will eventually be represented as NaN. One solution to this issue is to leverage a dimensionality reduction technique, which can help to minimize complexity within the model.

## For GPT-4

Every neural network, from a simple linear model to GPT-4, learns by gradient descent. Before we train any models, let's build the core optimization algorithm from scratch.

Minimize the function $f(x)=x^2$ using gradient descent. The minimum is obviously $x=0$ but you **must** arrive at it iteratively, the same way a neural network adjusts millions of parameters during training.

The derivative of $f(x)=x^2$ is $f^′(x)=2x$. At each step, update $x$ using the gradient descent rule:

$x_{new} = x_{old} − α ⋅ f^′(x_{old})$

where $α$ is the learning rate, a small constant that controls step size.

![[Pasted image 20260622205141.png|343]]
*Gradient descent following the steepest path downhill*




