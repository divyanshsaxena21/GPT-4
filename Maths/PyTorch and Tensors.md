
A **tensor** is the fundamental data structure in PyTorch, essentially a multi-dimensional array (like NumPy's ndarray) that can live on a GPU for hardware-accelerated math. Every neural network weight, input, and gradient is a tensor.



### Tensors

Linguistically, “tensor” functions as a generic term inclusive of some more familiar mathematical entities:

- A _scalar_ is a zero-dimensional tensor, containing a single number.  
      
    
- A _vector_ is a one-dimensional tensor, containing multiple scalars of the same type. A _tuple_ is a one-dimensional tensor containing different data types.  
      
    
- A _matrix_ is a two-dimensional tensor, containing multiple vectors of the same type.  
      
    
- Tensors with three or more dimensions, like the [three-dimensional tensors used to represent RGB images](https://developer.ibm.com/articles/introduction-to-convolutional-neural-networks/) in computer vision algorithms, are collectively referred to as _N-dimensional tensors_.



### Modules

PyTorch uses modules as the building blocks of deep learning models, which allows for the quick and straightforward construction of neural networks without the tedious work of manually coding each algorithm.


 Broadly speaking, there are three primary classes of modules used to build and optimize deep learning models in PyTorch:

- **nn modules** are deployed as the layers of a neural network. The _torch.nn_ package contains a large library of modules that perform common operations like [convolutions](https://developer.ibm.com/articles/introduction-to-convolutional-neural-networks/), pooling and regression. For example, _torch.nn.Linear(n,m)_ calls a [linear regression](https://www.ibm.com/think/topics/linear-regression) algorithm with n inputs and m outputs (whose initial inputs and parameters are then established in subsequent lines of code).  
     
    
- The **autograd module** provides a simple way to automatically compute gradients, used to optimize model parameters via [gradient descent](https://www.ibm.com/think/topics/gradient-descent), for any function operated within a neural network. Appending any tensor with _requires_grad=True_ signals to autograd that every operation on that tensor should be tracked, which enables automatic differentiation.  
     
    
- **Optim modules** apply optimization algorithms to those gradients. _Torch.optim_ provides modules for various optimization methods, like stochastic gradient descent (SGD) or root mean square propagation (RMSprop), to suit specific optimization needs.



PyTorch is the most widely used deep learning framework. Its core data structure is the **tensor**, which works like a NumPy array but adds two superpowers: automatic differentiation (autograd) and GPU acceleration.

**Reshaping** changes a tensor's dimensions without touching its data. A (2,4) tensor has 8 elements, and you can reshape it to (4,2), (1,8) or (8,1). The total element count must stay the same. This is used constantly: for example, flattening a 2D image into a 1D vector for a linear layer.

**Averaging** along a dimension collapses that dimension. For a (3,2) tensor, averaging along `dim=0` (rows) produces a (2,) tensor with column means. Averaging along `dim=1` (columns) produces a (3,) tensor with row means. Think of `dim=` as "the dimension that disappears."

**Concatenation** joins tensors along a dimension. Concatenating two (2,3) tensors along `dim=1` produces a (2,6) tensor. The tensors must agree on all other dimensions.

**MSE Loss** is available as `torch.nn.functional.mse_loss`, computing $1/N∑(pred_i−target_i)^2$. Using built-in loss functions is preferred because they handle edge cases and are numerically stable.


