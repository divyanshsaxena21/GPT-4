
![[Pasted image 20260711192815.png]]

![[Pasted image 20260711192834.png]]



## Concept

The handwritten digit classifier is the "hello world" of deep learning. It takes a flattened 28×28=78428×28=784-dimensional MNIST image and predicts which digit (0-9) it represents.

The architecture is a two-layer MLP:

1. **Linear** (784→512): projects the high-dimensional pixel space into a learned 512-dimensional representation
2. **ReLU**: introduces non-linearity so the network can learn curved decision boundaries between digits
3. **Dropout** (p=0.2): randomly zeros 20% of activations during training. This forces the network to spread information across neurons rather than relying on a few. During evaluation, dropout is turned off.
4. **Linear** (512→10): projects to 10 class scores, one per digit
5. **Sigmoid**: squashes each score to (0,1)

Dropout is a form of regularization. Without it, the 400,000+ parameters (784×512+512×10) can easily memorize the training data. With dropout, the network must learn robust features that survive random neuron removal.

Despite its simplicity, this architecture achieves over 97% accuracy on MNIST. More complex architectures (CNNs) push this above 99%, but the MLP approach teaches the core pattern.


