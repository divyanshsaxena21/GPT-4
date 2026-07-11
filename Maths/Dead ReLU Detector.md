
![[Pasted image 20260711143345.png]]

![[Pasted image 20260711143552.png]]


## Concept

Dead ReLU neurons output zero for every sample in the batch. Because ReLU's gradient is zero for negative inputs, these neurons receive no gradient updates and are permanently stuck. The `detect_dead_neurons` method measures this per ReLU layer, and `suggest_fix` maps the severity pattern to the most appropriate intervention.

The fix priority reflects real debugging experience: severe death (> 50%) means the activation function itself is the problem (switch to LeakyReLU). Early-layer death (> 30% in layer 1) means initialization is bad (re-init). Depth-increasing death means the learning rate is too aggressive (reduce it).



## Key Takeaways

- A dead ReLU neuron outputs zero for every sample in the batch. It receives zero gradient and can never recover. This is a permanent failure mode.
- The severity pattern determines the fix: widespread death needs a new activation function, early-layer death needs re-initialization, and depth-correlated death needs a lower learning rate.
- Detection requires checking after the ReLU layer, not after the Linear layer. The Linear output being negative is expected; it's the ReLU output being zero for all samples that indicates death.
- LeakyReLU, PReLU, ELU, and GELU all avoid this problem by having non-zero gradients for negative inputs.

