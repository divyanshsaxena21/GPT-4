

![[Pasted image 20260707174821.png]]

- Features at different scales makes training unstable
- Mean $\approx$ 0, Variance $\approx$ 1  -> Training Stable



## Batch Normalization

The normalized output is computed as:

$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$

The final output after applying learnable scale and shift is:

$y_i = \gamma_i \hat{x}_i + \beta_i$

where

### Mean

$\mu = \frac{1}{n} \sum_{i=1}^{n} x_i$

Computes the mean of the features.

### Variance

$\sigma^2 = \frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)^2$

Computes the variance of the features.

### Learnable Parameters

- \($\gamma$): Learnable scaling parameter.
- ($\beta$): Learnable shifting parameter.

### Numerical Stability

$\epsilon = 10^{-5}$

A small constant added to the variance to prevent division by zero.



# Batch Vs Layer Normalization

![[Pasted image 20260707175752.png]]



![[Pasted image 20260707175900.png]]




![[Pasted image 20260707222616.png]]


![[Pasted image 20260707222633.png]]

![[Pasted image 20260707222705.png]]

