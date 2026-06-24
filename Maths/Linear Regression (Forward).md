
Linear Regression assumes a **linear** relationship in the data: no exponents, no logarithms, just weighted sums.

For example, imagine predicting Uber prices from three features: time of day, distance in miles, and estimated duration. The model assumes:

$price(x,y,z)=w1⋅x+w2⋅y+w3⋅z$

Training means finding the values of w1,w2,w3w1​,w2​,w3​ that make predictions closest to the real prices in the dataset. This is exactly what gradient descent does, and it's the same process used to train GPT, just with millions more weights.

