
In [machine learning (ML)](https://www.ibm.com/think/topics/machine-learning), a loss function is used to measure model performance by calculating the deviation of a model’s predictions from the correct, “ground truth” predictions. Optimizing a model entails adjusting model parameters to minimize the output of some loss function.


- A loss function is a type of _objective function_, which in the context of [data science](https://www.ibm.com/think/topics/data-science) refers to any function whose minimization or maximization represents the objective of model training. The term “loss function,” which is usually synonymous with _cost function_ or _error function_, refers specifically to situations where minimization is the training objective for a machine learning model.

- In simple terms, a loss function tracks the degree of error in an [artificial intelligence (AI) model's](https://www.ibm.com/think/topics/ai-model) outputs. It does so by quantifying the difference (“loss”) between a predicted value—that is, the model’s output—for a given input and the actual value or _ground truth_. If a model’s predictions are accurate, the loss is small. If its predictions are inaccurate, the loss is large.

- The fundamental goal of machine learning is to train models to output good predictions. Loss functions enable us to define and pursue that goal mathematically. During training, models “learn” to output better predictions by adjusting parameters in a way that reduces loss. A machine learning model has been sufficiently trained when loss has been minimized below some predetermined threshold.


## Regularization

Though models are trained and validated by making predictions on a training data set, performing well on training examples is not the ultimate objective. The true goal of machine learning is to train models that generalize well to new examples.

Relying solely on the minimization of a singular loss function is called “empirical risk minimization.” While it has an obvious, simple appeal, it runs the risk of a model [overfitting](https://www.ibm.com/think/topics/overfitting) the training data and thus generalizing poorly. To reduce this risk, among other purposes, many algorithms and architectures introduce _regularization terms_ that modify the primary loss function.

For example, mean absolute error (MAE)—which in this context is called L1 regularization—can be used to enforce sparsity by penalizing the number of activated neurons in a neural network or the magnitude of their activation.


## Types of loss functions

There exists a wide variety of different loss functions, each suited to different objectives, data types and priorities. At the highest level, the most commonly used loss functions are divided into regression loss functions and classification loss functions.  

- **Regression loss functions** measure errors in predictions involving _continuous_ values. Though they most intuitively apply to models that directly estimate quantifiable concepts such as price, age, size or time, regression loss has a wide range of applications. For example, a regression loss function can be used to optimize an image model whose task entails estimating the color value of individual pixels.  


- **Classification loss functions** measure errors in predictions involving _discrete_ values, such as the category a data point belongs to or if an email is spam or not. Types of classification loss can be further subdivided into those suitable for _binary classification_ and those suitable for _multi-class classification_.

### Cross-entropy loss functions

In most cases, classification loss is calculated in terms of entropy. Entropy, in plain language, is a measure of uncertainty within a system. For an intuitive example, compare flipping coins to rolling dice: the former has lower entropy, as there are fewer potential outcomes in a coin flip (2) than in a dice toss (6).

In supervised learning, model predictions are compared to the ground truth classifications provided by data labels. Those ground truth labels are certain and thus have low or no entropy. As such, we can measure loss in terms of the difference in certainty we’d have using the ground truth labels to the certainty of the labels predicted by the model.

The formula for cross-entropy loss (CEL) is derived from that of Kullback-Leibler divergence (KL divergence), which measures the difference between two probability distributions. Ultimately, minimizing loss entails minimizing the difference between the ground truth distribution of probabilities assigned to each potential label and the relative probabilities for each label predicted by the model.\



## Binary Cross-Entropy (for Two Classes)

$$
L = -\frac{1}{n} \sum_{i=1}^{n}
\left[
y_i \ln(p_i) + (1 - y_i)\ln(1 - p_i)
\right]
$$

Where:

- $y_i$ is the true label (0 or 1).
- $p_i$ is the predicted probability of the positive class.
- $n$ is the total number of samples.

---

## Categorical Cross-Entropy (for Multiple Classes)

$$
L = -\frac{1}{n}
\sum_{i=1}^{n}
\sum_{c=1}^{C}
y_{i,c}\ln(p_{i,c})
$$

Where:

- $y_{i,c}$ is 1 if sample $i$ belongs to class $c$ (one-hot encoded), otherwise 0.
- $p_{i,c}$ is the predicted probability that sample $i$ belongs to class $c$.
- $C$ is the total number of classes.
- $n$ is the total number of samples.