# Train/Dev/Test, Regularization, Initialization | Content

## Regularization Techniques

  - L1, L2: how to implement and why they reduce overfitting 
  - Dropout: how to implement and why they reduce overfitting
  - Data Augmentation
  - Early stopping

L2 and dropout basically sets values for W close to 0, reducing the number of features used by the model preveting more complicated models to arise


## Feature Normalization (std = 1, mean = 0)

How and why they help gradient descent: having every feature on the same scale allow the gradient descent to be faster since there's a limited space to move in

## Vanishing and Exploding gradients

**Problem**: In a network of n hidden layers, n derivatives will be multiplied together. If the derivatives are large then the gradient will increase exponentially as we propagate down the model until they eventually explode, and this is what we call the problem of exploding gradient. Alternatively, if the derivatives are small then the gradient will decrease exponentially as we propagate through the model until it eventually vanishes, and this is the vanishing gradient problem.

**Solutions**:

- Weights initialization: initializing weights to certain values helps avoiding vanishing gradients
- Reducing the amount of layers
- Gradient clipping
- Weight regularizaiton
