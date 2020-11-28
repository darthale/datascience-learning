# Optmiziation Algorithms | Content


The following techniques describe different optimization algorithms. The aim for this algorithms is to to speed up gradient descent process, so that we can train our NN faster even on big datasets.


## Mini-batch gradient descent

  - Vectorization already allows you to efficiently compute gradient descent on m examples
  - If m is very large (i.e 5000000), you need to process the full sample before taking a step into gradient descent, and to take another step you need to re-process the full sample again and so on
  - We can split m into smaller batches (i.e 1000 samples) instead (and the same for our labels Y). So now will run grandient descent on the mini-batches
  - With mini-batch gradient descent, a single pass through the training set, that is 1 epoch, allows you to take 5,000 gradient descent steps. You still want to take multiple passes through the training set until the gradient will converge of course. With normal batch gradient descent, a single pass through the training set allows you to take only one gradient descent step
  - mini-batch size can also be seen as an additional hyperparameter


## Exponentialy weighted averages: gradient descent with momentum

  - Gradient descent with momentum is much faster than gradient descent (gd). To implement it we leverage the exponentially weighted average
  - With normal gd you'll have oscillations before converging. To prevent this oscillations from diverging, you need to use a small learning rate that generates many steps of gd
  - gd with momentum has smaller vertical oscillations, make it overall faster since it's taking a more straightforward path
  - here another hyperparameter comes up, which is the Beta parameter that control the exponential weighted average
  - gd with momentum also helps avoiding our process to get stuck into local minimums


## RMSprop

  - RMSprop also aims to speed up normal gradient descent


## Adam

  - It combines momentum and RMSprop
  - It's one of the few algorithms that generalise very well
  - Adam stands for Adaptive Moment Estimation
  - Here we have multiple hyperparameters: beta 1, beta 2 and epsilon, along with the usual learning rate


## Learning rate decay

  - With learning rate decay, you slowly reduce your leearning rate over time
  - If you keep the same value for alpha, then by the time the gd is approaching the minimum it will wonder around without exactly converging
  - If by the time the gd is approaching the minimum you decrease the learning rate, you will better chances to exactly converge
  - Decay rate here becomes another hyperparameter
  - You can include the epoch number as part of the equation that determines your learning rate decay


## The problem with local optima

  - Saddle points are points with derivatives 0
  - Plateaus are also a problem, because they will slow down the gradient descent
  - Local optima are less of a problem with NN that have a large number of parameters

