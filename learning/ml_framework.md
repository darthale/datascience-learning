- [Intro](#intro)
- [1. Problem definition and dataset](#1-problem-definition-and-dataset)
- [2. Measuring success](#2-measuring-success)
- [3. Stategy to evaluate your model](#3-stategy-to-evaluate-your-model)
- [4. Data preparation](#4-data-preparation)
- [5. Model architecture](#5-model-architecture)
- [6. Developing a model that overfits](#6-developing-a-model-that-overfits)
- [7. Regularization](#7-regularization)


# Intro

The goal of this document is to define a generalised framework to leverage when approaching a machine learning problem. There are going to be references specific to deep learning, but all the elements of the framework can be applied to other statistical learning methods.

# 1. Problem definition and dataset

1. What will your input data be? What are you trying to predict? You can only learn to predict something if you have available training data: for example, you can
only learn to classify the sentiment of movie reviews if you have both movie reviews and sentiment annotations available. Data availability is usually the limiting factor at this stage 
2. What type of problem are you facing? Is it binary classification? Multiclass classification? Scalar regression? Vector regression? Multiclass, multilabel classification? Something else, like clustering, generation, or reinforcement learning? Identifying the problem type will guide your choice of model architecture, loss function, and so on.

You can’t move to the next stage until you know what your inputs and outputs are, and what data you’ll use. Be aware of the hypotheses you make at this stage:
 - You hypothesize that your outputs can be predicted given your inputs
 - You hypothesize that your available data is sufficiently informative to learn the relationship between inputs and outputs

Until you have a working model, these are merely hypotheses, waiting to be validated or invalidated. Not all problems can be solved; just because you’ve assembled examples of inputs X and targets Y doesn’t mean X contains enough information to predict Y.

One class of unsolvable problems you should be aware of is *nonstationary problems*. Suppose you’re trying to build a recommendation engine for clothing, you’re training
it on one month of data (August), and you want to start generating recommendations in the winter. One big issue is that the kinds of clothes people buy change from season
to season: clothes buying is a nonstationary phenomenon over the scale of a few months. What you’re trying to model changes over time. In this case, the right move is
to constantly retrain your model on data from the recent past, or gather data at a timescale where the problem is stationary. For a cyclical problem like clothes buying, a
few years’ worth of data will suffice to capture seasonal variation—but remember to make the time of the year an input of your model.

Keep in mind that machine learning can only be used to memorize patterns that are present in your training data. You can only recognize what you’ve seen before. Using machine learning trained on past data to predict the future is making the assumption that the future will behave like the past. That often isn’t the case. 

# 2. Measuring success 

To control something, you need to be able to observe it. To achieve success, you must
define what you mean by success—accuracy? Precision and recall? Customer-retention
rate? Your metric for success will guide the choice of a loss function: what your model
will optimize. It should directly align with your higher-level goals, such as the success
of your business.


For balanced-classification problems, where every class is equally likely, accuracy and
area under the receiver operating characteristic curve (ROC AUC) are common metrics. For
class-imbalanced problems, you can use precision and recall. For ranking problems or
multilabel classification, you can use mean average precision. And it isn’t uncommon
to have to define your own custom metric by which to measure success. 

# 3. Stategy to evaluate your model

Once you know what you’re aiming for, you must establish how you’ll measure your
current progress. Three common approaches are:
1. Maintaining a hold-out validation set: the way to go when you have plenty of
data
2. Doing K-fold cross-validation: the right choice when you have too few samples
for hold-out validation to be reliable
3. Doing iterated K-fold validation: for performing highly accurate model evaluation when little data is available

Just pick one of these. 


# 4. Data preparation

Once you know what you’re training on, what you’re optimizing for, and how to evaluate your approach, you’re almost ready to begin training models. But first, you should format your data in a way that can be fed into a machine-learning model—here:

For NN:

 - Your data should be formatted as tensors.
 - The values taken by these tensors should usually be scaled to small values: for example, in the [-1, 1] range or [0, 1] range.
 - You may want to do some feature engineering, especially for small-data problems.
 - If different features take values in different ranges (heterogeneous data), then the data should be normalized.


# 5. Model architecture

Your goal at this stage is to achieve statistical power: that is, to develop a small model that is capable of beating a dumb baseline. 


**Note**: *it’s not always possible to achieve statistical power. If you can’t beat a random baseline after trying multiple reasonable architectures, it may be that the answer
to the question you’re asking isn’t present in the input data. Remember that you make
two hypotheses:
- You hypothesize that your outputs can be predicted given your inputs.
- You hypothesize that the available data is sufficiently informative to learn the
relationship between inputs and outputs.


It may well be that these hypotheses are false, in which case you must go back to the
drawing board.*

Assuming that things go well, you need to make three key choices to build your first working model:

1. *Last-layer activation*: this establishes useful constraints on the network’s output. For a classification problem you should use sigmoid; for a regression one you don't need a last-layer artivation, etc. 
2. *Loss function*: this should match the type of problem you’re trying to solve. For
a binary classification problem you could use **binary_crossentropy**, a regression one you could use **mse**, and so on.
3. *Optimization configuration*: what optimizer will you use? What will its learning rate be? 


Regarding the choice of a loss function, note that it isn’t always possible to directly
optimize for the metric that measures success on a problem. Sometimes there is no easy way to turn a metric into a loss function; loss functions, after all, need to be computable given only a mini-batch of data (ideally, a loss function should be computable for as little as a single data point) and must be differentiable (otherwise, you can’t use backpropagation to train your network). 

For instance, the widely used classification metric ROC AUC can’t be directly optimized. Hence, in classification tasks, it’s common to optimize for a proxy metric of ROC AUC, such as crossentropy. In general, you can hope that the lower the crossentropy gets, the higher the ROC AUC will be.

# 6. Developing a model that overfits

Once you’ve obtained a model that has statistical power, the question becomes, is your
model sufficiently powerful? Does it have enough layers and parameters to properly
model the problem at hand  Remember that the universal tension in machine learning is between
optimization and generalization; the ideal model is one that stands right at the border
between underfitting and overfitting; between undercapacity and overcapacity. To figure out where this border lies, first you must cross it. To figure out how big a model you’ll need, you must develop a model that overfits.


This is fairly easy:

1. Add layers.
2. Make the layers bigger.
3. Train for more epochs.

Always monitor the training loss and validation loss, as well as the training and validation values for any metrics you care about. When you see that the model’s performance on the validation data begins to degrade, you’ve achieved overfitting.

The next stage is to start regularizing and tuning the model, to get as close as possible to the ideal model that neither underfits nor overfits. 


# 7. Regularization

This step will take the most time: you’ll repeatedly modify your model, train it, evaluate on your validation data (not the test data, at this point), modify it again, and repeat, until the model is as good as it can get. 


For NN, these are some things you should try:
 - Add dropout.
 - Try different architectures: add or remove layers.
 - Add L1 and/or L2 regularization.
 - Try different hyperparameters (such as the number of units per layer or the learning rate of the optimizer) to find the optimal configuration.
 - Optionally, iterate on feature engineering: add new features, or remove features that don’t seem to be informative.


*Be mindful of the following: every time you use feedback from your validation process to tune your model, you leak information about the validation process into the model.*

Repeated just a few times, this is innocuous; but done systematically over many iterations, it will eventually cause your model to overfit to the validation process (even though no model is directly trained on any of the validation data). This makes the evaluation process less reliable.


Once you’ve developed a satisfactory model configuration, you can train your final production model on all the available data (training and validation) and evaluate it
one last time on the test set. If it turns out that performance on the test set is significantly worse than the performance measured on the validation data, this may mean either that your validation procedure wasn’t reliable after all, or that you began overfitting to the validation data while tuning the parameters of the model. In this case, you may want to switch to a more reliable evaluation protocol (such as iterated K-fold
validation). 
