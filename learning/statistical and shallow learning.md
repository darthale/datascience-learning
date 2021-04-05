[WORK IN PROGRESS]


- [Intro](#intro)
- [Cross-validation](#cross-validation)
- [Regularization and model selection](#regularization-and-model-selection)
  * [Shrinkage methods](#shrinkage-methods)
- [Tree-Based methods](#tree-based-methods)
  * [Bagging, Random Forests, Boosting (Ensemble methods)](#bagging--random-forests--boosting--ensemble-methods-)
  * [Bagging](#bagging)
  * [Random Forests](#random-forests)
  * [Boosting](#boosting)
  * [Useful resources on ensemble learning](#useful-resources-on-ensemble-learning)

# Intro

The goal of this repository is to approach statistical learning through a mix of theory and real use cases.

As we proceed through new chapters, we'll briefly summarise what learnt and update a table of contents that could be useful to navigate the space for future projects.

# Bias Variance trade-off


# Cross-validation

Cross-validation:

- is part of those methods defined as *resampling methods*
- can be used to estimate the test error associated with a given statistical learning method in order to eveluate its performance. This is known as *model assessment*
- can be used to select the level of flexibility for a model. This is known as *model selection*
- k-fold is the most famous and probably most used

When we perform cross-validation, our goal might be to determine how well a given statistical learning procedure can be expected to perform on indipendent data: in this case we interested in estimate of the test MSE. Other times we are only interested int he location of the minimum point in the estimated test MSE curve. This is because we might want to perform cross-validation on multiple statistical learning methods, or on a single method using different levels of flexibility, in orer to identify the method with the lowet test error. 

The general procedure is as follows:

1. Shuffle the dataset randomly.
2. Split the dataset into k groups
3. For each unique group:
   1. Take the group as a hold out or test data set
   2. Take the remaining groups as a training data set
   3. Fit a model on the training set and evaluate it on the test set
   4. Retain the evaluation score and discard the model
4. Summarize the skill of the model using the sample of model evaluation scores

Importantly, each observation in the data sample is assigned to an individual group and stays in that group for the duration of the procedure. This means that each sample is given the opportunity to be used in the hold out set 1 time and used to train the model k-1 times.

# Feature Scaling: Normalization vs Standardization

https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/

https://www.statisticshowto.com/probability-and-statistics/normal-distributions/normalized-data-normalization

Why is important for NN and ML? For NN, standardizing features, makes the cost optimization process easier/quikcer since features have a similar scale and the learning rate can assume bigger values (improving the speed of gradient descent)


Very Important: when you want to scale your features, the scaling need to be calculated exculsively on your training data. If you include your test data, then you can incur in data leakage. At prediction time you will always use the same scaler fitted to the training data, because the assumptions is that the training dataset captures the distribution of your data. (https://datascience.stackexchange.com/questions/39932/feature-scaling-both-training-and-test-data/39933)

# Regularization and model selection

Here we talk about linear models and a different fitting procedure instead of least squares. These procedures can yield better:

- Prediction accuracy: if n >> p (where n is the number of obsversations and p the the number of variables) then the least squares estimates tend to have low variance (it will perform well on test data). If n is not much larger than p, there can be a lot of variability in the least squares. By constraining or shrinking the estimated coeffiecients, we can reduce the variance at the cost of a negligible increase in the bias. This can lead to big improvements in the accuracy of our predictions on test data
- Model interpretability: here we want to exclude variables that are not associated with the response. These variables would lead to unnecessary complexity in the resulting model. By setting the coeffiecent for these variables to zero we can obtain a model that is easier to interpret


There are different methods:

- subset selection: identify a subset of the p predictors and then fit the least squares on the reduced number of vars
- shrinkage: fitting a model to the p predictors and then shrink the estimated coeff towards zero. This is called regularization and has the effect of reducing variance. We can also end up with coefficients to zero and in this case the method will perform variables selection
- dimension reduction: PCA



## Shrinkage methods


The 2 most common methods are **Ridge Regression** and **Lasso Regression**. 

In both cases, the fit procedures are similiar to the least squares one: in addition we have a *tuning paramater* that has to be determined separately. The greater the value of this parameter the smaller will be the coeffienct esitmates. 

Here's a brief overview of the 2 methods:

1. **Ridge Regression**: 
   - Performs L2 regulatization: adds penalty equivalent to the square of the magnitude of the coefficients
   - It's advantage over least squares is rooted in the bias-variance trade-off. As the tuning parameter value increases, the flexibility of the regression fit decreases, leading to decreased variance but increased bias
   - It never sets coeffiecients to zero
   - It is mostly used to prevent overfitting
   - It works well in the presence of highly correlated features

2. **Lasso Regression**:
   - Performs L1 regularization, adds penalty equivalent to absolute value of the magnitude of the coefficients
   - Here coefficients can end up being zero (variable selections). Model generated by this type of regression are then easier to understand and interpret.
   - It yields to a sparse model (many coeff to zero)


We can use cross-validation to find the value for the tuning parameter:

1. We choose a grid of values for the parameter and compute the cross-validation errror for each value of the param
2. We then select the tuning parameter value for which the CV error is smallest
3. The model is re-fit using all of the available obsvervations and the selected value of the tuning parameter


# Tree-Based methods

In this chapter, we describe tree-based methods for regression and classification. These involve stratifying or segmenting the predictor space into a number of simple regions. In order to make a prediction for a given observation, we typically use the mean or the mode of the training observations in the region to which it belongs. Since the set of splitting rules used
to segment the predictor space can be summarized in a tree, these types of approaches are known as decision tree methods.
Tree-based methods are simple and useful for interpretation.

We also introduce bagging, random forests, and boosting. Each of these approaches involves producing multiple trees which are then combined to yield a single consensus prediction.

For a **regression tree**, the predicted response for an observation is given by the mean response of the training observations that belong to the same terminal node.

For a **classification tree**, we predict that each observation belongs to the most commonly occurring class of training observations in the region to which it belongs. In interpreting the results of a classification tree, we are often interested not only in the class prediction corresponding to a particular terminal node region, but also in the class proportions among the training observations that fall into that region.

Advantages and disadvantages of trees are:

- Trees are very easy to explain to people
- Trees can easily handle qualitative predictors without the need to
create dummy variables

- Trees generally do not have a good level of predictive accuracy 
- Trees can be very non-robust. A small change in the data can cause a large change in the final estimated tree


## Bagging, Random Forests, Boosting (Ensemble methods)

Aggregating many decision trees, using methods like bagging, random forests, and boosting, can lead to a substantial improvement in the predictive performance of trees.

Bagging, random forests, and boosting use trees as building blocks to construct more powerful prediction models.


## Bagging

Bootstrap aggregation, or bagging, is a general-purpose procedure for reducing the variance of a statistical learning method. It is particularly useful and frequently used in the context of decision trees.

A natural way to reduce the variance (and hence increase the prediction accuracy) of a statistical learning method is to take many training sets from the population, build a separate prediction model using each training set, and average the resulting predictions.

This is not practical because we generally do not have access to multiple training sets. Instead, we can bootstrap, by taking repeated samples from the (single) training data set. In this approach we generate B different bootstrapped training data sets. We then train our method on the b*th* bootstrapped training set in order to get many prediction models, and finally average
all the predictions. This is called **bagging**.

The process would look like:

1. Multiple subsets are created from the original dataset, selecting observations with replacement
2. A base model (weak model) is created on each of these subsets (each individual tree has high variance, but low bias)
3. The models run in parallel and are independent of each other
4. The final predictions are determined by combining the predictions from all the models

Bagging improves prediction accuracy at the cost of interpretability.
We can still calculate the importance of each variable in the resulting tree, either by using the RSS (for regression trees) or the Gini index (for classification trees).

## Random Forests

Random forests provide an improvement over bagged trees by way of a random small tweak that decorrelates the trees. 

As in bagging, we build a number forest of decision trees on bootstrapped training samples. But when building these decision trees, each time a split in a tree is considered, a random sample of m predictors is chosen as split candidates from the full set of p predictors. The split is allowed to use only one of those m predictors.

Suppose that there is one very strong predictor in the data set, along with a number of other moderately strong predictors. Then in the collection of bagged trees, most or all of the trees will use this strong predictor in the top split. Consequently, all of the bagged trees will look quite similar to each other. Hence the predictions from the bagged trees will be highly correlated. Unfortunately, averaging many highly correlated quantities does not lead to as large of a reduction in variance as averaging many uncorrelated quantities.

Random forests overcome this problem by forcing each split to consider only a subset of the predictors. We can think of this process as decorrelating the trees, thereby making the average of the resulting trees less variable and hence more reliable.

The main difference between bagging and random forests is the choice of predictor subset size m.


## Boosting

Boosting works in a similar way, except that the trees are grown sequentially: each tree is grown using information from previously grown trees. Boosting does not involve bootstrap sampling; instead each tree is fit on a modified version of the original data set.

Boosting reduces variance, and also reduces bias. It reduces variance because you are using multiple models (bagging). It reduces bias by training the subsequent model by telling him what errors the previous models made (the boosting part).


## Useful resources on ensemble learning

- https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-for-ensemble-models/
