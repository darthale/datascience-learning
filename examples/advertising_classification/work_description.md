# Advertising dataset: a classification problem

The goal of this exercise is to show how to use Amazon Sagemaker pre-built container with a classification problem.

## Dataset

The dataset is hosted in kaggle and can be found here: https://www.kaggle.com/fayomi/advertising.

Based on a set of predictors and the associated outcome, we want to estimate whether a click on an ad will happen or not.

We are asked to predict a categorial indipendent variable, based on labelled data. This tells us: supervised learning --> classification problem.

## File and notebooks

The adveritsing classification directory contains: 

- data_exploration.ipynb: a notebook where some basic data exploration has ben done and also few logistic regression models have been fitted
- logistic_training.py: a python script that fits a logistic regression model on training data 
- model_training_prediction_evaluation.ipynb: a notebook writte in Sagemaker notebook. 


## Sagemaker approach

Since the goal of this repository is to teach to myself how to use Sagemaker, here I briefly describe what I tried to achieve through this problem. 

Going through the model_training_prediction_evaluation notebook we can see how:

- Data is split into train and test and stored in the S3 bucket associated with the notebook
- A SKLearn estimator is instantiated (pointing to the logistic_trainin.py script)
- A model is fitted through the estimator above on some training data and stored as artifact in the S3 bucket associated
- A batch transform object is instantiated (pointing to the artifact generated in the training phase)
- Predictions are made on the test files through batch transform

## Use cases

We could devise the following use cases from looking at this process:

1. Train a model locally, upload artifact to S3 and use batch transform to make batch prediction.
2. Train a model in Sagemaker use its compute power
3. Train a model in Sagemaker through pre-built container and training code stored in a python file
4. Train a model in Sagemaker, download the artifact from S3 locally and integrate that trained model in some custom application
