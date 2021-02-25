# Feature Scaling: Normalization vs Standardization

https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/

https://www.statisticshowto.com/probability-and-statistics/normal-distributions/normalized-data-normalization

Why is important for NN and ML? For NN, standardizing features, makes the cost optimization process easier/quikcer since features have a similar scale and the learning rate can assume bigger values (improving the speed of gradient descent)


Very Important: when you want to scale your features, the scaling need to be calculated exculsively on your training data. If you include your test data, then you can incur in data leakage. At prediction time you will always use the same scaler fitted to the training data, because the assumptions is that the training dataset captures the distribution of your data. (https://datascience.stackexchange.com/questions/39932/feature-scaling-both-training-and-test-data/39933)