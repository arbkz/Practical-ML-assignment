---
title: "Weight Lifting Data Analysis"
author: "arbkz"
date: "6/16/2019"
output: 
  html_document: 
    fig_height: 8
    fig_width: 8
    keep_md: yes
---


## Intro

The goal of this analysis is to use the Weight Lifting Exercises dataset to train a predictor that can determine how well a dumbell bicep curl is performed, and help identify common errors in weight lifting technique.

The data used to train the model is the [*Weight Lifting Exercises*](http://groupware.les.inf.puc-rio.br/har) dataset.

We split the data into training/testing and validation sets and use the training data to train 3 component models (LDA, KNN and RF).
We then use the testing data to tune the models in terms of what cross validation and preprocessing methods to use.  
We then combine/stack the component models using a random forest model and use this ensemble model to predict the class for 20 datapoints in the validation set.


## The Data

This analysis uses data from the [*Weight Lifting Exercises dataset*](http://groupware.les.inf.puc-rio.br/har). 

Six  test subjects participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different ways: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E).

The link to the data for this project can be found in the references section

## Data Processing 

We download and load the orinal testing and training data into training and testing variables.

The original training data consists of 19622 rows and 160 different columns and we see that there are 100 columns which have 97%+ NA values.
Since certain models like Random Forests will not cope well with NA values, rather than impute the data from such limited sample set we will remove these columns from the analysis all together.

We will also eliminate the user_name, timestamp and window information as we are not doing any time series forecasting and it doesnt make sense to include things like username in a generalised prediction model.

We will relabel the original testing dataset as **validation_set** and use data this data later to test our model and generate the predictions for the quiz.
We then split the training data (70/30) further and label these as **training_set** and **testing_set**. 


```r
library(caret)
library(dplyr)
library(ggplot2)
library(GGally)

library(doParallel)
clust <- makePSOCKcluster(6)
registerDoParallel(clust)

training_data_url <- 'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv'
testing_data_url <- 'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv'

#setwd("~/Documents/R\ Projects/coursera/practical_machine_learning_wk4_assignment")

if (!file.exists('./sample_data'))     dir.create('./sample_data')
if (!file.exists('./sample_data/pml-training.csv'))   download.file(testing_data_url,'./sample_data/pml-testing.csv', method = 'curl')
if (!file.exists('./sample_data/pml-testing.csv'))    download.file(training_data_url,'./sample_data/pml-training.csv', method = 'curl')

training <- read.csv('./sample_data/pml-training.csv', stringsAsFactors = T, na.strings = c('#DIV/0!', 'NaN', 'NA'))
testing <- read.csv('./sample_data/pml-testing.csv', stringsAsFactors = T, na.strings = c('#DIV/0!', 'NaN', 'NA'))

summary_na <- apply(training, 2, function(a) {sum(is.na(a)) / length(a)})
in_summary_na <- summary_na < 0.95 

training <- training[in_summary_na]
training <- subset(training, select = -c(1:7))

validation_set <- testing[in_summary_na]
validation_set <- subset(validation_set, select = -c(1:7))

set.seed(8738)

inTrain <- createDataPartition(y=training$classe, p = 0.7, list = F)
training_set <- training[inTrain,]
testing_set <- training[-inTrain,]
```

## Exploratory Data Analysis

Even after removing 100+ columns there are still too many columns/covariates to do a pairs plot of them all.

Instead we take smaller subset of measurements and then look at the relationships in pairs plot.

We can see the mean total_accel_belt and roll_belt is much lower for class A than the other classes but in general it is hard to see anything clearly from the data due to the sheer number of data points and covariates.



```r
dim(training_set)
```

```
## [1] 13737    53
```

```r
ggpairs(data.frame(training_set[,grep("total", names(training_set))],training_set$classe))
```

```
## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.
## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.
## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.
## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.
```

![](practical_machine_learning_week_4_assigment_files/figure-html/eda-1.png)<!-- -->

```r
ggpairs(data.frame(training_set[,grep("roll", names(training_set))],training_set$classe))
```

```
## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.
## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.
## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.
## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.
```

![](practical_machine_learning_week_4_assigment_files/figure-html/eda-2.png)<!-- -->

## Model Selection

As we need to classify the quality of the exercise technique into multiple different categories there are limited options in terms of the model types we can use. 

We will build 3 models: Linear Discriminant Analysis, Random Forest and K Nearest Neighbors.
Then final model will then be  an ensemble of the 3 different models which are combined using random forest model.

We will pre-process the data for using a lda model based estimator using the Box-Cox transformation to normalise the data.

Other pre-processing including pca, scale and centre was attempted but resulted in inferior prediction accuracy on the testing data set.

We use the training_set to train our model and then the testing_set to create a combined model and select which model to use for final validation.

We use cross-validation to train the random-forest with the default setting (k folds with k = 10). 
After trying various different options, it appears that using cross-validation in the LDA or KNN model training did not appear to improve the model fit so in the end these models is built off the raw data. 



```r
set.seed(7737)
```

### Random Forest model

With a random forest model and we get a very good in-sample accuracy with no real tuning required. 
The drawback is that final model is around (52 Mb) and it takes quite a while to build.


```r
mod1_rf <- train(classe ~ . , method = "rf", data = training_set, trControl = trainControl(method = "cv"), allowParalell = TRUE)
pred_rf <- predict(mod1_rf, testing_set)
confusionMatrix(pred_rf,testing_set$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673   12    0    0    0
##          B    1 1123    2    0    1
##          C    0    3 1023    6    1
##          D    0    1    1  958    3
##          E    0    0    0    0 1077
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9947          
##                  95% CI : (0.9925, 0.9964)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9933          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9994   0.9860   0.9971   0.9938   0.9954
## Specificity            0.9972   0.9992   0.9979   0.9990   1.0000
## Pos Pred Value         0.9929   0.9965   0.9903   0.9948   1.0000
## Neg Pred Value         0.9998   0.9966   0.9994   0.9988   0.9990
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2843   0.1908   0.1738   0.1628   0.1830
## Detection Prevalence   0.2863   0.1915   0.1755   0.1636   0.1830
## Balanced Accuracy      0.9983   0.9926   0.9975   0.9964   0.9977
```

```r
# If we want to cut down the number of variables in the model and just take the top 20 variable svarImp(mod1_rf)
```

We can look at simplifying the rf model by using only the top 20 covariates from the full model in terms of importance.


```r
top_covariates_rf <- mod1_rf$finalModel$importance
covariate_rank <- sort(sapply(top_covariates_rf, function(el) { el[1]}), index.return = TRUE,decreasing = TRUE) 
keep_list <- covariate_rank$ix[1:20]

min_training_set <- subset(training_set, select = keep_list)
min_training_set$classe <- training_set$classe
min_testing_set <- subset(testing_set, select = keep_list)
min_testing_set$classe <- testing_set$classe

mod4_rf_min <- train(classe ~ . , method = "rf", data = min_training_set, trControl = trainControl(method = "cv"), allowParalell = TRUE)
pred_rf_min <- predict(mod4_rf_min, min_testing_set)

confusionMatrix(pred_rf_min,min_testing_set$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1671    9    0    1    0
##          B    3 1119    6    0    0
##          C    0   11 1018    9    3
##          D    0    0    2  954    5
##          E    0    0    0    0 1074
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9917         
##                  95% CI : (0.989, 0.9938)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9895         
##                                          
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9982   0.9824   0.9922   0.9896   0.9926
## Specificity            0.9976   0.9981   0.9953   0.9986   1.0000
## Pos Pred Value         0.9941   0.9920   0.9779   0.9927   1.0000
## Neg Pred Value         0.9993   0.9958   0.9983   0.9980   0.9983
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2839   0.1901   0.1730   0.1621   0.1825
## Detection Prevalence   0.2856   0.1917   0.1769   0.1633   0.1825
## Balanced Accuracy      0.9979   0.9903   0.9937   0.9941   0.9963
```

We can see that the resulting model has in-sample accuracy of 99%+ but is smaller and trains much faster. 
We could use this model if complexity and performance is a major bottleneck.

### LDA model

Next we will build a Linear Discriminant Model.

When using this kind of model based predictor it's important to pre-process the variables, so we use Box-Cox to normalise the variables and ensure all the covariates are on a similar scale.

Other preprocessing options like centering and scaling were attempted but not used in the final model as they decreased the in-sample accuracy.


```r
mod3_lda <- train(classe ~ ., data = training_set, method = "lda", preProcess = c("BoxCox"))

pred_lda <- predict(mod3_lda, testing_set)
confusionMatrix(pred_lda,testing_set$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1366  195   98   64   29
##          B   39  694   97   47  180
##          C  132  146  684  113  109
##          D  132   32  116  705  106
##          E    5   72   31   35  658
## 
## Overall Statistics
##                                          
##                Accuracy : 0.6979         
##                  95% CI : (0.686, 0.7096)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.6176         
##                                          
##  Mcnemar's Test P-Value : < 2.2e-16      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8160   0.6093   0.6667   0.7313   0.6081
## Specificity            0.9083   0.9235   0.8971   0.9216   0.9702
## Pos Pred Value         0.7797   0.6566   0.5777   0.6462   0.8215
## Neg Pred Value         0.9255   0.9078   0.9272   0.9460   0.9166
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2321   0.1179   0.1162   0.1198   0.1118
## Detection Prevalence   0.2977   0.1796   0.2012   0.1854   0.1361
## Balanced Accuracy      0.8622   0.7664   0.7819   0.8264   0.7892
```

The LDA model gives us a decent accuracy (~70%) across the board and it is quick to build and only 6.2Mb.


### K Nearest Neighbors model

K Nearest neighbors is a good option for this kind of classification problem with multiple categories.
After experimenting with cross validation it seems like it doesn't improve the in-sample accuracy, so we will use the train function with no CV.


```r
mod5_knn <- train(classe ~ . , method = "knn", data = min_training_set)
pred_knn <- predict(mod5_knn, testing_set)
confusionMatrix(pred_knn,testing_set$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1639   41    8    7    1
##          B   23 1002   36    4   36
##          C    3   58  944   39   19
##          D    9   20   24  904   34
##          E    0   18   14   10  992
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9314          
##                  95% CI : (0.9246, 0.9377)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9132          
##                                           
##  Mcnemar's Test P-Value : 6.692e-07       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9791   0.8797   0.9201   0.9378   0.9168
## Specificity            0.9865   0.9791   0.9755   0.9823   0.9913
## Pos Pred Value         0.9664   0.9101   0.8881   0.9122   0.9594
## Neg Pred Value         0.9916   0.9714   0.9830   0.9877   0.9814
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2785   0.1703   0.1604   0.1536   0.1686
## Detection Prevalence   0.2882   0.1871   0.1806   0.1684   0.1757
## Balanced Accuracy      0.9828   0.9294   0.9478   0.9600   0.9540
```


### Final Model

The final model combines the 3 models created above using a random forest model.


```r
pred_DF <- data.frame(pred_rf = pred_rf, pred_knn = pred_knn, pred_lda = pred_lda, classe = testing_set$classe)


mod_comb1 <- train(classe ~., method = "rf", data = pred_DF, trControl = trainControl(method = "cv"))
pred_comb1 <- predict(mod_comb1, pred_DF)
confusionMatrix(pred_comb1,testing_set$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673   12    0    0    0
##          B    1 1123    2    0    1
##          C    0    3 1023    6    1
##          D    0    1    1  958    3
##          E    0    0    0    0 1077
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9947          
##                  95% CI : (0.9925, 0.9964)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9933          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9994   0.9860   0.9971   0.9938   0.9954
## Specificity            0.9972   0.9992   0.9979   0.9990   1.0000
## Pos Pred Value         0.9929   0.9965   0.9903   0.9948   1.0000
## Neg Pred Value         0.9998   0.9966   0.9994   0.9988   0.9990
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2843   0.1908   0.1738   0.1628   0.1830
## Detection Prevalence   0.2863   0.1915   0.1755   0.1636   0.1830
## Balanced Accuracy      0.9983   0.9926   0.9975   0.9964   0.9977
```
In the testing_set this doesn't  improve the accuracy over the Random Forest on its own but this may be as a result of Random Forest overfitting on the training data.
We can assume that the performance of the stacked model will be better in the real world if we see that rf performance is degraded.

## Final Predictions

Finally we run the model on the validation set to get our final predictions.

Our out-of-sample error estimation for the final model is around 0.53% and based on the quiz answers the model predicted correctly for all 20 test cases. 
  

```r
pred_final_rf <- predict(mod1_rf, validation_set)
pred_final_rf_min <- predict(mod4_rf_min, validation_set)
pred_final_lda <- predict(mod3_lda, validation_set)
pred_final_knn <- predict(mod5_knn, validation_set)

pred_DF_final <- data.frame(pred_rf = pred_final_rf, pred_knn = pred_final_knn, pred_lda = pred_final_lda)

pred_final_comb <- predict(mod_comb1, pred_DF_final)
pred_final_comb
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

```r
mod_comb1$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 2
## 
##         OOB estimate of  error rate: 0.53%
## Confusion matrix:
##      A    B    C   D    E  class.error
## A 1673    1    0   0    0 0.0005973716
## B   12 1123    3   1    0 0.0140474100
## C    0    2 1023   1    0 0.0029239766
## D    0    0    6 958    0 0.0062240664
## E    0    1    1   3 1077 0.0046210721
```

## References
This analysis was made possible by the work of Wallace Ugulino, Eduardo Velloso, Hugo Fuks.

Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13). Stuttgart, Germany: ACM SIGCHI, 2013. [Link](http://groupware.les.inf.puc-rio.br/har#sbia_paper_section#ixzz5r8PNujCr)

More information on the dataset is available from the website [here](http://groupware.les.inf.puc-rio.br/har)

link to data:

* [**Training Data**](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv)
* [**Testing Data**](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv)
