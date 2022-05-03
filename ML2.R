#########################################
#   Machine Learning HW2                #
#   15Nov2021                           #
#   Predicting Annuity Purchase         #
#########################################


#Load libraries
library(tidyverse)
library(caret)
library(leaps)
library(glmnet)
library(ggplot2)
library(earth)
library(mgcv)
library(ROCR)
library(InformationValue)
library(funModeling)
library(ClustOfVar)
library(randomForest)
library(xgboost)
library(Ckmeans.1d.dp)
library(pdp)

#Read in data
setwd('C:/Users/Richard Pincus/Documents/Classes - MSA/AA502/Machine Learning/hw/hw2/Homework2_ML')
train = read.csv('insurance_t.csv')

train0 = train

#Set alpha level
a_samp = 1-pchisq(log(nrow(train)),1)



#Impute missing values from before 
train1 = train %>%
  rowwise() %>%
  #Creat flag var
  mutate(CCPURC_Missing = as.factor(as.numeric(is.na(CCPURC)))) %>%
  mutate(CC_Missing = as.factor(as.numeric(is.na(CC)))) %>%
  mutate(INV_Missing = as.factor(as.numeric(is.na(INV)))) %>%
  mutate(CRSCORE_Missing = as.factor(as.numeric(is.na(CRSCORE)))) %>%
  mutate(AGE_Missing = as.factor(as.numeric(is.na(AGE)))) %>%
  mutate(HMVAL_Missing = as.factor(as.numeric(is.na(HMVAL)))) %>%
  mutate(LORES_Missing = as.factor(as.numeric(is.na(LORES)))) %>%
  mutate(INCOME_Missing = as.factor(as.numeric(is.na(INCOME)))) %>%
  mutate(CCBAL_Missing = as.factor(as.numeric(is.na(CCBAL)))) %>%
  mutate(INVBAL_Missing = as.factor(as.numeric(is.na(INVBAL)))) %>%
  mutate(POSAMT_Missing = as.factor(as.numeric(is.na(POSAMT)))) %>%
  mutate(POS_Missing = as.factor(as.numeric(is.na(POS)))) %>%
  mutate(PHONE_Missing = as.factor(as.numeric(is.na(PHONE)))) %>%
  mutate(ACCTAGE_Missing = as.factor(as.numeric(is.na(ACCTAGE)))) %>% 
  
  mutate(CCPURC = replace_na(CCPURC,'M')) %>%
  mutate(CC = replace_na(CC,'M')) %>%
  mutate(INV = replace_na(INV, 'M')) %>%
  mutate(CRSCORE = replace_na(CRSCORE, 666.0)) %>%
  mutate(AGE = replace_na(AGE, 48.0)) %>%
  mutate(HMVAL = replace_na(HMVAL, 107.0)) %>%
  mutate(LORES = replace_na(LORES, 6.50)) %>%
  mutate(INCOME = replace_na(INCOME, 35.0)) %>%
  mutate(CCBAL = replace_na(CCBAL, 0)) %>%
  mutate(INVBAL = replace_na(INVBAL, 0)) %>%
  mutate(POSAMT = replace_na(POSAMT, 0)) %>%
  mutate(POS = replace_na(POS, 0)) %>%
  mutate(PHONE = replace_na(PHONE, 0)) %>%
  mutate(ACCTAGE = replace_na(ACCTAGE, 0))



train1a = train1 %>% mutate(CCPURC = as.factor(CCPURC), 
                            CC = as.factor(CC), 
                            INV = as.factor(INV))




#Subset variables to those used in previous GAM
train2 = train1a %>% dplyr::select(INS, ACCTAGE, ACCTAGE_Missing, DDA, DDABAL, CHECKS, TELLER, SAVBAL, ATMAMT, CD, 
                          IRA, INV, INV_Missing, MM, CC, CC_Missing, BRANCH) %>% 
                    mutate(INS = as.factor(INS))


train2.df <- as.data.frame(train2)
# str(train2.df)

# Random Forest model
set.seed(11)
rf1 = randomForest(INS ~ ., data = train2.df, ntree = 500, importance = TRUE)


# Plot the change in error across different number of trees
plot(rf1, main = "Number of Trees Compared to MSE")
#Error stabliszes around 100 trees

#Variable importance
varImpPlot(rf1,
           sort = TRUE,
           # n.var = 10,
           main = "Top 10 - Variable Importance")
importance(rf1)



# Tune an random forest mtry value
set.seed(11)
tuneRF(x = train2.df[,-1], y = train2.df[,1], 
       plot = TRUE, ntreeTry = 100, stepFactor = 0.5)
#mtry = 4 is the lowest OOB Error

set.seed(11)
rf2 = randomForest(INS ~ ., data = train2.df, ntree = 100, mtry = 4, importance = TRUE)

varImpPlot(rf2,
           sort = TRUE,
           # n.var = 14,
           main = "Order of Variables")
importance(rf2, type = 1)

#I say keep all vars for the sake of prediction



#Get predictions
p_hat_rf = predict(rf2, type = "prob")
train2.df$p_hat_rf = p_hat_rf[,2]
train2.df$pred_rf = predict(rf2, type = "response")



#Create ROC curve and get AUROC


#Plot the ROC Curve
plotROC(train2.df$INS, train2.df$p_hat_rf)
plotROC(train2.df$INS, p_hat_rf)

Concordance(train2.df$INS, train2.df$p_hat_rf)







#Build an XGBoost model


#Create dataframes for target ad predictors
train_x = model.matrix(INS ~ ., data = train2)[, -1]
train_y = ifelse(as.numeric(train2$INS) == 1, 0, 1)
# train_y = as.numeric(train2$INS)
class(train_x)

#Check if target mapped properly
sum(train_y == train2$INS) == nrow(train2)

#Build XGBoost model
set.seed(11)
xgb1 <- xgb.cv(data = train_x, label = train_y, subsample = 1, nrounds = 100, nfold=10, objective = "binary:logistic", eval_metric='auc')

# arrange(xgb1$evaluation_log, test_error_mean )

arrange(xgb1$evaluation_log, desc(test_auc_mean) )

#Get index of minimum logloss value for optimal nrounds
# nrounds.in = xgb1$evaluation_log$iter[xgb1$evaluation_log$test_logloss_mean == min(xgb1$evaluation_log$test_logloss_mean)]
# nrounds.in = xgb1$evaluation_log$iter[xgb1$evaluation_log$test_error_mean == min(xgb1$evaluation_log$test_error_mean)]
nrounds.in = xgb1$evaluation_log$iter[xgb1$evaluation_log$test_auc_mean == max(xgb1$evaluation_log$test_auc_mean)]


#the nrounds with the lowest mean logloss is 10, error is 9, 
# AUC is 13 so go with 13 for tuning




#Tuning through caret
tune_grid = expand.grid(
  nrounds = 13,
  eta = c(0.05, 0.1, 0.15, 0.2 , 0.25, 0.3, 0.35, 0.4),
  max_depth = c(1:10),
  gamma = c(0, 1, 2, 3),
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = c(0.25, 0.5, 0.75, 1)
)


set.seed(11)
xgb3 = caret::train(x = train_x, y = factor(train_y),
                        method = "xgbTree",
                        tuneGrid = tune_grid,
                        trControl = trainControl(method = 'cv', # Using 10-fold cross-validation
                                                 number = 10))

plot(xgb3)


#Best params:
#   gamma = 3
#   subsample = 0.5
#   depth = 4
#   eta = 0.3


#Variable importance
xgb.final <- xgboost(data = train_x, label = train_y, subsample = 0.5, nrounds = 10, eta = 0.3, gamma = 3, max_depth = 4, objective = "binary:logistic", eval_metric='auc')

xgb.importance(feature_names = colnames(train_x), model = xgb.final)

xgb.ggplot.importance(xgb.importance(feature_names = colnames(train_x), model = xgb.final))



#Get predictions
pred <- predict(xgb.final, train_x)

#Plot the ROC Curve
plotROC(train2.df$INS, pred)




##Try something with crazy long runtime just for fun
#Tuning through caret
tune_grid = expand.grid(
  nrounds = 13,
  eta = c(0.05, 0.1, 0.15, 0.2 , 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95),
  max_depth = c(1:10),
  gamma = c(0, 1, 2, 3, 4, 5),
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = c(0.25, 0.5, 0.75, 1)
)

set.seed(11)
xgb4 = caret::train(x = train_x, y = factor(train_y),
                    method = "xgbTree",
                    tuneGrid = tune_grid,
                    trControl = trainControl(method = 'cv', # Using 10-fold cross-validation
                                             number = 10))

plot(xgb4)



#Best params:
#   gamma = 4
#   subsample = 0.75
#   depth = 3 or 10
#   eta = 0.4


#Variable importance
xgb.final2 <- xgboost(data = train_x, label = train_y, subsample = 0.75, nrounds = 13, eta = 0.4, gamma = 4, max_depth = 10, objective = "binary:logistic", eval_metric='auc')

xgb.importance(feature_names = colnames(train_x), model = xgb.final2)

xgb.ggplot.importance(xgb.importance(feature_names = colnames(train_x), model = xgb.final2))



#Get predictions
pred2 <- predict(xgb.final2, train_x)

#Plot the ROC Curve
plotROC(train2.df$INS, pred2)


