#########################################
#   Machine Learning HW3                #
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
library(nnet)
library(NeuralNetTools)
library(reshape2)
library(e1071)
library(pdp)
library(ALEPlot)
library(lime)
library(iml)
library(patchwork)


#Read in data
setwd('C:/Users/Richard Pincus/Documents/Classes - MSA/AA502/Machine Learning/hw/hw3/Homework3_ML')
train = read.csv('insurance_t.csv')
val = read.csv('insurance_v.csv')

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



train2 = train1 %>% mutate(CCPURC = as.factor(CCPURC), 
                           CC = as.factor(CC), 
                           INV = as.factor(INV) 
                           )

xcolnames = colnames(train2)
need.facts = NA
for (i in 1:ncol(train2)){
  if (nrow(unique(train2[,i])) < 10 & !is.factor(train2[,i])){
    print(xcolnames[i])
    need.facts[i] = xcolnames[i]
  }
}

#Get columns names needing to be converted to factors
need.facts = need.facts[!is.na(need.facts)]

#Extract and convert columns to factors 
train2_fact = train2 %>% select(all_of(need.facts)) %>% mutate(across(.cols = everything(), .fns = as.factor))

#Add columns back to original data
train2 = train2 %>% select(-all_of(need.facts)) %>% cbind(train2_fact)




#Transform Validation in the same way
val1 = val %>%
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



val2 = val1 %>% mutate(CCPURC = as.factor(CCPURC), 
                           CC = as.factor(CC), 
                           INV = as.factor(INV) 
)

xcolnamesv = colnames(val2)
need.factsv = NA
for (i in 1:ncol(val2)){
  if (nrow(unique(val2[,i])) < 10 & !is.factor(val2[,i])){
    print(xcolnamesv[i])
    need.factsv[i] = xcolnamesv[i]
  }
}

#Get columns names needing to be converted to factors
need.factsv = need.factsv[!is.na(need.factsv)]

#Extract and convert columns to factors 
val2_fact = val2 %>% select(all_of(need.factsv)) %>% mutate(across(.cols = everything(), .fns = as.factor))

#Add columns back to original data
val2 = val2 %>% select(-all_of(need.factsv)) %>% cbind(val2_fact)









#Cluster train variables
factor.vars = NA
for (i in 1:ncol(train2)){
  factor.vars[i] = is.factor(train2[,i])
}

#Get list of var names that are quantitative and qualitative
qual.var.list = c(colnames(train2)[factor.vars], 'BRANCH')
quant.var.list = colnames(train2)[!(colnames(train2) %in% qual.var.list)]

#Get quant and qual vars
quant.var = train2 %>% dplyr::select(all_of(quant.var.list))
qual.var = train2 %>% dplyr::select(all_of(qual.var.list))

#Cluster variables to reduce dimensions
var.clust.h = hclustvar(quant.var, qual.var)
stab=stability(var.clust.h,B=25) ## This will take time!
plot(stab)
sort(-stab$meanCR) #16 clusters is the winner
h16 = cutreevar(var.clust.h, 16)
h16$var


#Select variables from clustering
train3 = train2 %>% select(INS, ACCTAGE, ACCTAGE_Missing, DEPAMT, DEP, NSF, SAV, POS, POS_Missing, CD, IRA, INVBAL, MM, INCOME, INCOME_Missing, AGE, AGE_Missing, CC, CC_Missing)





##
##
## Neural Net
##
##


# Standardizing Continuous Variables
train4 = train3 %>%
  mutate(s_ACCTAGE = scale(ACCTAGE),
         s_DEPAMT = scale(DEPAMT),
         s_DEPAMT = scale(DEP),
         s_POS = scale(POS),
         s_INVBAL = scale(INVBAL),
         s_INCOME = scale(INCOME),
         s_AGE = scale(AGE)) %>% 
  select(-ACCTAGE, -DEPAMT, -DEP, -POS, -INVBAL, -INCOME, -AGE)




#Build a neural net
# Neural Network model
set.seed(12)
nn1 <- nnet(INS ~ s_ACCTAGE + ACCTAGE_Missing + s_DEPAMT + s_DEPAMT + NSF + SAV + s_POS + POS_Missing + 
                  CD + IRA + s_INVBAL + MM + s_INCOME + INCOME_Missing + s_AGE + AGE_Missing + CC + CC_Missing,
                  data = train4, size = 5)

#PLot
plotnet(nn1)



# Optimize Number of Hidden Nodes and Regularization (decay option)
tune_grid1 <- expand.grid(
  .size = c(3, 4, 5, 6, 7),
  .decay = c(0, 0.5, 1)
)

#Tune
set.seed(12)
nn1.caret <- train(INS ~ s_ACCTAGE + ACCTAGE_Missing + s_DEPAMT + s_DEPAMT + NSF + SAV + s_POS + POS_Missing + 
                         CD + IRA + s_INVBAL + MM + s_INCOME + INCOME_Missing + s_AGE + AGE_Missing + CC + 
                         CC_Missing,
                       data = train4,
                       method = "nnet", # Neural network using the nnet package
                       tuneGrid = tune_grid1,
                       trControl = trainControl(method = 'cv', # Using 10-fold cross-validation
                                                number = 10),
                       trace = FALSE)

nn1.caret$bestTune



#Build best model
set.seed(12)
nn2 <- nnet(INS ~ s_ACCTAGE + ACCTAGE_Missing + s_DEPAMT + s_DEPAMT + NSF + SAV + s_POS + POS_Missing + 
                  CD + IRA + s_INVBAL + MM + s_INCOME + INCOME_Missing + s_AGE + AGE_Missing + CC + CC_Missing,
                  data = train4, size = 3, decay = 1)

#PLot
plotnet(nn2)



#Get predictions
pred1 = (predict(nn2, type='raw'))
pred1


#Plot the ROC Curve
plotROC(train4$INS, pred1) 




############################
############################
#This will take long to run
############################
############################


#Scale all quantitative variables
quant.var.s = quant.var %>% mutate(across(.cols = everything(), .fns = scale))

#Combine qual.vars and sclae qaunt.vars.s
train5 = cbind(qual.var, quant.var.s)


#Train NNet with all vars

# Optimize Number of Hidden Nodes and Regularization (decay option)
tune_grid1.5 <- expand.grid(
  .size = seq(3, 15, 1),
  .decay = c(0, 0.5, 1)
)

#Tune
set.seed(12)
nn.try.caret <- train(INS ~ .,
                   data = train5,
                   method = "nnet", # Neural network using the nnet package
                   tuneGrid = tune_grid1.5,
                   trControl = trainControl(method = 'cv', # Using 10-fold cross-validation
                                            number = 10),
                   trace = FALSE)

nn.try.caret$bestTune


#Build best model
set.seed(12)
nn3 <- nnet(INS ~ .,
            data = train5, size = 3, decay = 0.5)

#PLot
plotnet(nn3)



#Get predictions
pred.test = (predict(nn3, type='raw'))
pred.test


#Plot the ROC Curve
plotROC(train5$INS, pred.test) 







# Hinton Diagram 1
nn_weights1 <- matrix(data = nn3$wts[1:78], ncol = 3, nrow = 26)
rownames(nn_weights1) <- c("bias", nn3$coefnames[1:25])
colnames(nn_weights1) <- c("h1", "h2", "h3")

ggplot(melt(nn_weights1), aes(x=Var1, y=Var2, size=abs(value), color=as.factor(sign(value)))) +
  geom_point(shape = 15) +
  scale_size_area(max_size = 40) +
  labs(x = "", y = "", title = "Hinton Diagram of NN Weights") +
  theme_bw()

# Hinton Diagram 2
nn_weights2 <- matrix(data = nn3$wts[79:156], ncol = 3, nrow = 26)
rownames(nn_weights2) <- c(nn3$coefnames[26:51])
colnames(nn_weights2) <- c("h1", "h2", "h3")

ggplot(melt(nn_weights2), aes(x=Var1, y=Var2, size=abs(value), color=as.factor(sign(value)))) +
  geom_point(shape = 15) +
  scale_size_area(max_size = 40) +
  labs(x = "", y = "", title = "Hinton Diagram of NN Weights") +
  theme_bw()

# Hinton Diagram 3
nn_weights3 <- matrix(data = nn3$wts[157:234], ncol = 3, nrow = 26)
rownames(nn_weights3) <- c(nn3$coefnames[52:77])
colnames(nn_weights3) <- c("h1", "h2", "h3")

ggplot(melt(nn_weights3), aes(x=Var1, y=Var2, size=abs(value), color=as.factor(sign(value)))) +
  geom_point(shape = 15) +
  scale_size_area(max_size = 40) +
  labs(x = "", y = "", title = "Hinton Diagram of NN Weights") +
  theme_bw()



##
##
## Naive Bayes Model
##
##


# Naive Bayes model
set.seed(12)
nb1 = naiveBayes(INS ~ ., data = train4, laplace = 0, usekernel = TRUE)

# Optimize laplace and kernel - CARET ONLY ABLE TO TUNE CLASSIFICATION PROBLEMS FOR NAIVE BAYES
tune_grid2 <- expand.grid(
  usekernel = c(TRUE, FALSE),
  fL = seq(0, 1, 0.1),
  adjust = seq(0, 5, 1)
)

set.seed(12)
nb2.caret <- train(INS ~ ., data = train4,
                       method = "nb", 
                       tuneGrid = tune_grid2,
                       trControl = trainControl(method = 'cv', # Using 10-fold cross-validation
                                                number = 10))

nb2.caret$bestTune


# Naive Bayes model
set.seed(12)
nb2 = naiveBayes(INS ~ ., data = train3, laplace = 0, usekernel = FALSE, adjust = 0)



#Get predictions
pred2 = predict(nb2, train3, type='raw')[,2]
pred2



#Plot the ROC Curve
plotROC(train3$INS, pred2)








#Recreate previous XGBoost model as best and final model\
train6 = cbind(qual.var, quant.var)
train_x <- model.matrix(INS ~ ., data = train6)[, -1]
train_y <- as.numeric(train6$INS) - 1


set.seed(12)
xgb_final_train <- xgboost(data = train_x, label = train_y, subsample = 1, nrounds = 11, eta = 0.3, max_depth = 5, objective = "binary:logistic", eval_metric = "auc")



# Tuning final model through caret
tune_grid = expand.grid(
  nrounds = 11,
  eta = 0.3,
  max_depth = 5,
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = c(0.001, 1)
)

set.seed(12)
xgb_final = caret::train(x = train_x, y = factor(train_y),
                   method = "xgbTree",
                   tuneGrid = tune_grid,
                   trControl = trainControl(method = 'cv', # Using 10-fold cross-validation
                                            number = 10), objective = "binary:logistic", eval_metric = "auc")
xgb_final$bestTune

plot(xgb_final)  #max tree depth 5, eta .3, subsample 1



#variable importance
xgb.importance(feature_names = colnames(train_x), model = xgb_final_train)
xgb.ggplot.importance(xgb.importance(feature_names = colnames(train_x), model = xgb_final_train))

#auroc
p_hat2 <- predict(xgb_final_train, train_x ,type="prob")
plotROC(train6$INS, p_hat2) 
# train$p_hat2 <- NULL
#.8456



#Run XGBoost on Validation now
val_x = model.matrix(INS ~ ., data = val2)[, -1]
p_hat_v = predict(xgb_final_train, val_x ,type="prob")

xgb_final_train$feature_names %in% colnames(val_x)
xgb_final_train$feature_names

for (i in 1:ncol(train_x)){
  if (!(colnames(val_x)[i] %in% colnames(train_x))){
    print(colnames(val_x)[i])
  }
}
  
colnames(train_x)

#auroc
plotROC(val2$INS, p_hat_v) 
# train$p_hat2 <- NULL
#.8456









#Model interpretation for final XGBoost

#PDP
set.seed(12)
xgboost_pred = Predictor$new(xgb_final, data = data.frame(train_x), 
                             y = factor(train_y), type = "prob")


pd_plot <- FeatureEffects$new(xgboost_pred, method = "pdp")
pd_plot$plot(c("ACCTAGE"))
pd_plot$plot()

pd_plot


# LIME for customer 732
point = 732
set.seed(12)
lime.explain = LocalModel$new(xgboost_pred, x.interest = data.frame(t(train_x[point,])), k = 5)
plot(lime.explain) + theme_bw()






#Shapley Values for customer 732
point = 732
set.seed(12)
shap = Shapley$new(xgboost_pred, x.interest = data.frame(t(train_x[point,])))
shap$plot()

class(data.frame(train_x))
as.data.frame(train_x[point,])
