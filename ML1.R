#########################################
#   Machine Learning                    #
#   07Nov2021                           #
#   Hurricane Readiness                 #
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

#Read in data
setwd('C:/Users/Richard Pincus/Documents/Classes - MSA/AA502/Machine Learning/hw/hw1')
train = read.csv('insurance_t.csv')

train0 = train

#Set alpha level
a_samp = 1-pchisq(log(nrow(train)),1)


#Check for missing values
for (i in 1:ncol(train)){
  if (sum(is.na(train[,i])) > 0){
    print(colnames(train)[i])
  }
}



#
#Impute missings for ACCTAGE
#

#check out ACCTAGE
train$ACCTAGE

#Use median to impute
train$ACCTAGE.i = ifelse(is.na(train$ACCTAGE), 
                         median(train$ACCTAGE, na.rm=T), 
                         train$ACCTAGE)
train$ACCTAGE.i.fl = ifelse(is.na(train$ACCTAGE), 
                         1, 
                         0)




#
#Impute missings for PHONE
#

#check out PHONE
train$PHONE

#Use median to impute
train$PHONE.i = ifelse(is.na(train$PHONE), 
                         median(train$PHONE, na.rm=T), 
                         train$PHONE)
train$PHONE.i.fl = ifelse(is.na(train$PHONE), 
                            1, 
                            0)




#
#Impute missings for POS
#

#check out POS
train$POS

#Use median to impute
train$POS.i = ifelse(is.na(train$POS), 
                         median(train$POS, na.rm=T), 
                         train$POS)
train$POS.i.fl = ifelse(is.na(train$POS), 
                            1, 
                            0)





#
#Impute missings for POSAMT
#

#check out POSAMT
train$POSAMT

#Use median to impute
train$POSAMT.i = ifelse(is.na(train$POSAMT), 
                         median(train$POSAMT, na.rm=T), 
                         train$POSAMT)
train$POSAMT.i.fl = ifelse(is.na(train$POSAMT), 
                            1, 
                            0)



#
#Impute missings for INV
#

#check out INV
train$INV

#Use median to impute
train$INV.i = ifelse(is.na(train$INV), 
                         median(train$INV, na.rm=T), 
                         train$INV)
train$INV.i.fl = ifelse(is.na(train$INV), 
                            1, 
                            0)



#
#Impute missings for INVBAL
#

#check out INVBA:
train$INVBAL

#Use median to impute
train$INVBALi = ifelse(is.na(train$INVBAL), 
                         median(train$INVBAL, na.rm=T), 
                         train$INVBAL)
train$INVBAL.i.fl = ifelse(is.na(train$INVBAL), 
                            1, 
                            0)



#
#Impute missings for CC
#

#check out CC
train$CC

#Use median to impute
train$CC.i = ifelse(is.na(train$CC), 
                         median(train$CC, na.rm=T), 
                         train$CC)
train$CC.i.fl = ifelse(is.na(train$CC), 
                            1, 
                            0)



#
#Impute missings for CCBAL
#

#check out CCBAL
train$CCBAL

#Use median to impute
train$CCBAL.i = ifelse(is.na(train$CCBAL), 
                         median(train$CCBAL, na.rm=T), 
                         train$CCBAL)
train$CCBAL.i.fl = ifelse(is.na(train$CCBAL), 
                            1, 
                            0)



#
#Impute missings for CCPURC
#

#check out CCPURC
train$CCPURC

#Use median to impute
train$CCPURC.i = ifelse(is.na(train$CCPURC), 
                         median(train$CCPURC, na.rm=T), 
                         train$CCPURC)
train$CCPURC.i.fl = ifelse(is.na(train$CCPURC), 
                            1, 
                            0)



#
#Impute missings for INCOME
#

#check out INCOME
train$INCOME

#Use median to impute
train$INCOME.i = ifelse(is.na(train$INCOME), 
                         median(train$INCOME, na.rm=T), 
                         train$INCOME)
train$INCOME.i.fl = ifelse(is.na(train$INCOME), 
                            1, 
                            0)



#
#Impute missings for LORES
#

#check out LORES
train$LORES

#Use median to impute
train$LORES.i = ifelse(is.na(train$LORES), 
                         median(train$LORES, na.rm=T), 
                         train$LORES)
train$LORES.i.fl = ifelse(is.na(train$LORES), 
                            1, 
                            0)



#
#Impute missings for HMVAL
#

#check out HMVAL
train$HMVAL

#Use median to impute
train$HMVAL.i = ifelse(is.na(train$HMVAL), 
                         median(train$HMVAL, na.rm=T), 
                         train$HMVAL)
train$HMVAL.i.fl = ifelse(is.na(train$HMVAL), 
                            1, 
                            0)



#
#Impute missings for AGE
#

#check out AGE
train$AGE

#Use median to impute
train$AGE.i = ifelse(is.na(train$AGE), 
                         median(train$AGE, na.rm=T), 
                         train$AGE)
train$AGE.i.fl = ifelse(is.na(train$AGE), 
                            1, 
                            0)



#
#Impute missings for CRSCORE
#

#check out CRSCORE
train$CRSCORE

#Use median to impute
train$CRSCORE.i = ifelse(is.na(train$CRSCORE), 
                         median(train$CRSCORE, na.rm=T), 
                         train$CRSCORE)
train$CRSCORE.i.fl = ifelse(is.na(train$CRSCORE), 
                            1, 
                            0)


#Remove vars with missings
train1 = train %>% dplyr::select(-ACCTAGE, -PHONE, -POS, -POSAMT, -INV, -INVBAL, -CC, -CCBAL, -CCPURC, -INCOME, -LORES, -HMVAL, -AGE, -CRSCORE)
# colnames(train1)




#Create categorical variables of any vars with less then 10 unique values
train2 = train1
for (i in 1:ncol(train1)){
  if (length(unique(train1[,i])) < 10){
    train2[,i] = factor(train1[,i])
  }
  else if (colnames(train1)[i] == 'BRANCH') {
    train2[,i] = factor(train1[,i])
  }
  else {
    train2[,i] = train1[,i]
  }
}

str(train2)



#Check out correlations with the target INS
# correlation_table(data=train1, target="INS")
# 
# plot(train1$CRSCORE, train1$INS)



#Cluster variables
factor.vars = NA
for (i in 1:ncol(train2)){
  factor.vars[i] = is.factor(train2[,i])
}

#Get list of var names that are quantitative and qualitative
qual.var.list = colnames(train2)[factor.vars]
quant.var.list = colnames(train2)[!(colnames(train2) %in% quant.var.list)]

#Get quant and qual vars
quant.var = train2 %>% dplyr::select(all_of(quant.var.list))
qual.var = train2 %>% dplyr::select(all_of(qual.var.list))

#Cluster variables to reduce dimensions
var.clust.h = hclustvar(quant.var, qual.var)
stab=stability(var.clust.h,B=50) ## This will take time!
plot(stab)
sort(stab$meanCR) #7 clusters is the winner
h7 = cutreevar(var.clust.h, 7)
h7$var

##
##Now build a MARS model to predict purchase of INS
##

#Remove vars with missings
train1 = train %>% dplyr::select(-ACCTAGE, -PHONE, -POS, -POSAMT, -INV, -INVBAL, -CC, -CCBAL, -CCPURC, -INCOME, -LORES, -HMVAL, -AGE, -CRSCORE)
# colnames(train1)


#First reduce number of variables using stepwise selection
# Cross-validation in stepwise regression - don't forget to set your seed!
set.seed(12)
step.model <- train(factor(INS) ~ ., data = train1,
                    method = "glmStepAIC", # Uses glmStepAIC for logistic steps
                    family = 'binomial'(link='logit'),
                    direction = 'backward',
                    # tuneGrid = data.frame(nvmax = 1:37), # Maximum number of variables to have in model - our tuning parameter
                    trControl = trainControl(method = 'cv', # Using 10-fold cross-validation
                                             number = 10))

summary(step.model)

# .outcome ~ DDA + DDABAL + DEP + CHECKS + TELLER + SAV + SAVBAL + 
#   ATM + ATMAMT + CD + CDBAL + IRA + IRABAL + MM + MMBAL + BRANCHB14 + 
#   BRANCHB15 + BRANCHB16 + BRANCHB2 + ACCTAGE.i + PHONE.i + 
#   POS.i + POSAMT.i + INV.i + CC.i + INCOME.i.fl + AGE.i.fl
formula(step.model)


#Build MARS model
set.seed(12)
mars1 <- earth(factor(INS) ~ factor(DDA) + DDABAL + factor(DEP) + factor(CHECKS) + factor(TELLER) + factor(SAV) + SAVBAL + 
                 factor(ATM) + ATMAMT + factor(CD) + CDBAL + factor(IRA) + IRABAL + factor(MM) + MMBAL + factor(BRANCH) + 
                 ACCTAGE.i + factor(PHONE.i) + factor(POS.i) + POSAMT.i + factor(INV.i) + factor(CC.i) + factor(INCOME.i.fl) + factor(AGE.i.fl),
               data = train1, glm=list(family=binomial), trace=0.5, nfold=10, pmethod='cv')
summary(mars1)



#Get variable importance
evimp(mars1)



#Get predictions
train1$p_hat_mars = predict(mars1, type = "response")


#Create ROC curve and get AUROC


#Plot the ROC Curve
plotROC(train1$INS, train1$p_hat_mars)


#Calculate concordance of predicted probabilities
Concordance(train1$INS, train1$p_hat_mars)


#Calculate KS Statistic
pred <- prediction(fitted(mars1), factor(train1$INS))
perf <- performance(pred, measure = "tpr", x.measure = "fpr")
KS <- max(perf@y.values[[1]] - perf@x.values[[1]])
cutoffAtKS <- unlist(perf@alpha.values)[which.max(perf@y.values[[1]] - perf@x.values[[1]])]
print(c(KS, cutoffAtKS))


#Confusion matrix
confusionMatrix(train1$INS, train1$p_hat_mars, threshold = cutoffAtKS)






##
##Now build a GAM model to predict purchase of INS
##

formula(step.model)
# GAM's using splines
set.seed(12)
gam1 <- mgcv::gam(factor(INS) ~ factor(DDA) + s(DDABAL) + factor(DEP) + factor(CHECKS) + factor(TELLER) + factor(SAV) + s(SAVBAL) + 
                    factor(ATM) + s(ATMAMT) + factor(CD) + s(CDBAL) + factor(IRA) + s(IRABAL) + factor(MM) + s(MMBAL) + factor(BRANCH) + 
                    s(ACCTAGE.i) + factor(PHONE.i) + factor(POS.i) + s(POSAMT.i) + factor(INV.i) + factor(CC.i) + factor(INCOME.i.fl) + factor(AGE.i.fl),
                  data = train, family='binomial'(link='logit'))
summary(gam1)



#Get predictions
train1$p_hat_gam <- predict(gam1, type = "response")


#Create ROC curve and get AUROC


#Plot the ROC Curve
plotROC(train1$INS, train1$p_hat_gam)




#Calculate concordance of predicted probabilities
Concordance(train1$INS, train1$p_hat_gam)


#Calculate KS Statistic
pred <- prediction(fitted(gam1), factor(train1$INS))
perf <- performance(pred, measure = "tpr", x.measure = "fpr")
KS <- max(perf@y.values[[1]] - perf@x.values[[1]])
cutoffAtKS <- unlist(perf@alpha.values)[which.max(perf@y.values[[1]] - perf@x.values[[1]])]
print(c(KS, cutoffAtKS))


#Confusion matrix
confusionMatrix(train1$INS, train1$p_hat_gam, threshold = cutoffAtKS)
