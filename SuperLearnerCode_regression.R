##example script used to fit the candidate learners and superlearner for continous outcome 
#minimizing MAE loss
##############################################################
rm(list=ls())

library(glmnet)
library(xtable)
library(ROCR)
library(SuperLearner)
library(kernlab)
library(xgboost)
library(nnet)
library(randomForest)
library(varhandle)
library(dplyr)
library(readxl)
library(parallel)
library(quantregForest)
library(ranger)
library(ggplot2)
library(tidyr)
library(qrnn)
library(caret)


(num_cores = RhpcBLASctl::get_num_cores())
num_cores


source('method_mae.R')
source('method_huber.R')
source('SL.quantreg.R')
source('summarySL.R')

##load data

load("Cohort_hpcData.RData")

lossname = 'MAE'

Y = test$avg_pdc
X = test[ ,cov_list_dummy[[2]]]
X[,"comor_combined"] = 1*X[,"comor_combined"] 


##Specify a library of algorithms to included in the ensemble
SL.glmnetNoStandardize = function(...){
  SL.glmnet(...,standardize=FALSE)
}

SL.qrnn1 = function(...){
  SL.qrnn( ..., n.hidden = 1)
}

SL.qrnn2 = function(...){
  SL.qrnn( ..., n.hidden = 2)
}

SL.qrnn3 = function(...){
  SL.qrnn( ..., n.hidden = 3)
}

##
#n_estimators : int, default=100, increase n_estimators
#
SL.xgbDepth6eta07 = function(...){
  SL.xgboost(..., shrinkage = 0.7, max_depth = 6,
             objective =  reg:pseudohubererror,
             eval.metric = 'mae')
}

SL.xgbDepth6eta03 = function(...){
  SL.xgboost(..., shrinkage = 0.3, max_depth = 6,
             objective =  reg:pseudohubererror,
             eval.metric = 'mae')
}


SL.xgbDepth15eta07 = function(...){
  SL.xgboost(..., shrinkage =  0.7, max_depth = 15,
             objective =  reg:pseudohubererror,
             eval.metric = 'mae')
}

SL.xgbDepth15eta03 = function(...){
  SL.xgboost(..., shrinkage =  0.3, max_depth = 15,
             objective =  reg:pseudohubererror,
             eval.metric = 'mae')
}

SL.ksvm_cross10 = function(...){
  SL.ksvm(..., cross = 10)
}

SL.xgboostHuber = function(...) SL.xgboost(...,
                                           objective = "reg:pseudohubererror",
                                           eval.metric = 'mae')


SL.ranger_QR_mtry3_ntrees500 = function(...) SL.ranger.qr(..., 
                    mtry = 3,
                    num.trees = 500)

SL.ranger_QR_mtry7_ntrees500 = function(...) SL.ranger.qr(...,
                                                       mtry = 7,
                                                       num.trees = 500)

SL.ranger_QR_mtry10_ntrees500 = function(...) SL.ranger.qr(..., 
                                                       mtry = 10,
                                                       num.trees = 500)


SL.ranger_QR_mtry3_ntrees700 = function(...) SL.ranger.qr(..., 
                                                       mtry = 3,
                                                       num.trees = 700)

SL.ranger_QR_mtry7_ntrees700 = function(...) SL.ranger.qr(..., 
                                                       mtry = 7,
                                                       num.trees = 700)

SL.ranger_QR_mtry10_ntrees700 = function(...) SL.ranger.qr(..., 
                                                         mtry = 10,
                                                        num.trees = 700)


cov_list_EHR = cov_list_dummy[[1]]



##EHR only -- screen based on covariates from EHR only 
#####
 EHR <- function(X,...){
   returnCols <- rep(FALSE, ncol(X))
   returnCols[names(X) %in% cov_list_EHR] <- TRUE
   return(returnCols)
 }

SL.library = list(c('SL.glm', 'All'),
                  c('SL.glm', 'EHR'),
               c('SL.xgbDepth6eta07', 'All'),
               c('SL.xgbDepth6eta07', 'EHR'),
               c('SL.xgbDepth6eta03', 'All'),
               c('SL.xgbDepth6eta03', 'EHR'),
               c('SL.xgbDepth15eta07', 'All'),
               c('SL.xgbDepth15eta07', 'EHR'),
               c('SL.xgbDepth15eta03', 'All'),
               c('SL.xgbDepth15eta03', 'EHR'),
               c('SL.ranger_QR_mtry3_ntrees500', 'All'),
               c('SL.ranger_QR_mtry3_ntrees500', 'EHR'),
               c('SL.ranger_QR_mtry7_ntrees500', 'All'),
               c('SL.ranger_QR_mtry7_ntrees500', 'EHR'),
               c('SL.ranger_QR_mtry10_ntrees500', 'All'),
               c('SL.ranger_QR_mtry10_ntrees500', 'EHR'),
               c('SL.ranger_QR_mtry3_ntrees700', 'All'),
               c('SL.ranger_QR_mtry3_ntrees700', 'EHR'),
               c('SL.ranger_QR_mtry7_ntrees700', 'All'),
               c('SL.ranger_QR_mtry7_ntrees700', 'EHR'),
               c('SL.ranger_QR_mtry10_ntrees700', 'All'),
               c('SL.ranger_QR_mtry10_ntrees700', 'EHR'),
               c("SL.qrnn1", 'All'),
               c("SL.qrnn1", 'EHR'),
               c("SL.qrnn2", 'All'),
               c("SL.qrnn2", 'EHR'),
               c("SL.qrnn3", 'All'),
               c("SL.qrnn3", 'EHR'),
                  c("SL.ksvm", 'All'),
                 c("SL.ksvm", 'EHR'),
                 c('SL.ksvm_cross10', 'All'),
               c('SL.ksvm_cross10', 'EHR') )


fit.data.SL_NNLS <- CV.SuperLearner(Y=Y,X=X,
                              SL.library= SL.library,
                              family=gaussian(),
                              method= "method.mae",
                              cvControl = list(V = 10), 
                              innerCvControl = list(list(V = 5)),
                              verbose=TRUE, parallel = 'multicore')


summary(fit.data.SL_NNLS)

