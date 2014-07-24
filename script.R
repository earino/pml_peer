library(caret)
library(dplyr)
library(doMC)
library(parallel)
library(randomForest)

registerDoMC(cores = 8)
set.seed(1)

pml.training <- read.csv("pml-training.csv", stringsAsFactors=FALSE)
pml.testing <- read.csv("pml-testing.csv", stringsAsFactors=FALSE)

pml.training$classe <- factor(pml.training$classe)

inTraining <- createDataPartition(pml.training$classe, p=.9, list=FALSE)
training.superset <- pml.training[inTraining,]
validation <- pml.training[-inTraining,]

inTesting <- createDataPartition(training.superset$classe, p=.25, list=FALSE)
training <- training.superset[-inTesting,]
testing <- training.superset[inTesting,]

axis <- grep("_[xyz]$", names(training))
y <- training$classe
x <- training[, axis]

fit.rf <- train(x, y, method="rf", preProcess=c("medianImpute"), trControl=trainControl(method="cv"))
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml.testing[is.na(pml.testing)] <- 0
pml_write_files(predict(fit.rf,pml.testing))
