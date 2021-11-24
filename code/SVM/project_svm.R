rm(list = ls(all=TRUE))
#load libraries

#install.packages("splitstackshape")
library(splitstackshape)
library(raster)
library(sp)
#setwd("AIA_project")

#training dataset
tile_train_1=as.data.frame(stack("tile_train_1.tif"))
tile_train_2=as.data.frame(stack("tile_train_2.tif"))
tile_train_3=as.data.frame(stack("tile_train_3.tif"))
gt_train_1=as.data.frame(stack("gt_train_1.tif"))
gt_train_2=as.data.frame(stack("gt_train_2.tif"))
gt_train_3=as.data.frame(stack("gt_train_3.tif"))

names(tile_train_1)=c("x1","x2","x3","x4","x5","x6","x7","x8")
names(tile_train_2)=c("x1","x2","x3","x4","x5","x6","x7","x8")
names(tile_train_3)=c("x1","x2","x3","x4","x5","x6","x7","x8")

names(gt_train_1)=c("y")
names(gt_train_2)=c("y")
names(gt_train_3)=c("y")

tile_train=rbind(tile_train_1,tile_train_2,tile_train_3)
gt_train=rbind(gt_train_1,gt_train_2,gt_train_3)

training.data.fr=cbind(tile_train,gt_train)             #join x and y
training.data.fr=na.omit(training.data.fr)              #remove NA values
set.seed(1)
training.data.fr <- stratified(training.data.fr, "y", 2000) #stratified sampling
write.csv(training.data.fr, "training.csv")

rm(tile_train_1, tile_train_2, tile_train_3, gt_train_1, gt_train_2, gt_train_3)


#validation
tile_valid_1=stack("tile_valid_1.tif")
gt_valid_1=stack("gt_valid_1.tif")

validation.data.fr=data.frame(getValues(tile_valid_1), as.factor(getValues(gt_valid_1)))
colnames(validation.data.fr)=c("x1","x2","x3","x4","x5","x6","x7","x8","y")
validation.data.fr=na.omit(validation.data.fr)
set.seed(1)
validation.data.fr <- stratified(validation.data.fr, "y", 1000)
#write.csv(testing.data.fr,"validation.csv")

library(e1071)


#kernel.vec=c("linear","radial","polynomial")
C.vec <- 10^seq(0,3,len=20)
C.vec.poly<-10^seq(0,2,len=15)
gamma.vec <- 10^seq(-3,1,len=10)
degree.vec<-c(2,3)



##############GRIDSEARCH FUNCTIONS with different kernels#############
#The output is overall accuracies with different combinations = needed to build plots
#one function with combination of different parameters including kernel in another file, output - best parameters
#also could be done as one f-n but we needed OA separately for plotting

#Linear
Lingridsearch <- function(model, training, validation, Crange) {
  Clength <- length(Crange)
  
  # store overall accuracies in 2D array
  OA.vec <- array(0, dim=c(Clength))
  
  # extract left hand side (dependent variable) of formula
  #dv <- all.vars(update(model, .~0))
  
  # loop over the search ranges
  for (C in Crange){
      SVM.lin.model <- svm(formula=model, data=training, type="C-classification", kernel = "linear", cost = C,  scale = FALSE)
      ypred <- predict(SVM.lin.model, validation)
      CM <- table(prediction=ypred,truth=validation$y)
      OA <- sum(diag(CM))/length(validation$y)
      OA.vec[match(C, Crange)] <- OA
      cat("C = ",C, ";\t OA = ", OA, "\n") 
  }
  return(OA.vec)
}
#RBF kernel
RBFgridsearch <- function(model, training, validation, Crange, gammarange) {
  Clength <- length(Crange)
  gammalength <- length(gammarange)
  
  # store overall accuracies in 2D array
  OA.vec <- array(0, dim=c(Clength, gammalength))
  
  # extract left hand side (dependent variable) of formula
  #dv <- all.vars(update(model, .~0))
  
  # loop over the search ranges
  for (C in Crange){
    for (gamma in gammarange) {
      SVM.RBF.model <- svm(formula=model, data=training, type="C-classification", kernel = "radial", cost = C, gamma = gamma, scale = FALSE)
      ypred <- predict(SVM.RBF.model, validation)
      CM <- table(prediction=ypred,truth=validation$y)
      OA <- sum(diag(CM))/length(validation$y)
      OA.vec[match(C, Crange), match(gamma, gammarange)] <- OA
      cat("C = ",C,";\t gamma = ",gamma, ";\t OA = ", OA, "\n") 
    }
  }
  
  # search for best parameters
  return(OA.vec)
}

#Polynomial kernel
Polygridsearch <- function(model, training, validation, Crange, gammarange,degreerange) {
  Clength <- length(Crange)
  gammalength <- length(gammarange)
  degreelength <- length(degreerange)
  
  # store overall accuracies in 2D array
  OA.vec <- array(0, dim=c(Clength, gammalength, degreelength))
  
  # extract left hand side (dependent variable) of formula
  #dv <- all.vars(update(model, .~0))
  
  # loop over the search ranges
  for (C in Crange){
    for (gamma in gammarange) {
      for (degree in degreerange){
      SVM.polynomial.model <- svm(formula=model, data=training, type="C-classification", kernel = "polynomial", cost = C, gamma = gamma,degree=degree, scale = FALSE)
      ypred <- predict(SVM.polynomial.model, validation)
      CM <- table(prediction=ypred,truth=validation$y)
      OA <- sum(diag(CM))/length(validation$y)
      OA.vec[match(C, Crange), match(gamma, gammarange), match(degree,degreerange)] <- OA
      cat("C = ",C,";\t gamma = ",gamma,";\t degree=", degree, ";\t OA = ", OA, "\n") 
    }
    }
  }
  return(OA.vec)
}  
#Training models with different combinations

OA.linear.vec=Lingridsearch(model=y ~ ., training=training.data.fr, validation=validation.data.fr,
                            Crange=C.vec)
OA.RBF.vec=RBFgridsearch(model=y ~ ., training=training.data.fr, validation=validation.data.fr,
                         Crange=C.vec, gamma=gamma.vec)

training.data.fr.poly <- stratified(training.data.fr, "y", 500)
OA.poly.vec=Polygridsearch(model=y ~ ., training=training.data.fr.poly, validation=validation.data.fr,
                         Crange=C.vec.poly, gamma=gamma.vec, degree=degree.vec)

# search for best parameters.
#Linear kernel
lin.best.OA.index <- which(OA.linear.vec == max(OA.linear.vec), arr.ind=TRUE)
lib.best.C<-C.vec[lin.best.OA.index[1]]
lin.best.OA<-max(OA.linear.vec)

#RBF kernel
RBF.best.OA.index <- which(OA.RBF.vec == max(OA.RBF.vec), arr.ind=TRUE)
RVF.best.C <- C.vec[RBF.best.OA.index[1]]
RBF.best.gamma <- gamma.vec[RBF.best.OA.index[2]]
RBF.best.OA<-max(OA.RBF.vec)

#polynomial kernel
poly.best.OA.index <- which(OA.poly.vec == max(OA.poly.vec), arr.ind=TRUE)
poly.best.C <- C.vec[poly.best.OA.index[1]]
poly.best.gamma <- gamma.vec[poly.best.OA.index[2]]
poly.best.degree<-degree.vec[poly.best.OA.index[3]]
poly.best.OA<-max(OA.poly.vec)

install.packages("plotly")
library(plotly)
fig=plot_ly(x=C.vec,y=gamma.vec,z=100*OA.RBF.vec)%>%add_surface()%>%
  layout(title=" 
         RBF kernel: C, gamma, OA", scene = list(xaxis = list(title = " C "), yaxis = list(title = "gamma"), zaxis = list(title = "OA")))

windows()
fig

library(ggplot2)

RBF_fig_C_OA=ggplot(x=C.vec[OA.RBF.vec],y=100*OA.RBF.vec)
window()
RBF_fig_C_OA
#######testing 2019####################

install.packages("tidyverse")
library(tidyverse)

tile_test_1=stack("tile_test_1.tif")
gt_test_1=raster("gt_test_1.tif")

testing.data.fr=data.frame(getValues(tile_test_1), as.factor(getValues(gt_test_1)))
colnames(testing.data.fr)=c("x1","x2","x3","x4","x5","x6","x7","x8","y")

#replace NA with mean
for(i in 1:ncol(testing.data.fr)){
  testing.data.fr[is.na(testing.data.fr[,i]), i] <- mean(testing.data.fr[,i], na.rm = TRUE)
}
#write.csv(testing.data.fr,"validation.csv")


SVM.model <- svm(y ~ . , data=training.data.fr, type="C-classification", kernel = "radial",gamma=RBF.best.gamma, cost = RVF.best.C, scale = FALSE)
ypred2019=predict(SVM.model, testing.data.fr)
CM=table(prediction=ypred2019,truth=testing.data.fr$y)
OA=sum(diag(CM))/length(testing.data.fr$y)

summary(SVM.lin.model)
#save.image(file="session1108.RData")

#load("session1108.RData")

#######testing 2020################# didn't work # exceeding memory 

tile_test_2020=as.data.frame(stack("big_data_2020.tif"))
gt_test_2020=stack("cropland_align_2020_30m.tif")

testing2020.data.fr=data.frame(getValues(tile_test_2020), as.factor(getValues(gt_test_2020)))
colnames(testing2020.data.fr)=c("x1","x2","x3","x4","x5","x6","x7","x8","y")
for(i in 1:ncol(testing2020.data.fr)){
  testing2020.data.fr[is.na(testing2020.data.fr[,i]), i] <- mean(testing2020.data.fr[,i], na.rm = TRUE)
}


ypred2020=predict(SVM.lin.model, testing2020.data.fr)
CM2020=table(prediction=ypred202020,truth=testing2020.data.fr$y)
OA2020=sum(diag(CM2020))/length(testing2020.data.fr$y)

summary(SVM.lin.model)

datafr2019=data.frame(ypred2019)
#write.csv(testing.data.fr,"validation.csv")
write.csv(as.integer(datafr2019$ypred2019),"ypred2019new.csv")
read.csv("ypred2019.csv")
