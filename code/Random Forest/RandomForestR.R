library(raster)
library(randomForest)
library(splitstackshape)
library(ggplot2)
library(caret)
library(dHSIC)
library(KRLS)
library(varSel)

df.train.1 <- stack("tile_train_1.tif")
gt.train.1 <- raster("gt_train_1.tif")
df.train.2 <- stack("tile_train_2.tif")
gt.train.2 <- raster("gt_train_2.tif")
df.train.3 <- stack("tile_train_3.tif")
gt.train.3 <- raster("gt_train_3.tif")

df.train.1 <- as.data.frame(df.train.1)
gt.train.1 <- as.data.frame(gt.train.1)
df.train.2 <- as.data.frame(df.train.2)
gt.train.2 <- as.data.frame(gt.train.2)
df.train.3 <- as.data.frame(df.train.3)
gt.train.3 <- as.data.frame(gt.train.3)

names(df.train.2) <- names(df.train.1)
names(df.train.3) <- names(df.train.1)
names(gt.train.2) <- names(gt.train.1)
names(gt.train.3) <- names(gt.train.1)

df.train.f <- rbind(df.train.1, df.train.2, df.train.3)
gt.train.f <- rbind(gt.train.1, gt.train.2, gt.train.3)

r <- rownames(na.omit(df.train.f))

df.train <- df.train.f[r,]
gt.train <- gt.train.f[r,]

df <- cbind(df.train, gt.train)

df.strata <- stratified(df, c("gt.train"), 20000)

df.y <- df.strata$gt.train
df.y <- as.factor(df.y)
df.strata$gt.train <- NULL
df.x <- df.strata

f <- JMdist(df.y, df.x)

fdd <- gausskernel(as.matrix(df.y), sigma=1)

d <- dhsic(df.x, df.y)

featurePlot(df.x, as.factor(df.y))

#gt.train.1 <- as.factor(gt.train.1)

ntree = 1000    #number of trees to produce per iteration
mtry = 2       # number of variables used as input to split the variables
r_forest = randomForest(df.x, y=df.y ,ntree = ntree, keep.forest=TRUE, mtry=mtry,importance = TRUE) 

imp =importance(r_forest)  #for ALL classes individually
imp                        #display importance output in console
varImpPlot(r_forest)
varUsed(r_forest)
importance(r_forest)

mtry <- tuneRF(df.x,df.y, ntreeTry=ntree,
               step=1.2,improve=0.01, trace=TRUE, plot=TRUE, mtryStart = 2)
best.m <- mtry[mtry[, 2] == min(mtry[, 2]), 1]
print(mtry)
print(best.m)

tree_nr_evaluation = data.frame(
  Trees=rep(1:nrow(r_forest$err.rate), times=6), 
  Type=rep(c("OOB", "1", "2", "3", "4", "5"),
           each=nrow(r_forest$err.rate)),
  Error = c(r_forest$err.rate[, "OOB"],
            r_forest$err.rate[, "1"],
            r_forest$err.rate[, "2"],
            r_forest$err.rate[, "3"],
            r_forest$err.rate[, "4"],
            r_forest$err.rate[, "5"]))

ggplot(data=tree_nr_evaluation, aes(x=Trees, y=Error)) + geom_line(aes(color=Type))

valid.x <- stack("tile_valid_1.tif")
valid.y <- raster("gt_valid_1.tif")

names(valid.x) <- names(df.train.1)

valid.x <- as.data.frame(valid.x)
valid.y <- as.data.frame(valid.y)

r <- rownames(na.omit(valid.x))

vx <- valid.x[r,]
vy <- valid.y[r,]

vy = as.factor(vy)

predictions = predict(valid.x, r_forest, format=".tif", overwrite=TRUE, progress="text", type="response") 

confusionMatrix(as.factor(as.data.frame(predictions)), as.factor(valid.y))


