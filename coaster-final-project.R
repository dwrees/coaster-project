library(kknn)
install.packages('kknn')
library(MASS)
library(kknn)
source("http://www.rob-mcculloch.org/2022_uml/webpage/R/docv.R")

###########################################
## Final Project
###########################################

###########################################
## kNN:
###########################################

coaster<-read.csv("C:/Users/User/Desktop/coasters-2015.csv")

ddftr = data.frame(speed = coaster$Speed, height = coaster$Height)
ddfte = data.frame(speed = sort(coaster$Speed))

kf5 = kknn(height~speed,ddftr,ddfte,k=5,kernel = "rectangular")
names(kf5)

plot(ddftr$speed,ddftr$height,xlab='height',ylab='speed',cex=.5,col='blue')
lines(ddfte$speed,kf50$fitted.values,col="red",lwd=2)
title(main='knn fit with 5 neighbors',cex.main=1.5)

set.seed(99) 
kv = 2:20
cvres = docvknn(matrix(ddftr$speed,ncol=1),ddftr$height,kv,nfold=10)

rmseres = sqrt(cvres/nrow(ddftr))
plot(kv,rmseres,xlab='k = num neighbors',ylab='rmse',type='b',col='blue')
title(main=paste0("best k is ",kv[which.min(cvres)]))
print(rmseres)

# RMSE at k=5 is 28.478487

#######################################
## Single Tree:
#######################################

cd<-read.csv("C:/Users/User/Desktop/coasters-2015.csv")
y = cd$Height
x = cd$Speed

ddf = data.frame(x,y)
n = nrow(ddf)

# train/test split
set.seed(34)

ntr = floor(.75*n)
nte = n - ntr
iitr = sample(1:n,ntr)

library(rpart)

set.seed(99)
big.tree = rpart(y~.,method="anova",data=ddf[iitr,],control=rpart.control(minsplit=2,cp=.0001))

nbig = length(unique(big.tree$where))
cat("size of big tree: ",nbig,"\n")

## CV results:
plotcp(big.tree)

iibest = which.min(big.tree$cptable[,"xerror"])
bestcp=big.tree$cptable[iibest,"CP"]
bestsize = big.tree$cptable[iibest,"nsplit"]+1

#prune to good tree
best.tree = prune(big.tree,cp=bestcp)
nbest = length(unique(best.tree$where))
cat("size of best tree: ", nbest,"\n")

## plot tree
plot(best.tree,uniform=TRUE)
text(best.tree,digits=4,use.n=TRUE)

## In sample fits
yhat = predict(best.tree)
plot(x[iitr],y[iitr],xlab='height',ylab='speed')
points(x[iitr],yhat,col='blue')

## prediction
ypred = predict(best.tree,ddf[-iitr,])
plot(x[-iitr],y[-iitr],xlab='height',ylab='speed')
points(x[-iitr],ypred,col='blue')
rmsetree = sqrt(mean((y[-iitr] - ypred)^2))
print(rmsetree)

########################################
## Boosting
########################################

library(MASS)
library(rpart)

df = coaster
df$Height = df$Height-mean(df$Height)
n = length(df$Speed)

### maintain function on a grid of values
NUM_GRID = 100 
range.x = range(df$Speed)
grid.x = seq(range.x[1], range.x[2], length.out = NUM_GRID)

#give the fv such the new.x is closest to x
myPredict = function(x, fv, new.x) {
  fv[which.min(sapply(x, function(z) sqrt(sum((z - new.x) ^ 2))))]
}

## Paramaters
MAX_DEPTH = 1 #fit stumps
lambda = 0.1

fhat=rep(0,NUM_GRID)
residuals=df$Height

## boosting loop
firsttime=TRUE
#firsttime=FALSE
if(firsttime) {
  sleeptime = 1.0 
  Bcut=10 
  MAX_B = 30 #number of boosting iterations 30
} else {
  sleeptime = .5 
  Bcut=0 
  MAX_B = 30 #number of boosting iterations 30
}
par( mfrow = c( 2, 2 ) )
fmat = matrix(0.0,NUM_GRID,MAX_B)
fmat[,1]=fhat

for (B in seq(1, MAX_B-1)) {
  if (B %% 1 == 0) print(paste("Iteration ==> ", B, sep=""))
  
  # plot points and current fit  
  plot(df$Speed, df$Height)
  points(grid.x, fhat, cex=1, col="red", type="l",lwd=2)
  title(main="data and current f")
  
  # plot the current fit 
  plot(grid.x, fhat, cex=1, col="red", ylim=range(df$Height),type="l",lwd=2)
  title(main="current f")
  if(B>1) points(grid.x,fmat[,B-1],col="grey")
  
  # plot residuals, with fit and crushed fit
  plot(df$Speed, residuals, col="black")
  title(main="fit and crushed fit to resids")
  
  # fit residuals
  tmp.tr = rpart(res~speed, data.frame(res=residuals, speed=df$Speed), 
                 control=rpart.control(maxdepth = MAX_DEPTH))
  # plot fit and crushed fit
  if(B>Bcut) {
    tempfit = predict(tmp.tr,data.frame(speed=grid.x))
    points(grid.x,tempfit,col="red",pch=16)
    points(grid.x,lambda*tempfit,col="green",pch=16)
  } else {
    for (cx in grid.x) {
      py = predict(tmp.tr, data.frame(speed=cx))
      points(cx, py, col="red",pch=16)
      points(cx, lambda * py, col="green",pch=16)
    }
  }
  
  # update fit
  for (i in 1:NUM_GRID) {
    py = predict(tmp.tr, data.frame(speed=grid.x[i]))
    fhat[i] = fhat[i] + lambda * py
  }
  fmat[,B+1]=fhat
  
  # update residuals
  for (i in 1:n) {
    residuals[i] = df$Height[i] - myPredict(grid.x, fhat, df$Speed[i])
  }
  
  # plot the new current fit 
  if(B<Bcut) {
    readline("update f?")
  } else {
    Sys.sleep(sleeptime)
  }
  plot(grid.x, fhat, cex=1, col="red", ylim=range(df$Height),type="l",lwd=2)
  title(main=paste("new current f, interation: ",B))
  
  
  if(B<Bcut) {
    readline("GO?")
  } else {
    Sys.sleep(sleeptime)
  }
}

##plot function iterations
if(0) {
  par(mfrow=c(1,1))
  sleeptime = .3
  for(i in 1:ncol(fmat)) {
    plot(df$Speed,df$Height,pch=1,col="blue")
    lines(grid.x,fmat[,i],col="red",lwd=2)
    Sys.sleep(sleeptime)
  }
}


## Iteration 29: 

### boosting
library(gbm)

#fit using boosting
boostfit = gbm(y~.,data=ddf[iitr,],distribution="gaussian",
               interaction.depth=4,n.trees=30,shrinkage=.2)
yhatbst = predict(boostfit,n.trees=30)
plot(x[iitr],y[iitr],xlab='speed',ylab='height')
points(x[iitr],yhatbst,col='blue')

# Test values (2-30) for n.trees:

boostfit1 = gbm(y~.,data=ddf[iitr,],distribution="gaussian",
                interaction.depth=4,n.trees=30,shrinkage=.2)
yhatbst1 = predict(boostfit1,n.trees=30)
plot(x[iitr],y[iitr],xlab='mileage',ylab='price')
points(x[iitr],yhatbst1,col='blue')

ypredbst1 = predict(boostfit,newdata=ddf[-iitr,],n.trees=30)
rmsebst = sqrt(mean((y[-iitr] - ypredbst1)^2))
print(rmsebst)

