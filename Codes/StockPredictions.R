# Set the working Directory
setwd("F:/12_Capstone/Stocks")
library(psych)
library(car)

# Import the Data Set
stockdata=read.csv("APPLE.csv", header=TRUE)


# Split the Data set into Training and Test in 70:30 proportion
# First run the model on Traininf Data and then validate it with Test data.
library(caret)
set.seed(123)
index <- createDataPartition(stockdata$close, p=0.70, list=FALSE)
trainingdata <- stockdata[index,]
testdata <- stockdata[-index,]

#trainingdata <- stockdata
# List the Dimensions
dim(trainingdata)
dim(testdata)

## Summary Statistics
attach(trainingdata)
describe(trainingdata$curr_ratio)
describe(trainingdata$tot_debt_tot_equity)
describe(trainingdata$oper_profit_margin)
describe(trainingdata$asset_turn)
describe(trainingdata$ret_equity)
describe(trainingdata$sentiment)

with(trainingdata, boxplot(curr_ratio, main="curr_ratio"))
with(trainingdata, boxplot(tot_debt_tot_equity, main="tot_debt_tot_equity"))
with(trainingdata, boxplot(oper_profit_margin, main="oper_profit_margin"))

with(trainingdata, boxplot(asset_turn, main="asset_turn"))
with(trainingdata, boxplot(ret_equity, main="ret_equity"))
with(trainingdata, boxplot(sentiment, main="sentiment"))

### Predict and check on the Training Data
## Multiple Regression
pricereg <- lm(close ~ curr_ratio+tot_debt_tot_equity+oper_profit_margin+asset_turn+ret_equity+sentiment, data=trainingdata)
#pricereg <- lm(close ~ curr_ratio+tot_debt_tot_equity+oper_profit_margin+ret_equity, data=trainingdata)
pricereg
summary(pricereg)
anova(pricereg)
AIC (pricereg)
BIC (pricereg)

pricefit <- fitted(pricereg)
priceres <- residuals(pricereg)

abline(pricereg, col='red')

## Prediction for test data
pricepredicted = predict.lm(pricereg, testdata)
actuals_preds <- data.frame(cbind(actuals=testdata$close, predicteds=pricepredicted))  # make actuals_predicteds dataframe.

correlation_accuracy <- cor(actuals_preds)
correlation_accuracy
head(actuals_preds)
actuals_preds

min_max_accuracy = mean(apply(actuals_preds, 1, min) / apply(actuals_preds, 1, max))
min_max_accuracy
mape = mean(abs((actuals_preds$predicteds - actuals_preds$actuals))/actuals_preds$actuals)
mape

#library(DAAG)
#cvResults <- suppressWarnings(CVlm(df=stockdata, form.lm=close ~ curr_ratio, m=5, dots=FALSE, seed=29, legend.pos="topleft",  printit=FALSE, main="Small symbols are predicted values while bigger ones are actuals."));  # performs the CV
#attr(cvResults, 'ms') 


## Prediction of new observations
#newobs <- data.frame(curr_ratio = 1.3906, tot_debt_tot_equity = 0.7348, oper_profit_margin = 26.6504, sentiment = 3)
#predict.lm(pricereg, newdata=newobs)

##################################################################################################################
