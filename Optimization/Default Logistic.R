setwd("D:\\GL - class data\\Financial risk analytics\\Assignment")
train = read.csv("train.csv",header = TRUE)
str(train)

reg = glm(default~ nw.ta + inc.na + cashp + pat.nw + 
            fa.ta + inv.nw +
            debt.equity + debtr.turn + 
            debt.ta
          ,family=binomial,data=train)
summary(reg)

## reg is the model finalized through iterations, this will be used for testing

test = read.csv("test.project.csv",header = TRUE)

prob = predict(reg,test,type="response")

validation = cbind(test,prob)
write.csv(validation, file="validation.proj.csv")
