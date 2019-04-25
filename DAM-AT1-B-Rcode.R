
library(ISLR)
library(corrplot)
library(Amelia) # (visually representing the missing values)
library(dplyr)
library(ggplot2)
library(mosaic) # tally function
library(gmodels) # corsstables
library(ROCR)
library(DMwR)
library(randomForest)
library(pROC) # For ROC graph
library(car) # for vif
library(OptimalCutpoints) # youden Index ( optimal cutpoint in ROC curve)
library(caret) # confusion matrix
library(pdp) # partial dependency plots
library(parallel)
library(doParallel)
library(InformationValue)
library(vip)

#######################################
####  Repurchase Training Dataset  ####
#######################################

purchase<- read.csv("C:/Users/Public/Zarmina-Data science/DAM/assignment 1/AT1-B/repurchase_training.csv")

str(purchase)

missmap(purchase, main = "Missing values vs observed") # plots the missing values. In our data set there are o% missing and 100% observed values. No missing values.

str(purchase)
summary(purchase)

dim(purchase) # dimensions

######################################
####  Exploratory Data Analysis   ####
######################################


table(purchase$Target) # '0' no of people customers with one vehicle, '1' no of customers who bought more than one vehicle
table(purchase$gender) #  0.5277112(52.7%) of the values are NULL. male and female disctribution among the customers, showing the missing vlaue as well
table(purchase$car_model) # 18 different models
table(purchase$age_band) # 0.8556233(85.5%) of the values are null. number of customers with with different age groups


# 1: Bivariate relationship

#********************************#
## a. categorical vs categorical##
#********************************#

### i: Target Vs Age Vs Gender
xtabs(~ age_band+ Target, data= purchase) # 45-54 years of age has greator purchases (o= 4001 & 1= 57) and NULL are the gratest
xtabs(~ Target+ age_band, data= purchase)
xtabs(~ Target+ gender, data= purchase) # males has most purchases as compared to females
plot(xtabs(~ Target+ gender, data= purchase))

tally(~ age_band: gender+ Target, purchase) # 45-54 yrs males (0= 2138 & 1=39)

### ii: Target Vs car_model
xtabs(~car_model+Target, data=purchase) # model 2,5 3,1,4 have raltively greator purchases
plot(xtabs(~car_model+Target, data=purchase))

### Target Vs car_model vs Age
tally(~age_band: car_model+ Target, data=purchase)

### Target Vs car_model vs gender
tally(~gender: car_model+ Target, data=purchase)

tally(~Target+ gender: car_segment, data=purchase)

## crosstable
### age_band Vs Target
CrossTable(purchase$age_band, purchase$Target, chisq = TRUE, prop.t = F) # p =  7.975081e-38

### gender vs Target
CrossTable(purchase$gender, purchase$Target, chisq = TRUE, prop.t = F) # p =  1.481495e-43 

### car_model Vs Target
CrossTable(purchase$car_model, purchase$Target, chisq = TRUE, prop.t = F) # p =  6.162464e-42 

### car_segment vs Target
CrossTable(purchase$car_segment, purchase$Target, chisq = TRUE, prop.t = F) #  p =  7.368894e-15 

str(purchase)

##*******************************##
## b. Categorical Vs continous   ##
##******************************##

by(purchase$age_of_vehicle_years, purchase$Target, mean)
by(purchase$age_of_vehicle_years, purchase$Target, median)


str(purchase)
names(purchase)

##***************************##
## c. COntinous vs continous ##
##**************************##

num_vars<- unlist(lapply(purchase, is.numeric))
purchase_nums <- purchase[ ,num_vars]

purchase_corr<- cor(purchase_nums)
corrplot(purchase_corr, method = "number")
corrplot(purchase_corr, method = "circle")


############################
####   Model Building  ####
############################


## Logistic Regression Model
### Train and Test data split

trainset_size <- floor(0.70 * nrow(purchase)) 
trainset_size 

# First step is to set a random seed to ensurre we get the same result each time
# All random number generators use a seed 
set.seed(3521) 

# Get indices of observations to be assigned to training set...
# This is via randomly picking observations using the sample function

trainset_indices <- sample(seq_len(nrow(purchase)), size = trainset_size)
trainset_indices

# Assign observations to training and testing sets

trainset <- purchase[trainset_indices, ]
trainset
testset <- purchase[-trainset_indices, ]
testset

# Rowcounts to check
nrow(trainset)
nrow(testset)
nrow(purchase)

###########################
#### Variable selection ####
###########################

# In this section, we will select which variables we want to include in our model
# We'll do this by backwards selection - start with everything and remove one by one

# Let's start by throwing all the variables into the logistic regression
glm1 = glm(formula = Target ~ . ,
           data = trainset,
           family = "binomial")
summary(glm1) # AIC=82.051

glm2 = glm(formula = Target ~ .-ID ,
           data = trainset,
           family = "binomial")
summary(glm2) # AIC= 14338 

glm3 = glm(formula = Target ~ .-ID - age_band ,
           data = trainset,
           family = "binomial")
summary(glm3) # AIC= 14347


glm4 = glm(formula = Target ~ .-ID - age_band-gender ,
           data = trainset,
           family = "binomial")
summary(glm4) # AIC= 14422


glm5 = glm(formula = Target ~ car_model+ car_segment,
           data = trainset,
           family = "binomial")
summary(glm5) # AIC= 22482

glm6 = glm(formula = Target ~ car_model+ car_segment+ total_paid_services,
           data = trainset,
           family = "binomial")
summary(glm6) # AIC= 21688


glm7 = glm(formula = Target ~ .-ID - gender-age_band - total_paid_services,
           data = trainset,
           family = "binomial")
summary(glm7) # AIC= 14421
vif(glm7)
alias(glm7)


glm8 = glm(formula = Target ~ .-ID - gender-age_band - total_paid_services - car_model ,
           data = trainset,
           family = "binomial")
summary(glm8) # AIC= 14724
vif(glm8)

glm9 = glm(formula = Target ~ .-ID - gender-age_band - total_paid_services - car_model- car_segment ,
           data = trainset,
           family = "binomial")
summary(glm9) # AIC= 14730
vif(glm9)

glm10 = glm(formula = Target ~ .-ID - gender-age_band - total_paid_services- annualised_mileage - car_model - age_of_vehicle_years- car_segment,
            data = trainset,
            family = "binomial")
summary(glm10) # AIC= 14728 
vif(glm10)

###############################
####      Final model      ####
###############################

glm_final = glm(formula = Target ~ .-ID - gender-age_band - total_paid_services - car_segment -non_sched_serv_warr,
                data = trainset,
                family = "binomial")
summary(glm_final) # AIC= 14419
vif(glm_final)

#######################################
### Probabilities and predictions  ####
#######################################

# probability
trainset$pobability<- predict( glm_final, newdata = trainset, type = "response" )
testset$probability = predict(glm_final, newdata = testset, type = "response")
head(testset)

##### Assessing the predictive ability of the model####
fitted.results <- predict(glm_final,newdata=testset,type='response')
fitted.results <- ifelse(fitted.results > 0.5,1,0)
misClasificError <- mean(fitted.results != testset$Target)
print(paste('Accuracy',1-misClasificError))  # "Accuracy 0.978452870412669"



# assume that the optimum probability threshold is 0.5
# Create the class prediction 
testset$prediction = 0
testset[testset$probability >= 0.5, "prediction"] = 1

# Have a look at the data
#head(testset)

##############################
####   Confusion Matrix   ####
##############################


# Create a confusion matrix (along with other measures) using the table function
xtab<- table(predicted=testset$prediction,true=testset$Target)[2:1,2:1] # confusion matrix at threshold 0.5

caret::confusionMatrix(xtab)


#Precision = TP/(TP+FP)
precision <- xtab[1,1]/(xtab[1,1]+xtab[1,2])
precision # 0.8283133

#Recall = TP/(TP+FN)
recall <- xtab[1,1]/(xtab[1,1]+xtab[2,1])
recall # 0.257732

#F1
f1 <- 2*(precision*recall/(precision+recall))
f1 # 0.393138

#############################
####      ROC &  AUC     ####
#############################

Pred_purchase <- prediction(testset$probability,testset$Target)
roc_purchase <- performance(Pred_purchase, measure = "tpr", x.measure = "fpr")
roc_purchase_acc <- performance(Pred_purchase, "acc")

plot(roc_purchase, main= "ROC curve",ylab= "sensitivity",xlab = "1-specificity", col= "#377eb8", lwd= 4)
abline(a=0, b=1)


auc_purchase <- performance(Pred_purchase, measure = "auc")
auc<- auc_purchase@y.values[[1]]
auc  #  0.9008329


########################
########################
####  Random Forest ####
########################
########################

trainset_size <- floor(0.70 * nrow(purchase)) 
trainset_size 

# first step is to set a random seed to ensurre we get the same result each time
#All random number generators use a seed 

set.seed(3521) 

#get indices of observations to be assigned to training set...
#this is via randomly picking observations using the sample function

trainset_indices_rf <- sample(seq_len(nrow(purchase)), size = trainset_size)

#assign observations to training and testing sets

trainset_rf <- purchase[trainset_indices_rf, ]
testset_rf <- purchase[-trainset_indices_rf, ]

#Build random forest model
purchase.rf <- randomForest(as.factor(Target) ~ . -ID, data = trainset_rf, 
                            importance=TRUE, xtest=testset_rf[,-(1:2)],ntree=100, keep.forest = T)


#model summary
summary(purchase.rf)
print(purchase.rf)

#variables contained in model 
names(purchase.rf)

#predictions for test set
test_predictions_rf <- data.frame(testset_rf,purchase.rf$test$predicted)
test_probabilities_rf<- data.frame(test_predictions_rf, purchase.rf$test$votes [,2])

###########################################
#### AUC & ROC curve for Random Forest ####
###########################################

Pred_purchase_rf <- prediction(purchase.rf$test$votes [,2],testset_rf$Target)
roc_purchase_rf <- performance(Pred_purchase_rf, measure = "tpr", x.measure = "fpr")

plot(roc_purchase_rf, main= "ROC curve",ylab= "sensitivity",xlab = "1-specificity", col= "#377eb8", lwd=4)
abline(a=0, b=1)


auc_purchase_rf <- performance(Pred_purchase_rf, measure = "auc")
auc_rf<- auc_purchase_rf@y.values[[1]]
auc_rf  # 0.9950618

#accuracy for test set
mean(purchase.rf$test$predicted==testset_rf$Target) # 0.9926146

##############################
####   Confusion Matrix   ####
##############################

# Create a confusion matrix (along with other measures) using the table function
xtab_rf<- table(predicted=purchase.rf$test$predicted,true=testset$Target)[2:1,2:1]  # confusion matrix at threshold 0.5

caret::confusionMatrix(xtab_rf)


#Precision = TP/(TP+FP)
precision_rf <- xtab_rf[1,1]/(xtab_rf[1,1]+xtab_rf[1,2])
precision_rf # 0.9573964

#Recall = TP/(TP+FN)
recall_rf <- xtab_rf[1,1]/(xtab_rf[1,1]+xtab_rf[2,1])
recall_rf # 0.7582006

#F1
f1_rf <- 2*(precision_rf*recall_rf/(precision_rf+recall_rf))
f1_rf # 0.8462343

#################################
####   Variable Importance   ####
#################################

#quantitative measure of variable importance
vip(purchase.rf, bar = FALSE, horizontal = FALSE, size = 1.5) # vip pacakge
importance(purchase.rf)
imp<- varImpPlot(purchase.rf)


#Importance of each predictor.
importance(purchase.rf)


######################################
####   Partial Dependency Plots   ####
######################################

cluster = makeCluster(detectCores() - 1) 
registerDoParallel(cluster)


par(mfrow=c(2,3))
par.lastser <- partial(purchase.rf, pred.var = c("mth_since_last_serv"),type = c("classification"), chull = TRUE)
plot.lastser <- autoplot(par.lastser, contour = TRUE)
plot.lastser


par.ann_mil <- partial(purchase.rf, pred.var = c("annualised_mileage"), chull = TRUE)
plot.ann_mil <- autoplot(par.ann_mil, contour = TRUE)
plot.ann_mil

par.ser_dea <- partial(purchase.rf, pred.var = c("num_serv_dealer_purchased"), chull = TRUE)
plot.ser_dea <- autoplot(par.ser_dea, contour = TRUE)
plot.ser_dea

par.age_vehicle <- partial(purchase.rf, pred.var = c("age_of_vehicle_years"), chull = TRUE)
plot.age_vehicle <- autoplot(par.age_vehicle, contour = TRUE)
plot.age_vehicle

par.tot_serv <- partial(purchase.rf, pred.var = c("total_services"), chull = TRUE)
plot.tot_serv <- autoplot(par.tot_serv, contour = TRUE)
plot.tot_serv



###################################################################
####   Repurchase_Validation: probability & Class_predictions  ####
###################################################################


validation <- read.csv("C:/Users/Public/Zarmina-Data science/DAM/assignment 1/AT1-B/repurchase_validation.csv") 
validation$Target <- 0
validation <- rbind(purchase[1,], validation)

validation <- validation[-1, ]

p1<-predict(purchase.rf, newdata= validation, type = "response")

validation$Target_class <- p1

p2<- predict(purchase.rf, type = "prob", newdata= validation)

validation$Target_probability <- p2


###############################################
####  repurchase_validation_13326285.csv   ####
###############################################

validation_1 <-validation %>% select(ID, Target_class, Target_probability)
write.table(validation_1, file = "repurchase_validation_13326285.csv", sep = ",",row.names = FALSE)


# For only Positive "1" target values
validation_Positive <-validation%>% filter(Target_class==1) %>% select(ID, Target_class, Target_probability)
write.table(validation_Positive, file = "repurchase_validation_1_13326285.csv", sep = ",",row.names = FALSE)
