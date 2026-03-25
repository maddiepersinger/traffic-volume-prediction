# load necessary libraries
library(tidyverse)
library(dplyr)
library(caTools)
library(stats)
library(car)
library(olsrr)
library(janitor)
library(DMwR)
library(corrplot)
library(InformationValue)
library(dummy)
library(class)
library(e1071)
library(rpart)
library(rpart.plot)
library(MASS)
library(pROC)
library(caret)
library(glmnet)
library(patchwork)
library(parallel)
library(doParallel)
library(FactoMineR)
library(factoextra)
library(randomForest)
library(xgboost)
library(caretEnsemble)

# load in the csv file + indicate variable types
tmas_ga_2023 = read_csv(file = "~/Desktop/mercer/grad school/fall 25/infd799/TMAS2023_data.csv", col_types = "ffffffnfffffn")
glimpse(tmas_ga_2023)

# remove unneccessary variables
tmas_ga_2023 = tmas_ga_2023[, !(colnames(tmas_ga_2023) %in% c("State Code", "Year", "Station ID"))]
glimpse(tmas_ga_2023)

# clean variable names
tmas_ga_2023 = tmas_ga_2023 %>%
  janitor::clean_names() %>%
  rename(f_system = functional_system)

colnames(tmas_ga_2023)
attach(tmas_ga_2023)

# randomly sample data
set.seed(1234)
sample_set = sample.split(tmas_ga_2023$month, SplitRatio = 0.005)
tmas_data = subset(tmas_ga_2023, sample_set == TRUE)
attach(tmas_data)

# data preparation
sum(is.na(tmas_data))
set.seed(1234)
sample_set = sample(nrow(tmas_data), round(nrow(tmas_data)*0.75), replace = FALSE)
tmas_train = tmas_data[sample_set,]
tmas_test = tmas_data[-sample_set,]