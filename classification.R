# make copy of original sample after split
# use median value of vehicle_count as threshold
tmas_train_class = tmas_train %>%
  mutate(traffic_volume = as.factor(ifelse(vehicle_count > 272, "High", "Low"))) %>%
  dplyr::select(-vehicle_count) %>%
  mutate(traffic_volume = factor(traffic_volume, levels = c("Low", "High")))
tmas_train_class = as.data.frame(tmas_train_class)

tmas_test_class = tmas_test %>%
  mutate(traffic_volume = as.factor(ifelse(vehicle_count > 272, "High", "Low"))) %>%
  dplyr::select(-vehicle_count) %>%
  mutate(traffic_volume = factor(traffic_volume, levels = c("Low", "High")))
tmas_test_class = as.data.frame(tmas_test_class)

# balance data
  # check training set
  round(prop.table(table(dplyr::select(tmas_train_class, traffic_volume), exclude = NULL)),
      4) * 100
  # check test set
  round(prop.table(table(dplyr::select(tmas_test_class, traffic_volume), exclude = NULL)),
      4) * 100

# logistic regression
set.seed(1234)
log_mod1 = glm(traffic_volume ~ ., data = tmas_train_class, family = "binomial")
summary(log_mod1)
log_predict1 = predict(log_mod1, tmas_test_class, type ='response')
head(log_predict1)
log_predict1 = ifelse(log_predict1 >= 0.5, 1, 0)
head(log_predict1)
log_predict1_table = table(tmas_test_class$traffic_volume, log_predict1)
log_predict1_table
sum(diag(log_predict1_table))/nrow(tmas_test_class)
set.seed(1234)
log_mod1_cv = caret::train(traffic_volume ~ .,
                           data = tmas_train_class,
                           metric = "Accuracy",
                           method = "glm",
                           family = binomial(link = "logit"),
                           trControl = trainControl(method = "cv", number = 5))

log_mod1_cv$resample %>%
  arrange(Resample) %>%
  summarise(Avg_Accuracy = mean(Accuracy))

# logistic regression 2
set.seed(1234)
log_mod2 = glm(traffic_volume ~ . -weekday_weekend, data = tmas_train_class, family = "binomial")
summary(log_mod2)
vif(log_mod2)
tmas_test_class_copy = tmas_test_class %>%
  mutate(traffic_volume_01 = ifelse(traffic_volume == "High", 1, 0))
preds = predict(log_mod1, tmas_test_class_copy, type = "response")
ideal_cutoff = optimalCutoff(actuals = tmas_test_class_copy$traffic_volume_01,
                             predictedScores = preds,
                             optimiseFor = "Both")
ideal_cutoff
preds = ifelse(preds >= ideal_cutoff, 1, 0)
preds_table = table(tmas_test_class_copy$traffic_volume, preds)
sum(diag(preds_table))/nrow(tmas_test_class_copy)

# k-NN
set.seed(1234)
knn_grid = data.frame(k = 10)
knn_mod1_cv = train(traffic_volume ~ .,
                    data = tmas_train_class,
                    method = "knn",
                    metric = "Accuracy",
                    trControl = trainControl(method = "cv", number = 5),
                    tuneGrid = knn_grid)

knn_mod1_cv$results
  # use 4 cores for parallel processing
  cl = makeCluster(4)

  # start parallel processing
  registerDoParallel(cl)

  # set up CV for best k
  set.seed(1234)
  k_cv = train(traffic_volume ~ .,
             data = tmas_train_class,
             method = "knn",
             trControl = trainControl(method = "cv", number = 5),
             tuneGrid = data.frame(k = seq(1, 30, by = 2)))

  # end parallel processing
  stopCluster(cl)
  registerDoSEQ()
plot(k_cv)
k_cv

# naïve bayes
bayes_mod1 = naiveBayes(traffic_volume ~ ., data = tmas_train_class, laplace = 1)
bayes_predict1 = predict(bayes_mod1, tmas_test_class, type = "class")
bayes_predict1_table = table(tmas_test_class$traffic_volume, bayes_predict1)
bayes_predict1_table
confusionMatrix(bayes_predict1, tmas_test_class$traffic_volume, positive = "High")
set.seed(1234)
  # 5-fold CV
  K = 5 
  folds = createFolds(tmas_train_class$traffic_volume, k = K, list = TRUE)
  # tuning grid for laplace values
  laplace_values = c(0, 0.5, 1)
  # start parallel processing
  cl = makeCluster(detectCores() - 1)
  registerDoParallel(cl)
  cv_results = foreach(lap = laplace_values, .combine = rbind, .packages = "e1071") %dopar% {
  fold_acc = sapply(folds, function(test_ind) {
    train_fold = tmas_train_class[-test_ind, ]
    val_fold = tmas_train_class[test_ind, ]
    
    bayes_mod2 = naiveBayes(traffic_volume ~., data = train_fold, laplace = lap)
    preds = predict(bayes_mod2, newdata = val_fold)
    mean(preds == val_fold$traffic_volume)
  })
  
  data.frame(laplace = lap, mean_acc = mean(fold_acc))
  }
  stopCluster(cl)
  registerDoSEQ()
  cv_results

# LDA
lda_mod1 = lda(traffic_volume ~ ., data = tmas_train_class)
lda_mod1 = lda(traffic_volume ~ . -weekday_weekend, data = tmas_train_class)
lda_preds = predict(lda_mod1, newdata = tmas_test_class)
preds = lda_preds$class
confusionMatrix(preds, tmas_test_class$traffic_volume, positive = "High")
  # find posterior probabilities
  post_probs = lda_preds$posterior[, "High"]
  roc_curve = roc(response = tmas_test_class$traffic_volume,
                predictor = post_probs,
                levels = c("Low", "High"))

  optimal_cutoff = coords(roc_curve, "best", best.method = "youden")$threshold
  optimal_cutoff
  new_preds = ifelse(post_probs > optimal_cutoff, "High", "Low")
  new_preds = factor(new_preds, levels = c("Low", "High"))
  confusionMatrix(new_preds, tmas_test_class$traffic_volume, positive = "High")
  
# QDA
qda_mod1 = qda(traffic_volume ~ . -weekday_weekend, data = tmas_train_class)
qda_preds = predict(qda_mod1, newdata = tmas_test_class)
preds = qda_preds$class
confusionMatrix(preds, tmas_test_class$traffic_volume, positive = "High")
  # find posterior probabilities
  q_post_probs = qda_preds$posterior[, "High"]
  roc_curve = roc(response = tmas_test_class$traffic_volume,
                predictor = q_post_probs,
                levels = c("Low", "High"))

  optimal_cutoff = coords(roc_curve, "best", best.method = "youden")$threshold
  optimal_cutoff
  new_preds = ifelse(q_post_probs > optimal_cutoff, "High", "Low")
  new_preds = factor(new_preds, levels = c("Low", "High"))
  confusionMatrix(new_preds, tmas_test_class$traffic_volume, positive = "High")
  
# SVMs
  # select categorical features
  facts = tmas_train_class %>%
    dplyr::select(-traffic_volume)
  # ensure they are all factors
  facts = facts %>%
    mutate(across(everything(), as.factor))
  # make sure response is a factor
  resp = as.factor(tmas_train_class$traffic_volume)
  # run MCA
  mca_mod = MCA(facts, graph = FALSE)
  # plot the results with color-coded classes
  fviz_mca_ind(mca_mod,
               habillage = resp,
               addEllipses = TRUE,
               ellipse.type = "confidence",
               repel = FALSE,
               geom = "point")
lin_svm1 = svm(traffic_volume ~., data = tmas_train_class, kernel = "linear", cost = 0.01)
summary(lin_svm1)
svm_preds = predict(lin_svm1, tmas_test_class)
confusionMatrix(svm_preds, tmas_test_class$traffic_volume, positive = "High")
set.seed(1234)
  # turn on parallel processing
  cl = makeCluster(parallel::detectCores() - 1)
  registerDoParallel(cl)
  # tune the model
  tune_out = tune(svm, traffic_volume ~., data = tmas_train_class, kernel = "linear", ranges = list(cost = c(0.01, 0.1, 1)))
  # find the model with the optimal cost value
  lin_svm2 = tune_out$best.model
  summary(lin_svm2)
  # turn off parallel processing
  stopCluster(cl)
  registerDoSEQ()
svm_preds2 = predict(lin_svm2, tmas_test_class)
confusionMatrix(svm_preds2, tmas_test_class$traffic_volume, positive = "High")

# decision trees
tree_mod1 = rpart(data = tmas_train_class,
                  traffic_volume ~ .,
                  method = "class")

rpart.plot(tree_mod1)
tree_preds = predict(tree_mod1, tmas_test_class, type = "class")
confusionMatrix(tree_preds, tmas_test_class$traffic_volume, positive = "High")
max_depth = 2:10
cv_err = numeric(length(max_depth))

for (i in seq_along(max_depth)) {
  class_trees = rpart(data = tmas_train_class,
                      traffic_volume ~ .,
                      method = "class",
                      control = rpart.control(cp = 0.001, maxdepth = max_depth[i]))
  cv_err[i] = min(class_trees$cptable[,"xerror"])
}

plot(max_depth, cv_err, type="b", xlab="Max Depth", ylab="CV Error")
tree_mod2 = rpart(data = tmas_train_class,
                  traffic_volume ~ .,
                  method = "class",
                  control = rpart.control(cp = 0.001, maxdepth = 7))

rpart.plot(tree_mod2)
tree_preds2 = predict(tree_mod2, tmas_test_class, type = "class")
confusionMatrix(tree_preds2, tmas_test_class$traffic_volume, positive = "High")

# random forest
set.seed(1234)
rf_mod1 = train(traffic_volume ~.,
                data = tmas_train_class,
                metric = "Accuracy",
                method = "rf",
                trControl = trainControl(method = "none"),
                tuneGrid = expand.grid(.mtry = 3))
rf_pred = predict(rf_mod1, tmas_test_class)
confusionMatrix(rf_pred, tmas_test_class$traffic_volume, positive = "High")
mtry_vals = 1:(ncol(tmas_train_class) - 1)
oob_err = numeric(length(mtry_vals))
  # for loop to search
  for (i in seq_along(mtry_vals)) {
    rf_tune = randomForest(traffic_volume ~ .,
                         data = tmas_train_class,
                         ntree = 100,
                         mtry = mtry_vals[i])
    oob_err[i] = rf_tune$err.rate[100, "OOB"]
  }
plot(mtry_vals, oob_err, type = "b",
     xlab = "mtry", ylab = "OOB Error")
set.seed(1234)
rf_mod2 = train(traffic_volume ~.,
                data = tmas_train_class,
                metric = "Accuracy",
                method = "rf",
                trControl = trainControl(method = "none"),
                tuneGrid = expand.grid(.mtry = 2))
rf_pred2 = predict(rf_mod2, tmas_test_class)
confusionMatrix(rf_pred2, tmas_test_class$traffic_volume, positive = "High")

# XGBTree
set.seed(1234)
xgb_mod1 = train(traffic_volume ~.,
                 data = tmas_train_class,
                 metric = "Accuracy",
                 method = "xgbTree",
                 trControl = trainControl(method = "none"),
                 tuneGrid = expand.grid(nrounds = 100,
                                        max_depth = 6,
                                        eta = 0.3,
                                        gamma = 0.01,
                                        colsample_bytree = 1,
                                        min_child_weight = 1,
                                        subsample = 1))
xgb_pred = predict(xgb_mod1, tmas_test_class)
confusionMatrix(xgb_pred, tmas_test_class$traffic_volume, positive = "High")
set.seed(1234)
  # turn on parallel processing
  cl = makeCluster(parallel::detectCores() - 1)
  registerDoParallel(cl)
  # create tuning grid
  xgb_grid = expand.grid(nrounds = c(100, 250),
                       max_depth = c(3, 5),
                       eta = c(0.05, 0.1),
                       gamma = 0,
                       colsample_bytree = c(0.6, 0.8),
                       min_child_weight = c(1, 3),
                       subsample = 0.8)
  # tune model
  xgb_mod2 = train(traffic_volume ~.,
                 data = tmas_train_class,
                 metric = "Accuracy",
                 method = "xgbTree",
                 trControl = trainControl(method = "cv",
                                          number = 5,
                                          allowParallel = TRUE),
                 tuneGrid = xgb_grid)
  # turn off parallel processing
  stopCluster(cl)
  registerDoSEQ()
best_xgb = xgb_mod2$bestTune
best_xgb
xgb_pred2 = predict(xgb_mod2, tmas_test_class)
confusionMatrix(xgb_pred2, tmas_test_class$traffic_volume, positive = "High")

# stacked classification models
set.seed(1234)
  # turn on parallel processing
  cl = makeCluster(parallel::detectCores() - 1)
  registerDoParallel(cl)
  #create folds and cv control
  folds = createFolds(tmas_train_class$traffic_volume, k = 5)
  ctrl = trainControl(method = "cv",
                    index = folds,
                    savePredictions = "final",
                    classProbs = TRUE)
  # train base models
  class_ensembleLearners = c("glm", "knn", "naive_bayes", "lda", "qda", "rpart", "rf", "xgbTree")
  class_mods = caretList(traffic_volume ~. -weekday_weekend,
                       data = tmas_train_class,
                       metric = "Accuracy",
                       methodList = class_ensembleLearners,
                       trControl = ctrl)
  # turn off parallel processing
  stopCluster(cl)
  registerDoSEQ()
class_results = resamples(class_mods)
summary(class_results)
class_cor = modelCor(class_results)
corrplot(class_cor,
           method = "color",
           type = "upper",
           addCoef.col = "black",
           tl.col = "black",
           tl.srt = 45)  
set.seed(1234)
  # turn on parallel processing
  cl = makeCluster(parallel::detectCores() - 1)
  registerDoParallel(cl)
  # train base models
  folds = createFolds(tmas_train_class$traffic_volume, k = 5)
  ctrl = trainControl(method = "cv",
                    index = folds,
                    savePredictions = "all",
                    classProbs = TRUE,
                    allowParallel = TRUE)
  class_ensembleLearners = c("naive_bayes", "rpart", "rf")
  class_mods = caretList(traffic_volume ~. -weekday_weekend,
                       data = tmas_train_class,
                       metric = "Accuracy",
                       methodList = class_ensembleLearners,
                       trControl = ctrl)
  stopCluster(cl)
  registerDoSEQ()
class_results = resamples(class_mods)
summary(class_results)
class_cor = modelCor(class_results)
corrplot(class_cor,
           method = "color",
           type = "upper",
           addCoef.col = "black",
           tl.col = "black",
           tl.srt = 45)
set.seed(1234)
class_stack = caretStack(class_mods,
                         method = "glm",
                         metric = "Accuracy",
                         trControl = trainControl(method = "cv",
                                                  number = 5,
                                                  savePredictions = "all",
                                                  classProbs = TRUE))
prob_preds = predict(class_stack, newdata = tmas_test_class)
class_stack_preds = factor(apply(prob_preds, 1, function(x) colnames(prob_preds)[which.max(x)]),
                           levels = levels(tmas_test_class$traffic_volume))
confusionMatrix(data = class_stack_preds, reference = tmas_test_class$traffic_volume, positive = "High")