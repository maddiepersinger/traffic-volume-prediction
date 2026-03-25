# mutate datasets for Linear Regression
tmas_train_mlr = tmas_train %>%
  mutate(f_system = dplyr::recode(f_system, "1" = "Interstate", "2" = "Pr. Art. - Fr_Ex", "3" = "Pr. Art. - Other", "4" = "Minor Arterial", "5" = "Major Collector", "6" = "Minor Collector", "7" = "Local")) %>%
  mutate(direction = dplyr::recode(direction, "0" = "N/S or NE/SW", "1" = "N", "2" = "NE", "3" = "E", "4" = "SE", "5" = "S", "6" = "SW", "7" = "W", "8" = "NW", "9" = "E/W or SE/NW")) %>%
  mutate(month = dplyr::recode(month, "1" = "January", "2" = "February", "3" = "March", "4" = "April", "5" = "May", "6" = "June", "7" = "July", "8" = "August", "9" = "September", "10" = "October", "11" = "November", "12" = "December")) %>%
  mutate(day_of_week = dplyr::recode(day_of_week, "1" = "Sunday", "2" = "Monday", "3" = "Tuesday", "4" = "Wednesday", "5" = "Thursday", "6" = "Friday", "7" = "Saturday"))
tmas_test_mlr = tmas_test %>%
  mutate(f_system = dplyr::recode(f_system, "1" = "Interstate", "2" = "Pr. Art. - Fr_Ex", "3" = "Pr. Art. - Other", "4" = "Minor Arterial", "5" = "Major Collector", "6" = "Minor Collector", "7" = "Local")) %>%
  mutate(direction = dplyr::recode(direction, "0" = "N/S or NE/SW", "1" = "N", "2" = "NE", "3" = "E", "4" = "SE", "5" = "S", "6" = "SW", "7" = "W", "8" = "NW", "9" = "E/W or SE/NW")) %>%
  mutate(month = dplyr::recode(month, "1" = "January", "2" = "February", "3" = "March", "4" = "April", "5" = "May", "6" = "June", "7" = "July", "8" = "August", "9" = "September", "10" = "October", "11" = "November", "12" = "December")) %>%
  mutate(day_of_week = dplyr::recode(day_of_week, "1" = "Sunday", "2" = "Monday", "3" = "Tuesday", "4" = "Wednesday", "5" = "Thursday", "6" = "Friday", "7" = "Saturday"))

# simple LM (vehicle count modeled by functional system)
lm1 = lm(data = tmas_train_mlr, vehicle_count ~ f_system)
summary(lm1)
set.seed(1234)
lm1_cv = caret::train(vehicle_count ~ f_system,
                      data = tmas_train_mlr,
                      metric = "RMSE",
                      method = "lm",
                      trControl = trainControl(method = "cv", number = 5))

lm1_cv$resample %>%
  arrange(Resample) %>%
  summarise(Avg_RMSE = mean(RMSE))

# multiple LM
mlr1 = caret::train(vehicle_count ~ .,
                    data = tmas_train_mlr,
                    metric = "RMSE",
                    method = "lm",
                    trControl = trainControl(method = "cv", number = 5))
summary(mlr1)
mlr1$resample %>%
  arrange(Resample) %>%
  summarise(Avg_RMSE = mean(RMSE))

# create lm() object for diagnostic testing
mlr2 = lm(vehicle_count ~ . - weekday_weekend, data = tmas_train_mlr)
summary(mlr2)

# diagnostic testing
mean(mlr2$residuals)
ols_plot_resid_hist(mlr2)
ols_plot_resid_fit(mlr2)
durbinWatsonTest(mlr2)
ols_plot_cooksd_chart(mlr2)
tmas_train_mlr[266, ]
  summary(tmas_train_mlr)
cooks_dist = cooks.distance(mlr2)
  thresh = 4/length(cooks_dist)
  outlier_ind = unique(which(cooks_dist > thresh))
  length(outlier_ind)
  summary(tmas_train_mlr)summary(tmas_train_mlr[-outlier_ind, ])
ols_vif_tol(mlr2)

# improve MLR
mlr3 = lm(data = tmas_train_mlr, vehicle_count ~ f_system*hours + rural_urban + lane + direction + month + day +  day_of_week)
summary(mlr3)
  # lasso regression
    # create training matrix and vector
    x_train = model.matrix(data = tmas_train_mlr, vehicle_count ~ f_system*hours + rural_urban + lane + direction + month + day + day_of_week)[, -1]
    y_train = tmas_train_mlr$vehicle_count

    #create test matrix and vector
    y_test = model.matrix(data = tmas_test_mlr, vehicle_count ~ f_system*hours + rural_urban + lane + direction + month + day + day_of_week)[, -1]
    y_test = tmas_test_mlr$vehicle_count

    # create matrix and vector of full data sample
    x = model.matrix(data = tmas_data, vehicle_count ~ f_system*hours + rural_urban + lane + direction + month + day + day_of_week)[, -1]
    y = tmas_data$vehicle_count
  lam.grid = 10^seq(10, -2, length = 50)
  lasso_mod = glmnet(x_train, y_train, alpha = 1, lambda = lam.grid)
  plot(lasso_mod)
  # find best lambda through CV
  set.seed(1234)
  cv.out = cv.glmnet(x_train, y_train, alpha = 1)
  plot(cv.out)
  
  # use lambda.1se (1-SE method) - nearly same RMSE, but less complex
  bestlam = cv.out$lambda.1se
  bestlam
  lasso.pred = predict(lasso_mod, s = bestlam, newx = x_test)
  
  # calculate RMSE
  lasso_rmse = sqrt(mean((y_test - lasso.pred)^2))
  cat("RMSE:", lasso_rmse)
  out = glmnet(x, y, alpha = 1, lambda = lam.grid)
  lasso_coef = predict(out, type = "coefficients", s = bestlam)
  coef_vec = as.vector(lasso_coef)
  names(coef_vec) = rownames(lasso_coef)
  
  # number of total variables
  length(coef_vec)
  # number of variables selected by Lasso method
  length(coef_vec[coef_vec != 0])
  coef_vec[coef_vec == 0]
  
# decision tree
tree_mod1 = rpart(data = tmas_train_mlr,
                    vehicle_count ~ . - weekday_weekend,
                    method = "anova")
rpart.plot(tree_mod1,
             type = 2,
             fallen.leaves = TRUE,
             tweak = 1.0)
tree_pred = predict(tree_mod1, newdata = tmas_test_mlr)
tree_rmse = sqrt(mean((tmas_test_mlr$vehicle_count - tree_pred)^2))
cat("RMSE:", tree_rmse)
printcp(tree_mod1)
max_depth = 2:10
cv_err = numeric(length(max_depth))

for (i in seq_along(max_depth)) {
  trees = rpart(data = tmas_train_mlr,
                vehicle_count ~ . - weekday_weekend,
                method = "anova",
                control = rpart.control(cp = 0.001, maxdepth = max_depth[i]))
  cv_err[i] = min(trees$cptable[,"xerror"])
}

plot(max_depth, cv_err, type="b", xlab="Max Depth", ylab="CV Error")

# decision tree 2
tree_mod2 = rpart(data = tmas_train_mlr,
                  vehicle_count ~ . - weekday_weekend,
                  method = "anova",
                  control = rpart.control(cp = 0.001, maxdepth = 7))

best_cp = tree_mod2$cptable[which.min(tree_mod2$cptable[,"xerror"]), "CP"]
pruned_tree = prune(tree_mod2, cp = best_cp)
par(mar = c(1, 1, 1, 1))
rpart.plot(pruned_tree, type = 2, fallen.leaves = TRUE, tweak = 1.1)
tree_pred2 = predict(tree_mod2, newdata = tmas_test_mlr)
tree_rmse2 = sqrt(mean((tmas_test_mlr$vehicle_count - tree_pred2)^2))
cat("RMSE:", tree_rmse2)

# stacking regression models
set.seed(1234)
reg_ensembleLearners = c("lm", "rpart", "lasso")
reg_mods = caretList(vehicle_count ~. -weekday_weekend,
                     data = tmas_train_mlr,
                     metric = "RMSE",
                     methodList = reg_ensembleLearners,
                     trControl = trainControl(method = "cv",
                                              number = 5,
                                              savePredictions = "final",
                                              classProbs = FALSE))

reg_results = resamples(reg_mods)
summary(reg_results)
modelCor(reg_results)
set.seed(1234)
reg_stack = caretStack(reg_mods,
                       method = "lm",
                       metric = "RMSE",
                       trControl = trainControl(method = "cv",
                                                number = 5,
                                                savePredictions = "final",
                                                classProbs = FALSE))
reg_stacked_preds = predict(reg_stack, newdata = tmas_test_mlr)
num_preds = reg_stacked_preds$pred

reg_stacked_rmse = sqrt(mean((num_preds - tmas_test_mlr$vehicle_count)^2))

cat("RMSE:", reg_stacked_rmse)

  
