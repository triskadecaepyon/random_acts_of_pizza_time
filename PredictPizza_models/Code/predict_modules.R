library(caret)
library(plyr)
library(ggplot2)
library(gridExtra)
library(pROC)

load(file = '/Users/zdai/Desktop/UT/random_acts_of_pizza_time/PredictPizza_models/train_select.Rdata')
levels(train_select$requester_received_pizza) <- list('fail' = FALSE, 'success' = TRUE)
save(train_select, file = '/Users/zdai/Desktop/UT/random_acts_of_pizza_time/PredictPizza_models/train_select2.Rdata')
train_select_data <- train_select
train_select_data$month_part <- as.numeric(train_select_data$month_part) - 1
train_select_data$gratitude <- as.numeric(train_select_data$gratitude) - 1
train_select_data$hyperlink <- as.numeric(train_select_data$hyperlink) - 1
train_select_data$reciprocity <- as.numeric(train_select_data$reciprocity) - 1
train_select_data$sentiment_positive <- as.numeric(train_select_data$sentiment_positive) - 1
train_select_data$sentiment_negative <- as.numeric(train_select_data$sentiment_negative) -1
train_select_data$posted_before <- as.numeric(train_select_data$posted_before) - 1
save(train_select_data, file = '/Users/zdai/Desktop/UT/random_acts_of_pizza_time/PredictPizza_models/train_select_data.Rdata')

set.seed(2014)
train_indictor <- createDataPartition(train_select$requester_received_pizza, p = .50, list = F)
train_training <- train_select[train_indictor, ]
train_testing <- train_select[-train_indictor, ]
train_training_data <- train_select_data[train_indictor, ]
train_testing_data <- train_select_data[-train_indictor, ]
save(train_training, file = '/Users/zdai/Desktop/UT/random_acts_of_pizza_time/PredictPizza_models/train_training.Rdata')
save(train_testing, file = '/Users/zdai/Desktop/UT/random_acts_of_pizza_time/PredictPizza_models/train_testing.Rdata')
save(train_training_data, file = '/Users/zdai/Desktop/UT/random_acts_of_pizza_time/PredictPizza_models/train_training_data.Rdata')
save(train_testing_data, file = '/Users/zdai/Desktop/UT/random_acts_of_pizza_time/PredictPizza_models/train_testing_data.Rdata')

################ TRAINING #######################

indicator_variables <- names(train_training)[1:length(train_training)-1]
train_control <- trainControl(method = 'cv', summaryFunction = twoClassSummary, classProbs = T)

# Model 1:Logistic Regression
sink('/Users/zdai/Desktop/UT/random_acts_of_pizza_time/PredictPizza_models/logit_m.txt')
set.seed(2014)
logistic_regression_model <- train(requester_received_pizza ~ ., data = train_training, method = 'glm', metric = 'ROC', trControl = train_control)
summary(logistic_regression_model)
sink()
logit_inf <- varImp(logistic_regression_model)
save(logistic_regression_model, file = '/Users/zdai/Desktop/UT/random_acts_of_pizza_time/PredictPizza_models/logit_model.Rdata')
sink('/Users/zdai/Desktop/UT/random_acts_of_pizza_time/PredictPizza_models/gradient_boost_model.txt')
#Model 2: GBT
gradient_boost_tune = expand.grid(interaction.depth = seq(1, 9, 2),
                       n.trees = seq(500, 2000, 500),
                       shrinkage = c(.01, .1))
set.seed(2014)
gradient_boost_model <- train(x = train_training[, indicator_variables], y = train_training$requester_received_pizza,
              method = 'gbm', tuneGrid = gradient_boost_tune,
              metric = 'ROC', verbose = F, trControl = train_control)
summary(gradient_boost_model)
sink()
gradient_boost_inf <- varImp(gradient_boost_model)
save(gradient_boost_model, file = '/Users/zdai/Desktop/UT/random_acts_of_pizza_time/PredictPizza_models/gradient_boost_model.Rdata')
present = resamples(list('Logistic regression' = logistic_regression_model, 'Gradient boost tree' = gradient_boost_model))
parallelplot(present)

# Compare Influence
plot_inf <- function(x) {
  df = data.frame(x[[1]])
  names(df) = 'Influence'
  df$variable = row.names(df)
  var_order = df$variable[order(df$importance)]
  df$variable = factor(df$variable, levels = var_order)
  
  plot <-
    ggplot(df, aes(x = Influence, y = Variable)) +
    geom_segment(aes(yend = variable), xend = 0, colour = 'grey50') +
    geom_point(size = 3, colour = '#c0571d') +
    ggtitle(x[[2]]) + theme_bw() + guides(fill = F)
  return(plot)
}
p_logit_inf <- plot_inf(logit_inf)
p_gbm_inf <- plot_inf(ggradient_boost_inf)
grid.arrange(p_logit_inf, p_gbm_inf, main = 'Variable Influence')
# Predict
logit_predict <- predict(logistic_regression_model, train_testing[, indicator_variables], type = 'prob')
gbm_predict <- predict(gradient_boost_model, train_testing[, indicator_variables], type = 'prob')
mean_predict <- (logit_predict$success + gbm_predict$success) / 2
roc(train_testing$requester_received_pizza, mean_predict)





