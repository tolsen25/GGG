library(tidymodels)
library(vroom)

imputeTrain = vroom("missingTrain.csv") %>% select(-id)
trainSet = vroom("train.csv") %>% select(-id)

my_recipe <- recipe(type ~ ., data=imputeTrain) %>%
  step_impute_mean(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors())
  #step_impute_bag(type, impute_with = c("bone_length", "rotting_flesh","hair_length", "has_soul", "color"), trees = 100)

prepped = prep(my_recipe, imputeTrain)
newData = bake(prepped, new_data = imputeTrain)  


   # step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors5
   # step_other(all_nominal_predictors(), threshold = .01) %>% # combines categorical values that occur <5% into an "other" value6
   # step_dummy(all_nominal_predictors())

rmse_vec(trainSet[is.na(imputeTrain)], newData[is.na(imputeTrain)])
