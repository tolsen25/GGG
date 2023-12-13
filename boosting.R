library(tidymodels)
library(vroom)
library(embed)
library(vroom)
library(discrim)
library(keras)
library(parsnip)
library(nnet)
library(bonsai)
library(dbarts)
library(lightgbm)


trainSet = vroom("train.csv") %>% select(-id)
testSet = vroom("test.csv") 


treeModel = boost_tree(mtry = tune(), learn_rate =  tune(), trees = tune()) %>%
  set_engine("lightgbm") %>%
  set_mode("classification")


treeModel = boost_tree() %>%
  set_engine("lightgbm") %>%
  set_mode("classification")



my_recipe <- recipe(type ~ ., data=trainSet) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  step_lencode_glm(color, outcome = vars(type)) 


forestReg_workflow = workflow()  %>%
  add_recipe(my_recipe) %>% add_model(treeModel)

tuning_grid = grid_regular(mtry(range = c(1,5)), learn_rate(),trees() , levels = 5)# idk what this does

folds = vfold_cv(trainSet, v = 5, repeats = 1)

CV_results = forestReg_workflow %>% tune_grid(resamples = folds, grid = tuning_grid,
                                              metrics = metric_set(accuracy))

bestTune = CV_results %>% select_best("accuracy")

final_wf = forestReg_workflow %>% finalize_workflow(bestTune) %>% fit(trainSet)
final_wf = forestReg_workflow %>% finalize_workflow(forestReg_workflow) %>% fit(trainSet)

ggg = predict(final_wf, new_data = testSet, type = "class")

sub = testSet %>% mutate(
  type = ggg$.pred_class,
  id = id
  
) %>% select(id, type)


vroom_write(sub, "boostedTree2.csv", delim = ",")

