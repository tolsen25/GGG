library(tidymodels)
library(vroom)
library(embed)
library(vroom)
library(discrim)


trainSet = vroom("train.csv") %>% select(-id)
testSet = vroom("test.csv") 

bayesRegModel = naive_Bayes(Laplace = tune(), smoothness= tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes")


my_recipe <- recipe(type ~ ., data=trainSet) %>%
  step_lencode_glm(color, outcome = vars(type)) %>%
  step_pca(all_predictors(), threshold = .996)


bayesReg_workflow = workflow()  %>%
  add_recipe(my_recipe) %>% add_model(bayesRegModel)

tuning_grid = grid_regular(Laplace(), smoothness(), levels = 5)

folds = vfold_cv(trainSet, v = 5, repeats = 1)

CV_results = bayesReg_workflow %>% tune_grid(resamples = folds, grid = tuning_grid,
                                             metrics = metric_set(accuracy))

bestTune = CV_results %>% select_best("accuracy")

final_wf = bayesReg_workflow %>% finalize_workflow(bestTune) %>% fit(trainSet)


ggg = predict(final_wf, new_data = testSet, type = "class")

sub = testSet %>% mutate(
  type = ggg$.pred_class,
  id = id
  
) %>% select(id, type)


vroom_write(sub, "ggg_pca.csv", delim = ",")
