library(tidymodels)
library(vroom)
library(embed)
library(vroom)
library(discrim)
library(keras)
library(parsnip)
library(nnet)

trainSet = vroom("train.csv") %>% select(-id)
testSet = vroom("test.csv") 


my_recipe <- recipe(type ~ ., data=trainSet) %>%
  step_lencode_glm(color, outcome = vars(type)) %>%
  step_range(all_numeric_predictors(), min = 0, max = 1)

nn_model = mlp(hidden_units = tune(), epochs = 50) %>% 
  set_engine("nnet") %>% 
  set_mode("classification")

nn_workflow = workflow()  %>%
  add_recipe(my_recipe) %>% add_model(nn_model)

tuning_grid = grid_regular(hidden_units(range = c(1,99)), levels = 10)

folds = vfold_cv(trainSet, v = 5, repeats = 1)

CV_results = nn_workflow %>% tune_grid(resamples = folds, grid = tuning_grid,
                                             metrics = metric_set(accuracy))

bestTune = CV_results %>% select_best("accuracy")

final_wf = nn_workflow %>% finalize_workflow(bestTune) %>% fit(trainSet)


ggg = predict(final_wf, new_data = testSet, type = "class")

sub = testSet %>% mutate(
  type = ggg$.pred_class,
  id = id
  
) %>% select(id, type)

CV_results %>% collect_metrics() %>% filter(.metric == "accuracy") %>%
  ggplot(aes(x=hidden_units, y = mean)) + geom_line()





vroom_write(sub, "nnn.csv", delim = ",")






