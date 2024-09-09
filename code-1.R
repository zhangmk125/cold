#! /usr/bin/Rscript

library(mlr3extralearners)
library(mlr3verse)
library(data.table)
library(ggplot2)
library(DataExplorer)
library(gridExtra)
library(dplyr)
library(tidyr)
library(future)
library(lightgbm)
library(xgboost)
library(ranger)

set.seed(2001)
lgr::get_logger("mlr3")$set_threshold("warn")
lgr::get_logger("bbotk")$set_threshold("warn")

data <- read.table("file.txt", sep = '\t',
                   header = TRUE, stringsAsFactors = FALSE,
                   na.strings = "",fill = TRUE) %>%
  select(-gene)
data$response <- factor(data$response, levels = c("1","0"),
                        labels = c("Good", "Bad"))
data[is.na(data)] = 0

tsk = as_task_classif(data, target = "response", positive = "Good")
po_scale = po("scale")
data1 = po_scale$train(list(tsk))[[1]]$data()
tsk1 = as_task_classif(data1, target = "response", positive = "Good")
splits = partition(tsk1, ratio = 0.8)


learners = list(
  learner_lgb = as_learner(
    lrn("classif.lightgbm", predict_type = "prob")),
  learner_xgb = as_learner(
    lrn("classif.xgboost", predict_type = "prob")),
  learner_rf = as_learner(
    lrn("classif.ranger", predict_type = "prob")))
po_filter = po("filter", filter = flt("auc"), filter.frac = 0.5)
tune_ps_lgb = ps(
  learning_rate = p_dbl(-2, -1, trafo = function(x) 10^x),
  max_depth = p_int(1,20))
tune_ps_xgb = ps(
  eta = p_dbl(-2, -1, trafo = function(x) 10^x),
  max_depth = p_int(1, 20))
tune_ps_rf = ps(
  mtry.ratio = p_dbl(0, 1),
  num.trees = p_int(1, 500))

learners$learner_lgb = auto_tuner(
  tuner = tnr("grid_search", resolution = 5, batch_size = 10),
  learner = learners$learner_lgb,
  measure = msr("classif.auc"),
  resampling = rsmp("cv", folds = 2),
  search_space = tune_ps_lgb
)
learners$learner_lgb$predict_sets = c("train", "test")

learners$learner_xgb = auto_tuner(
  tuner = tnr("grid_search", resolution = 5, batch_size = 10),
  learner = learners$learner_xgb,
  measure = msr("classif.auc"),
  resampling = rsmp("cv", folds = 2),
  search_space = tune_ps_xgb
)
learners$learner_xgb$predict_sets = c("train", "test")

learners$learner_rf = auto_tuner(
  tuner = tnr("grid_search", resolution = 5, batch_size = 10),
  learner = learners$learner_rf,
  measure = msr("classif.auc"),
  resampling = rsmp("cv", folds = 2),
  search_space = tune_ps_rf
)
learners$learner_rf$predict_sets = c("train", "test")

rsmp_cv5 = rsmp("cv", folds = 5)
measure = msr("classif.auc")
design = benchmark_grid(tsk1, learners, rsmp_cv5)
head(design)
bmr = benchmark(design, store_models = TRUE)
bmr
bmr$aggregate(measure)
bmr$aggregate(list(
  msr("classif.auc"),
  msr("classif.prauc"),
  msr("classif.fbeta"),
  msr("classif.precision"),
  msr("classif.recall")
))[, c(
  "learner_id",
  "classif.auc",
  "classif.prauc",
  "classif.fbeta",
  "classif.precision",
  "classif.recall"
)]

pdf('roc.pdf',width = 4,height = 4)
autoplot(bmr, type = "roc")+theme_bw()
dev.off()

pdf('prc.pdf',width = 4,height = 4)
autoplot(bmr, type = "prc")+theme_bw()
dev.off()


#输出特征重要性分数
#输出lgb特征
filter_lgb = flt("importance",
                 learner = lrn("classif.lightgbm", predict_type = "prob"))
filter_lgb$calculate(tsk_new)
importance_lgb = as.data.table(filter_lgb, keep.rownames = TRUE)[1:10]
colnames(importance_lgb) = c("Feature", "Importance")
pdf('lgb_feature.pdf',width = 4,height = 4)
ggplot(data=importance_lgb,
            aes(x = reorder(Feature, Importance), y = Importance)) +
            geom_col() + coord_flip() + xlab("") +
            ggtitle("LightGBM Feature") +
            theme(plot.title = element_text(hjust = 0.5))
importance_lgb
dev.off()

#输出xgb特征
filter_xgb = flt("importance", 
                 learner = lrn("classif.xgboost", predict_type = "prob"))
filter_xgb$calculate(tsk_new)
importance_xgb = as.data.table(filter_xgb, keep.rownames = TRUE)[1:10]
colnames(importance_xgb) = c("Feature", "Importance")
pdf('xgb_feature.pdf',width = 4,height = 4)
ggplot(data=importance_xgb,
        aes(x = reorder(Feature, Importance), y = Importance)) +
   geom_col() + coord_flip() + xlab("") + 
   ggtitle("XgBoost Feature") + 
   theme(plot.title = element_text(hjust = 0.5)) 
importance_xgb
dev.off()

#输出rf特征
filter_rf = flt("importance", 
                learner = lrn("classif.ranger", predict_type = "prob", 
                               importance = "impurity"))
filter_rf$calculate(tsk_new)
importance_rf = as.data.table(filter_rf, keep.rownames = TRUE)[1:10]
colnames(importance_rf) = c("Feature", "Importance")
pdf('rf_feature.pdf',width = 4,height = 4)
ggplot(data=importance_rf,
       aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_col() + coord_flip() + xlab("") + 
  ggtitle("RandomForest Feature") + 
  theme(plot.title = element_text(hjust = 0.5)) 
importance_rf
dev.off()
