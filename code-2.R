#! /usr/bin/Rscript

library(mlr3)
library(mlr3extralearners)
library(mlr3verse)
library(data.table)
library(ggplot2)
library(DataExplorer)
library(gridExtra)
library(dplyr)
library(tidyr)
library(future)
library(mlr3viz)
library(RWeka)
library(e1071)

set.seed(100)
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

#####################################################################
#调试LMT超参数
learner_LMT = lrn("classif.LMT", 
                   C = FALSE,
                   W = to_tune(0.01, 1),
                   I = to_tune(1,50),
                   predict_type = "prob"
                   )

tuner = tnr("grid_search", resolution = 5, batch_size = 10)
instance = ti(
  task = tsk1,
  learner = learner_LMT,
  resampling = rsmp("cv", folds = 2),
  measures = msr("classif.auc"),
  terminator = trm("run_time",secs = 60)
)

tuner$optimize(instance)

       I     W learner_param_vals  x_domain classif.auc
   <int> <num>             <list>    <list>       <num>
1:     1  0.01          <list[3]> <list[2]>   0.6678009


#调试svm超参数
learner_svm = lrn("classif.svm",
                   type  = "C-classification",
                   kernel = "radial",
                   cost = to_tune(0.01, 100),
                   gamma = to_tune(0.01, 1),
                   predict_type = "prob"
                   )

tuner = tnr("grid_search", resolution = 5, batch_size = 10)
instance = ti(
  task = tsk1,
  learner = learner_svm,
  resampling = rsmp("cv", folds = 2),
  measures = msr("classif.auc"),
  terminator = trm("run_time",secs = 30)
)

tuner$optimize(instance)
     cost gamma learner_param_vals  x_domain classif.auc
    <num> <num>             <list>    <list>       <num>
1: 50.005  0.01          <list[4]> <list[2]>   0.6462499

#########################################################
learners = list(
  learner_LMT = as_learner(
    lrn("classif.LMT", 
         C = FALSE, 
         I = 1, 
         W = 0.01,
         predict_type = "prob"),
    store_models = TRUE),
  learner_svm = as_learner(
    lrn("classif.svm",
         type = "C-classification",
         cost = 50.005, 
         kernel = "radial",
         gamma = 0.01, 
         predict_type = "prob"),
    store_models = TRUE)
)
po_filter = po("filter", filter = flt("auc"), filter.frac = 0.5)

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

pdf('LMT-svm_roc.pdf',width = 4,height = 4)
autoplot(bmr, type = "roc")+theme_bw()
dev.off()

pdf('LMT-svm_prc.pdf',width = 4,height = 4)
autoplot(bmr, type = "prc")+theme_bw()
dev.off()


dev.off()

