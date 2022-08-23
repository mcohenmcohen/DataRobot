function (x, json_cols = c("Blueprint", "Best_Parameters_P1",
                           "Best_Parameters_P2", "Best_Parameters_P3", "Best_Parameters_P4",
                           "Best_Parameters_P5", "Task_Info_Extras", "X_Text_Params"),
          pct_cols = c("Sample_Pct", "Y_Negative", "Y_Zero", "X_Sparse",
                       "X_Num_NA", "X_Num_NA_Col"), time_cols = c("Project_Date"),
          ...)
{
  for (col in json_cols) {
    data.table::set(x, j = col, value = fixPythonJSON(x[[col]]))
    data.table::set(x, j = col, value = lapply(x[[col]],
                                               jsonlite::fromJSON))
  }
  for (col in pct_cols) {
    data.table::set(x, j = col, value = as.numeric(stringi::stri_replace_all_fixed(x[[col]],
                                                                                   "%", "")))
  }
  for (col in time_cols) {
    data.table::set(x, j = col, value = as.POSIXct(x[[col]]))
  }
  data.table::setnames(x, make.names(names(x), unique = TRUE,
                                     allow_ = TRUE))
  x[, `:=`(main_args, stringi::stri_split_fixed(main_args,
                                                ";"))]
  x[, `:=`(Filename, sapply(stringi::stri_split_fixed(Filename,
                                                      "/"), function(x) x[length(x)]))]
  x[, `:=`(Max_RAM_MB, as.numeric(Max_RAM/1024^2))]
  x[, `:=`(Max_RAM_GB, as.numeric(Max_RAM/1024^3))]
  x[, `:=`(blueprint_storage_MB, blueprint_storage_size_P1/(1024^2))]
  x[, `:=`(max_vertex_size_MB, max_vertex_storage_size_P1/(1024^2))]
  x[, `:=`(holdout_time_seconds, holdout_scoring_time/1000)]
  x[, `:=`(holdout_time_minutes, holdout_time_seconds/60)]
  x[, `:=`(model_training_time_minutes, Total_Time_P1/60)]
  x[, `:=`(task, stringi::stri_paste(main_task, "-", X_tasks))]
  x[, `:=`(task, stringi::stri_replace_all_fixed(task, "/BIND",
                                                 ""))]
  x[, `:=`(task, stringi::stri_replace_all_fixed(task, "SCTXT/WNGER2/",
                                                 "wnger2"))]
  x[, `:=`(task, stringi::stri_replace_all_fixed(task, "SCTXT/CNGEC2/",
                                                 "cngec2"))]
  x[, `:=`(task, stringi::stri_replace_all_fixed(task, "SCTXT/CNGER2/",
                                                 "cnger2"))]
  x[, `:=`(task, stringi::stri_replace_all_regex(task, "(wnger2)+",
                                                 "WNGER2/"))]
  x[, `:=`(task, stringi::stri_replace_all_regex(task, "(cngec2)+",
                                                 "CNGEC2/"))]
  x[, `:=`(task, stringi::stri_replace_all_regex(task, "(cnger2)+",
                                                 "CNGER2/"))]
  all_na <- sapply(x, function(x) all(is.na(x)))
  x <- x[, !all_na, with = FALSE]
  return(x)
}
<environment: namespace:shrinkR>