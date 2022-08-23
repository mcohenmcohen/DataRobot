yaml_order <- function(y){
  library('yaml')
  library('R.utils')
  list <- yaml.load_file(y)
  urls <- sapply(list, '[[', 'dataset_name')
  f <- tempfile()
  on.exit(unlink(f))
  sizes <- sapply(urls, function(x){
    gc(reset=TRUE)
    download.file(x, f)
    return(R.utils::countLines(f)[1])
  })
  unlink(f)
  return(as.yaml(list[order(sizes, decreasing=TRUE)]))
}

new <- yaml_order("~/workspace/datarobot/tests/ModelingMachine/out_of_core_rulefit_datasets_202.yaml")
cat(new, file="~/workspace/datarobot/tests/ModelingMachine/out_of_core_rulefit_datasets_202.yaml")
