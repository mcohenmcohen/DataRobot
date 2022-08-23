library(yaml)
dir <- '~/workspace/DataRobot/tests/ModelingMachine/'
f <- list.files(dir)
f <- f[grepl('yaml$', f)]
f <- paste0(dir, f)

dirty_hack <- function(x) x
y <- sapply(f, function(x){
  tryCatch(yaml.load_file(
    x,
    handlers=list(
      "bool#y"=dirty_hack,
      "bool#yes"=dirty_hack,
      "bool#T"=dirty_hack,
      "bool#TRUE"=dirty_hack,
      "bool#1"=dirty_hack
      )
    ), error=function(e) NULL)
})
y <- Reduce(c, y)
y <- unique(y)

safe_extract_from_y <- function(n){
  unlist(lapply(y, function(i){
    out <- i[[n]]
    if(is.null(out)){
      out <- NA
    }
    return(out)
  }))
}
dataset_name <- safe_extract_from_y('dataset_name')
target <- safe_extract_from_y('target')

y <- data.frame(
  dataset_name = dataset_name,
  target = target,
  stringsAsFactors=FALSE
)
y <- y[order(y$dataset_name, y$target),]
y <- unique(y)
y <- y[!is.na(y$dataset_name),]
y <- y[!is.na(y$target),]
write.csv(y, paste0(dir, 'datasets_with_targets.csv'), row.names=FALSE)

y[y$target=='TRUE',]
y[y$target=='FALSE',]

cat(as.yaml(y, column.major=F), file='all_datasets.yaml')
