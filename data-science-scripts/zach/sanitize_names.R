library(stringi)
name_sanitize <- function(x, remove=c('-', '$', '.', '{', '}', '"')){
  x <- stringi::stri_trim_both(x)
  for(c in remove){
    x <- stri_replace_all_fixed(x, c, '_')
  }
  blanks <- x == ''
  if(any(blanks)){
    x[blanks] <- '_blank'
  }
  return(x)
}

load_csv <- function(x, n) {
  data.table::fread(stringi::stri_paste("curl --silent ", x, " | head", " -n ", n), header=TRUE)
}
load_gz <- function(x, n) {
  data.table::fread(stringi::stri_paste("curl --silent ", x, " | gunzip | head", " -n ", n), header=TRUE)
}
load_zip <- function(x, n) {
  data.table::fread(stringi::stri_paste("curl --silent -r 0-20000 ", x, " | funzip -p | head", " -n ", n), header=TRUE)
}

load_names <- function(y, n=1){
  yaml <- yaml::yaml.load_file(y)
  out <- lapply(yaml, function(x){
    x$dataset_name <- utils::URLencode(x$dataset_name)
    print(x$dataset_name)
    if(grepl("\\.gz$", x$dataset_name)){
      print("...loading gzip")
      dat <- load_gz(x$dataset_name, n)
    } else if(grepl("\\.zip$", x$dataset_name)){
      print("...loading zip")
      dat <- load_zip(x$dataset_name, n)
    } else if(grepl("\\.csv$", x$dataset_name)) {
      print("...loading csv")
      dat <- load_csv(x$dataset_name, n)
    } else if(grepl("\\.xlsx$", x$dataset_name)) {
      warning("...cannot load .xlsx.  Sailing on")
      print("...cannot load .xlsx.  Sailing on")
      return(
        data.table(
          url = x$dataset_name,
          colnames = ''
        )
      )
    } else{
      warning(stringi::stri_paste('guessing', x$dataset_name, ' is a csv'))
      print(stringi::stri_paste('...guessing', x$dataset_name, ' is a csv'))
      dat <- load_csv(x$dataset_name, n)
    }
    return(
      list(
        url = x$dataset_name,
        colnames = colnames(dat)
      )
    )
  })
  out
}

names_200 <- load_names("~/workspace/mbtest-datasets/mbtest_datasets/data/Variety/mbtest_current_with_preds.yaml")
names_need_cleaning <- lapply(names_200, function(x){
  oldnames <- x$colnames
  newnames <- name_sanitize(oldnames)
  diff <- oldnames != newnames
  if(any(duplicated(newnames)) | any(newnames=='_blank')){
    return(
      list(
        url = x$url,
        oldnames = oldnames,
        newnames = make.names(newnames, unique = TRUE, allow_ = TRUE)
      )
    )
  }
  return(NULL)
})
keep <- !sapply(names_need_cleaning, is.null)
sum(keep)
jsonlite::toJSON(names_need_cleaning[keep], pretty=TRUE)
