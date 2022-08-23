library(colormap)
library(stringr)
library(ngram)
library(dplyr)

##Sets the colors 
colorblue <- colormap::colormap(c("#2b8cbe", "#ece7f2"),reverse = TRUE)
colorred <- colormap::colormap(c("#e34a33", "#fee8c8"),reverse = TRUE)

as.numeric.factor <- function(x) {as.numeric(levels(x))[x]}

##Functions expects coef_df
ngramviz <- function (x) {
  if (is.na(x) | str_count(x[1], "\\S+")<3) {
    x <- "NA - skipped"
    return(x)
  }
  x <- gsub("\n"," ",x)
  x <- preprocess(x, case = "lower", remove.punct = TRUE,
                  remove.numbers = FALSE, fix.spacing = TRUE)
  ##Get all the ngrams
  test2 <- ngram(x,n = 2)
  test1 <- ngram(x,n = 1)
  test1 <- get.ngrams(test1)
  test2 <- get.ngrams(test2)
  out <- c(test1,test2)
  outdf <- as.data.frame(out)
  outdf$textname <- outdf$out
  outdf$textname <- as.character(outdf$textname)
  ##Join ngrams
  coef_df3 <- left_join(outdf,coef_df,by = "textname") %>% filter (abscoff >0 )
  if (nrow(coef_df3)==0) {return(x)}
  coef_df3$freq<-str_count(coef_df3$textname,'\\w+')
  coef_df3 <- coef_df3 %>% arrange(desc(freq)) %>% select (textname,coefficient,freq)
  ##Find ngrams and replace them, going through the list 
  for (i in 1:nrow(coef_df3)) {
    x <- replaceword(x,coef_df3$textname[i],coef_df3$coefficient[i],coef_df3$freq[i])
  }
  x
}

replaceword <- function (x, textname,coefficient,freq) {
  if (coefficient > 0) {
      redvalue <- colorred[[round(coefficient * (length(colorred) - 1)) + 1]]
      xreplace <- paste0("<span style='background-color: ",redvalue,"'>",textname,"</span> ") 
    } else {
      bluevalue <- colorblue[[round(abs(coefficient) * (length(colorblue) - 1)) + 1]]
      xreplace <- paste0("<span style='background-color: ",bluevalue,"'>",textname,"</span> ") } 
  if (freq == 2) {
    x <- str_replace(x,coll(textname),xreplace)
  } else {
    x <- str_replace(x,regex(paste0('\\b',textname,'\\b')),xreplace)
  }
  x
}

unlistcat <- function (coflist) {
  x <- unlist(coflist)
  if (length(x) == 6) {return (x)}
}

tv_wc_htmlfile <- function (wc,df_text) {
  
  coef_df <<- wc %>% mutate (textname = as.character(ngram),abscoff = abs(coefficient))
  
  check <- df_text %>% select (textname)
  if (ncol(check)!=1 & nrow(check)<2) {
    print ("Error with df_text")
    break
  }
  
  x <- as.character(df_text$textname)
  y <- lapply(x,ngramviz)  ##Creates a list with all the text files
  y <- lapply(y,function (x) HTML(x)) 
  counter <- 0
  test <- lapply(y, function (x) {
    counter <<- counter + 1
    filename <- paste0("output_", counter, ".html")
    save_html(x, filename, background = "white")
  })
}
