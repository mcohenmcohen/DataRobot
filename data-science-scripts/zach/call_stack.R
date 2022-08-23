a <- function(x, named, ...){
  call_stack <- sys.calls()
  call_stack_dots_expanded <- lapply(1:length(call_stack), function(i){
    match.call(definition=a, call=call_stack[[i]], envir=sys.frames()[[1]])
  })
  call_stack_as_text <- lapply(call_stack_dots_expanded, deparse)
  me <- call_stack_as_text[length(call_stack_as_text)]
  parent <- call_stack_as_text[length(call_stack_as_text) - 1]
  root <- call_stack_as_text[1]
  print(paste('My call is', me))
  print(paste('My parent call is', parent))
  print(paste('My root call is', root))
  return(invisible())
}
b <- function(x, named, ...) a(x, named, ...)
c <- function(x, named, ...) b(x, named, ...)
d <- function(x, named, ...) c(x, named, ...)
e <- function(x, named, ...) d(x, named, ...)
f <- function(x, named, ...) e(x, named, ...)

f(1, named=1, unnamed=2)
