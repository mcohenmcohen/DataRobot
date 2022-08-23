# Copyright 2021 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
#
#
# This is proprietary source code of DataRobot, Inc. and its affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.

# NOTES:
# Example of parameter tuning:
# https://github.com/datarobot/DataRobot/tree/master/tests/raptor_lake_templates/models/hyperparameters
#
# TODO:
# REMOVE CONSTANT COLS
# CHECK ALL STATS TOO

check <- function(X){
  stopifnot(all(c("teamid_1", "teamid_2") %in% names(X)))
  stopifnot(is.character(X[["teamid_1"]]))
  stopifnot(is.character(X[["teamid_2"]]))
}

make_diff <- function(X1, X2, X_unique){

  # Use a shared factor encoding
  X1 <- factor(X1, levels=X_unique)
  X2 <- factor(X2, levels=X_unique)
  stopifnot(length(X1) == length(X2))

  # New levels will get encoded as NA.  This will mess us up later
  # Check for new NAs, and if they are present, make them their own level
  HAS_NA <- F
  if(anyNA(X1) | anyNA(X2)){
    X1 <- addNA(X1, ifany = F)
    X2 <- addNA(X2, ifany = F)
    HAS_NA <- T
  }
  stopifnot(length(X1) == length(X2))

  # Encode as sparse matrix
  X1 <- Matrix::sparse.model.matrix(~ 0 + X1, data.frame(X1))
  X2 <- Matrix::sparse.model.matrix(~ 0 + X2, data.frame(X2))
  stopifnot(dim(X1) == dim(X2))

  # Subtract the 2 sparse matrices
  out  <- X1 - X2

  # Remove the NA column if needed
  if(HAS_NA){
    stopifnot(colnames(out)[ncol(out)] == 'X1NA')
    out <- out[,-ncol(out)]
  }

  # Name columns
  colnames(out) <- gsub('^X1', '', colnames(out))

  # Return
  return(out)
}

fit <- function(X, y, output_dir, ...){
  "
  This hook defines how DataRobot will train this task.
  DataRobot runs this hook when the task is being trained inside a blueprint.
  As an output, this hook is expected to create an artifact containing a trained object, that is then used to transform new data.

  Parameters
  -------
  X: data.frame
      Training data that DataRobot passes when this task is being trained.
  y: vector
      Project's target column.
  output_dir: str
      A path to the output folder; the artifact [in this example - containing the trained recipe] must be saved into this folder.

  Returns
  -------
  None
      fit() doesn't return anything, but must output an artifact (typically containing a trained object) into output_dir
      so that the trained object can be used during transform.
  "

  # Checks
  check(X)

  # TODO: MAKE THIS A TUNING PARAMETER
  VAR_1 <- "teamid_1"
  VAR_2 <- "teamid_2"
  ALL_STATS <- c(
    "score", "fgm", "fga", "fgm3", "fga3", "ftm", "fta", "or",
    "dr", "ast", "to", "stl", "blk", "pf")
  keep_stats <- which(paste0(ALL_STATS, '_diff') %in% names(X))
  ALL_STATS <- ALL_STATS[keep_stats]
  stopifnot(length(ALL_STATS) > 0)

  # Find unique levels of subtract vars
  X1_unique <- unique(X[[VAR_1]])
  X2_unique <- unique(X[[VAR_2]])
  X_unique <- sort(unique(c(X1_unique, X2_unique)))

  # Check
  sim <- length(X_unique) / length(union(X1_unique, X2_unique))
  stopifnot(sim > .80)

  # Make the sparse diff matrix
  X_diff <- make_diff(X[[VAR_1]], X[[VAR_2]], X_unique)
  constant_cols <- colSums(abs(sign(X_diff)))

  # Make output data structure
  transformer <- list(
    X_unique = sort(unique(c(X1_unique, X2_unique))),
    VAR_1 = VAR_1,
    VAR_2 = VAR_2,
    ALL_STATS
  )

  # Save model
  outfile <- 'r_transform.rds'
  if(
    substr(output_dir,
          nchar(output_dir),
          nchar(output_dir)) == '/'
    ) {
    seperator = ''
  } else {
    seperator = '/'
  }

  model_path <- file.path(
    paste(
      output_dir, outfile, sep=seperator
    )
  )
  saveRDS(transformer, file = model_path)
}

transform <- function(X, transformer, ...){
  "
  This hook defines how DataRobot will use the trained object from fit() to transform new data.
  As an output, this hook is expected to return the transformed data.

  Parameters
  -------
  X: data.frame
      Data that DataRobot passes for transforming.
  transformer: Any
      Trained object, extracted by DataRobot from the artifact created in fit().
      In this example, transformer contains the median values stored in a vector.

  Returns
  -------
  data.frame
      Returns the transformed dataframe
  "

  # Checks
  check(X)

  # Use the training set factor levels to encode the transform set
  X_diff <- make_diff(X1, X2, transformer[['X_unique']])

  # Checks
  stopifnot(nrow(out) == nrow(X))
  stopifnot(ncol(out) == length(transformer[['X_unique']]))
  stopifnot(all(colnames(out) == X_unique))

  # Returns
  out
}
