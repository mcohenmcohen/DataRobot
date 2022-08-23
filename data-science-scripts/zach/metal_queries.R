# Clear
stop()
rm(list=ls(all=T))
gc(reset=T)

# Load Libraries
library(readr)
library(data.table)
library(stringr)
library(datarobot)
library(DBI)
library(odbc)

# TODO:
# Consider using model family instead of blueprint ids for similarity search

# Load the query template
query_template <- read_file('~/workspace/data-science-scripts/zach/query_template.sql')

# Define the project we're looking at and the metric we wanna maximize
# 610a877cb001c2e0e86bbe53 - SNF - R Squared - max 65
# 5e3ce4059525d0112c918ecc - basketball - R Squared - max 100 I think
# 6123a72349051fcbc993c753 - Rajiv Multi class - FVE Multinomial - max 65
# 6112fd2af2c53d2cf1e7ce40 - XOR - FVE Binomial
# 60e8903898563eda34811a57 - malware - FVE Binomial
# 6001e8b016c5a01292180f61 - contagion net - FVE Binomial
# 617e9317a1e7272ccb816dee - Hard Drive Survival Analysis
# 6180403c6715aaf8aacbc0be - HD surv 2
QUERY_PID <- '6180403c6715aaf8aacbc0be'
METRIC <- 'FVE Binomial' # R Squared, FVE Binomial, FVE Multinomial
MAX_SAMPLE_SIZE <- 100
QUERY_SET <- 'validation' # validation, crossValidation

# Load the leaderboard and format the data
query_project_leaderboard <- ListModels(QUERY_PID)

# Format the leaderboard data
# TODO: BETTER WAY TO ID MODELS TRAINED INTO VALID/HOLDOUT
query_project_summary_raw <- lapply(query_project_leaderboard, function(x){
  data.table(
    BLUEPRINT_ID = x[['blueprintId']],
    samplePct = x[['samplePct']],
    R2 = x[['metrics']][[METRIC]][[QUERY_SET]],
    modelType = x[['modelType']]
  )
})
query_project_summary <- rbindlist(query_project_summary_raw)
stopifnot("R2" %in% names(query_project_summary))
query_project_summary <- query_project_summary[is.finite(R2),]
query_project_summary <- query_project_summary[R2 != 0,]
query_project_summary <- query_project_summary[!grepl('blender', tolower(modelType)),]
query_project_summary <- query_project_summary[samplePct < MAX_SAMPLE_SIZE,]
query_project_summary <- query_project_summary[,list(R2 = max(R2)), by='BLUEPRINT_ID']
MIN_R2 <- query_project_summary[,abs(min(R2))]
query_project_summary[, R2 := asinh(R2)]
query_project_summary[,BLUEPRINT_ID := paste0("'", BLUEPRINT_ID, "'")]

# Turn the blueprint ID / R2 into a single string
PID_BID_METRIC_DATA <- query_project_summary[,paste(BLUEPRINT_ID, R2, sep=', ')]
PID_BID_METRIC_DATA <- paste('(', PID_BID_METRIC_DATA, ')')
PID_BID_METRIC_DATA <- paste(PID_BID_METRIC_DATA, collapse=',\n	    ')

# Compute the norm for the QUERY project (for use in the cosine similarity calculation later)
QUERY_NORM <- sqrt(sum(query_project_summary[['R2']] ^ 2))

# Now use the magic of string interpolation to construct the query!
# Go to https://chartio.com/datarobot/explore/ to manually run the query
METRIC <- gsub('Weighted ', '', METRIC)
query <- str_glue(query_template)
cat(query)

# TODO: Connect to snowflake via the API, run the query, and cache the results
# https://docs.snowflake.com/en/user-guide/admin-security-fed-auth-use.html#setting-up-browser-based-sso
# https://community.snowflake.com/s/article/How-TO
# https://db.rstudio.com/databases/snowflake/
# https://oh99766.us-east-1.snowflakecomputing.com
# myconn <- DBI::dbConnect(odbc::odbc(), "SNOWFLAKE_DSN_NAME", authenticator='externalbrowser')

# TODO: map the top N query results to blueprint JSONS

# TODO: use the public api to actually run the blueprint jsons on the query PID
