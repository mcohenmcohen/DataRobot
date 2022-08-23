
# Define server logic required to plot various variables against mpg
shinyServer(function(input, output, session) {

  ###################
  ## EDA
  ################## 
  output$Projectname <- renderText({ 
    paste0("Project Name: ", projectinfo$projectName)
  })
  
  output$Projectid <- renderText({ 
    paste0("Project Id: ",projectinfo$projectId)
  })
  
  output$EDA2 <- DT::renderDataTable({
    temp <- eda2[,c(3,5,1,2,4,7)]
    temp
    }
                                     
                                     )
  ###################
  ## Models
  ##################    
  
  modelID <- reactive({
    if (length(input$models_rows_selected)) { 
      temp <- modelFrame[input$models_rows_selected,]$rowid
      bestModel <<- allModels[[temp]]
      }
     bestModel
    })
  
  output$Modelid <- renderText({ 
    temp <- modelID()
    paste0("Selected Model Id: ",temp$modelId)
  })
  
  output$models <- DT::renderDataTable({
    temp <- modelFrame %>% select (-blueprintId,-featurelistId)
    temp
  }, server = FALSE,selection = 'single')
  
  ###################
  ## Predict
  ################## 
    
    # This is pulling in input fields from the EDA **THIS MAY NOT BE VALID FOR YOUR MODEL**
    # This is a problem with derived fields, such as dates
    # This may be to be redone  . . . not sure how to pull this from the API
  
  output$P_Modelid <- renderText({ 
    temp <- modelID()
      paste0("Selected Model Id: ",temp$modelId)
  })
  
  output$df_in <- renderUI({
    tagList (lapply(eda2$name,function (x)
        {df <- eda2 %>% filter (name == x)
       # print (df$featureType[1])
        numericInput(inputId = sprintf("dyinput_%s",x),label = toupper(x),value = df$importance[1])
       # if (df$featureType[1] == "Numeric") {numericInput(inputId = sprintf("dyinput_%s",x),label = toupper(x),value = df$importance[1])} else {
      #  selectInput(inputId = sprintf("dyinput_%s",x),label = toupper(x),choices = df$importance[1])}
        }
      ))
    })
      
  
  pred <- eventReactive(input$predict, {
    all_inputs <- names(session$input)
    ##Grab inputs with dyinput and put into a dataframe
    all2 <- all_inputs[sapply(all_inputs, function(x) str_detect(x, "dyinput"))]
    input_vals <- plyr::ldply(all2, function(i){
      data.frame(input_name = i, input_value = input[[i]],stringsAsFactors = FALSE)
    })
    ##Get rid of dyinput and spread them
    df2 <- input_vals %>% 
            mutate(input_name = gsub('dyinput_', '', input_name)) %>% 
            spread(key = input_name,value = input_value)
    
    #Queries production server
    bModel <- modelID()
    url = paste0(predictionServer, '/predApi/v1.0/',projectid, '/', bModel$modelId, '/reasonCodesPredictions')
    print (url)
    myRequest = httr::POST(url,
                           add_headers("datarobot-key" = drKey),
                           c(authenticate(username, apitoken)),
                           body = rjson::toJSON(unname(split(df2, 1:nrow(df2)))),
                           httr::content_type_json())
    
    response = content(myRequest, as = 'text', encoding = 'UTF-8')
    print (response)
    response
  })
  
  output$ptable <- renderTable({
    temp <- pred()
    if (is.null(jsonlite::fromJSON(temp)$message)){
    preds <- jsonlite::fromJSON(temp)$data$predictionValues[[1]]
    } else (preds <- paste0("ERROR: ",jsonlite::fromJSON(temp)$message ))
    preds
  })
  
  output$rtable <- renderTable({
    temp <- pred()
    reasoncodes <- jsonlite::fromJSON(temp)$data$reasonCodes[[1]]
    reasoncodes
  })


  ###################
  ## Batch Predict
  ################## 
  
  predbatch <- eventReactive(input$predictbatch, {
    df <- read.csv("preds.csv")
    dataset <- UploadPredictionDataset(projectid,dataSource = df)
    bModel <- modelID()
    predictJobId <- RequestPredictionsForDataset(projectid, modelId =  bModel$modelId, datasetId = dataset$id)
    predicts <- GetPredictions(projectid,predictJobId)
    tryCatch({
      FI_jobid <- RequestFeatureImpact(bModel)
      featureimpact <- GetFeatureImpactForJobId(projectid,FI_jobid)
    }, error=function(e) print("Feature impact created"))
    reasonCodeJobID <- RequestReasonCodesInitialization(bModel)
    reasonCodeJobIDInitialization <- GetReasonCodesInitializationFromJobId(projectid,reasonCodeJobID)
    reasonCodeRequest <- RequestReasonCodes(bModel, dataset$id, maxCodes = maxRCcode)
    reasonCodeRequestMetaData <- GetReasonCodesMetadataFromJobId(projectid, reasonCodeRequest, maxWait = 1800)
    reasonCodeMetadata <- GetReasonCodesMetadata(projectid, reasonCodeRequestMetaData$id)
    reasonCodeAsDataFrame <- GetAllReasonCodesRowsAsDataFrame(projectid,reasonCodeRequestMetaData$id)
    reasonCodeAsDataFrame$rowId <- NULL
    reasonCodeAsDataFrame
  })
  

  output$colortable <- DT::renderDataTable({
    df <- read.csv("preds.csv")
    rc <- predbatch()
    
    ##Isolate the columns with names and strength
    rows <- NULL
    j <- 6
    for (i in 1:maxRCcode) {
      rows <- c(rows,j,j+2)
      j <- j +5
    }
    rc3 <- rc[,rows]
    rc3 <- apply(rc3, 2, function(y) (gsub("\\+\\+\\+", "6", y)))
    rc3 <- apply(rc3, 2, function(y) (gsub("\\+\\+", "5", y)))
    rc3 <- apply(rc3, 2, function(y) (gsub("\\+", "4", y)))
    rc3 <- apply(rc3, 2, function(y) (gsub("\\-\\-\\-", "1", y)))
    rc3 <- apply(rc3, 2, function(y) (gsub("\\-\\-", "2", y)))
    rc3 <- apply(rc3, 2, function(y) (gsub("\\-", "3", y)))
    rc3 <- as.data.frame(rc3)
    
    #Create columns to hold the strength in the predictions file
    newcols <- paste0(colnames(df),"_val")
    dfnew <-  data.frame(matrix(0, ncol=length(newcols), byrow=T))
    colnames(dfnew) <- newcols
    dfnew <- c(df,dfnew)
    df4<- as.data.frame(dfnew)
    as.numeric.factor <- function(x) {as.numeric(levels(x))[x]}
    
    #Fill out dataframe
    for (i in 1:nrow(df)) {
      for (j in seq(from=1,to=maxRCcode*2, by=2)) {
        coldf <- paste0(rc3[i,j],"_val")
        df4[i,coldf] <- as.numeric.factor (rc3[i,j+1])
      }
    }
    
    #Add predictions
    temp <- NULL
    temp$prediction <- rc$prediction
    temp$predictionscore <- round(rc$class1Probability,4)
    temp <- cbind(temp,df4)
    temp <- as.data.frame(temp)
    
    #Formatting for plot
    brks <- c(0,1,2,3,4,5)
    clrs <- round(seq(255, 40, length.out = length(brks) + 1), 0) %>%
    {paste0("rgb(255,", ., ",", ., ")")}
    ##dark blue -- dark red
    clrs <- c('#FFFFFF','#2b8cbe','#90bbd9ff','#ece7f2','#FAC3C3','#FA7575','#FF0000')
    DT::datatable(temp,options = list(
                              columnDefs = list(list(targets = (which(colnames(df4)==newcols)+2), visible = FALSE))
                            )
              ) %>%
      formatStyle(
        columns = colnames(df),
        valueColumns = newcols,
        backgroundColor = styleInterval(brks, clrs)
      )
  })
  
  ###################
  ## GAMs
  ##################    
  ##Making sure the rating table is a nice dataframe, might have to remove the first few rows from the rating tables download
  ratingtable <- read.csv("Adult_Generalized_Additive2_Model_(40)_64.0_NoGender_rating_table.csv")
 
   output$FeatureStrengths<-renderPlot({
    df_rt <- ratingtable %>% 
      dplyr::group_by(Feature.Name) %>% 
      dplyr::summarize(Feature.Strength = max(Feature.Strength)) %>% 
      dplyr::arrange(desc(Feature.Strength))

    p <- ggplot(data = df_rt, aes(x = reorder(Feature.Name, Feature.Strength), y = Feature.Strength)) + 
      geom_bar(stat='identity', fill = 'blue') + coord_flip() +
      ylab('Feature Strength') + xlab('Feature Name(s)') + theme_igray() +
      ggtitle('Feature Strengths') + 
      theme(plot.title = element_text(hjust = 0.5))
    p
  })
  
   output$FeatureInteractions<-renderPlot({
     featureImportances <- ratingtable %>% 
       dplyr::group_by(Feature.Name) %>% 
       dplyr::summarize(Feature.Strength = max(Feature.Strength)) %>% 
       dplyr::arrange(desc(Feature.Strength))
     featureImportances$Feature.Name <- as.character(featureImportances$Feature.Name)
     # get the feature interaction strengths
     interactionTerms = featureImportances[substr(featureImportances$`Feature.Name`, 1, 1) == '(' & 
                                             substr(featureImportances$`Feature.Name`, nchar(featureImportances$`Feature.Name`), nchar(featureImportances$`Feature.Name`)) == ')' &
                                             grepl(' & ', featureImportances$`Feature.Name`, fixed = TRUE), ]
     interactionTerms$Feature1 = unlist(lapply(interactionTerms$`Feature.Name`, function(x) substr(unlist(strsplit(x, ' & '))[1], 3, 9999)))
     interactionTerms$Feature2 = unlist(lapply(interactionTerms$`Feature.Name`, function(x) {
       temp = unlist(strsplit(x, ' & '))[2];
       n = nchar(temp);
       return (substr(temp, 1, n - 2))
     }
     ))
     # make a grid listing the feature interaction strengths
     terms = sort(unique(c(interactionTerms$Feature1, interactionTerms$Feature2)))
     interactionGrid = expand.grid(term1 = terms, term2 = terms)
     interactionGrid$Feature.Strength = 0
     for (i in seq_len(nrow(interactionGrid)))
     {
       feature = paste0('( ', interactionGrid$term1[i], ' & ', interactionGrid$term2[i], ' )')
       if (sum(interactionTerms$`Feature.Name` == feature) > 0) 
         interactionGrid$Feature.Strength[i] = interactionTerms$Feature.Strength[interactionTerms$`Feature.Name` == feature]
       feature = paste0('( ', interactionGrid$term2[i], ' & ', interactionGrid$term1[i], ' )')
       if (sum(interactionTerms$`Feature.Name` == feature) > 0) 
         interactionGrid$Feature.Strength[i] = interactionTerms$Feature.Strength[interactionTerms$`Feature.Name` == feature]
     }
     interactionGrid$Strength[interactionGrid$Feature.Strength < 1E-50] = NA
     # plot the interaction strengths
     p <- ggplot(data = interactionGrid, aes(x = term1, y = term2, fill = Feature.Strength)) + geom_tile() +
       scale_fill_gradient(low = "green", high = "red") + theme_igray() +
       xlab('Feature') + ylab('Feature') + theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
       ggtitle('Feature Interactions') + 
       theme(plot.title = element_text(hjust = 0.5))
     
     p
     
   })
  
  
###################
## Testing
##################  
  
  # Check boxes for testing names
  output$choose_columns <- renderUI({
    colnames <- (eda2$name)
    checkboxGroupInput("columns", "Choose columns", 
                       choices  = colnames,
                       selected = colnames)
  })
  
  output$show_vals <- renderPrint({
    all_inputs <- names(session$input)
    all2 <- all_inputs[sapply(all_inputs, function(x) str_detect(x, "dyinput"))]
    input_vals <- plyr::ldply(all2, function(i){
      data.frame(input_name = i, input_value = input[[i]],stringsAsFactors = FALSE)
    })
    print (input_vals)
  })
})