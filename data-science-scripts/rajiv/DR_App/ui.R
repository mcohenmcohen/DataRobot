
shinyUI(
  ui = fluidPage(theme = shinytheme("flatly"),
 # sidebarPanel(),
  mainPanel(
    tabsetPanel(
      tabPanel("EDA Stats",
               HTML("<br><br>"), 
               textOutput("Projectname"),
               textOutput("Projectid"), 
               HTML("<br><br><br>"),
               DT::dataTableOutput('EDA2')
               ),
      tabPanel("Models", 
               HTML("<br><br>"), 
               textOutput("Modelid"),
               HTML("<br><br><br>"),
               DT::dataTableOutput('models')
               ),
      tabPanel("Predict",
               HTML("<br><br>"), 
               textOutput("P_Modelid"),
               actionButton("predict", label = "Predict"),
               tableOutput("ptable"),
               tableOutput("rtable"),
               uiOutput("df_in")
               ),
      tabPanel("Predict_Batch",
               HTML("<br><br>"), 
               actionButton("predictbatch", label = "Predict_Batch"),
               HTML("<br><br>"), 
               #tableOutput("batchtable"),
               DT::dataTableOutput('colortable')
               ),
     tabPanel("GAMs",
              HTML("<br><br>"),
              plotOutput("FeatureStrengths"),
              HTML("<br><br>"),
              plotOutput("FeatureInteractions")
              )
      ) #Tabset
      #tabPanel("Testing",uiOutput("choose_columns"),verbatimTextOutput("show_vals"))
    ) # Main
  ) #Fluid
) #Shiny