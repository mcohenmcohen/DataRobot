#### DR RaPP:  Power of the API

This is a shiny/R app that highlights a number of features of the API.  

It can:

- highlight features of a model
- list the models
- perform predictions/reasoncodes where the user inputs the data
- perform batch predictions
- analyze rating table from GAM model with visualizations



To run this, you will need to install the library listed in the `global.R` file as well as add DR specific information such as your API token, project  . . . 

To run batch predictions or the rating tables visualizations, you will need to provide a csv for the app to load.

One major issue when running individual predictions is that the list of fields is generated from the best model.  However, if you have dates, then DR automatically creates derived fields.  However, you can't enter information in for the derived fields for a prediction.  Ugh.  The goal of this code was to try and avoid having to pull from the source data file, but still not sure how to do this all via the API.  So again, a caveat if using a project with derived fields. 



