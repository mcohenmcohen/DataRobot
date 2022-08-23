import datarobot as dr

# Credentials (PROD)
API_KEY = "INSERTAPIKEY"
API_ENDPOINT = "https://app.datarobot.com/api/v2"

dr.Client(token=API_KEY, endpoint=API_ENDPOINT)

# Project and Model (Bias and Fairness configured)
project = dr.Project.get("618a7b6859abea883d816e06")
model = dr.Model.get(
    project=project.id,
    model_id="61ee763eb235a06d32d0352f",
)

# Get Fairness Insights (Assuming Per-Class Bias already pre-calculated)
fairness_scores_dict = model.get_fairness_insights()
print("Fairness scores are:")
print(fairness_scores_dict)
