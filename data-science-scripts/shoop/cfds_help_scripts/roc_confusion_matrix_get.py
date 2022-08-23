import datarobot as dr
import numpy as np

# Credentials (PROD)
API_KEY = "INSERTAPIKEY"
API_ENDPOINT = "https://app.datarobot.com/api/v2"

dr.Client(token=API_KEY, endpoint=API_ENDPOINT)

# Project and Model
project = dr.Project.get("61e6be93448c47e02a4775ed")
model = dr.Model.get(
    project=project.id,
    model_id="61e6beee07c3598a23f714af",
)

# ROC Curve
roc_cv = model.get_roc_curve(
    'validation',
    fallback_to_parent_insights=True,
)
best_f1_cv = roc_cv.get_best_f1_threshold()
print(f"Best F1 threshold: {best_f1_cv}")

# ROC Curve Confusion Matrix
# NOTE: model.get_confusion_chart is meant for Multiclass Confusion Matrix
# Use RocCurve.roc_points attribute for Binary Classification Confusion Matrix
roc_cv_points = roc_cv.roc_points
confusion_matrix_dict = {}
for roc_point in roc_cv_points:
    if roc_point.get('threshold') == best_f1_cv:
        confusion_matrix_dict = roc_point
        break

confusion_matrix_array = np.array(
    [
        [confusion_matrix_dict['true_negative_score'], confusion_matrix_dict['false_positive_score']],
        [confusion_matrix_dict['false_negative_score'], confusion_matrix_dict['true_positive_score']],
    ]
)
print("Confusion Matrix for Maximize F1 threshold")
print(f"{confusion_matrix_array[0]}\n{confusion_matrix_array[1]}")
