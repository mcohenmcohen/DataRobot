# https://datarobot.atlassian.net/browse/MBP-3704
# October 12tgh, 2018


curl -H "Content-Type: application/json" -H "datarobot-key: " -X POST --data '[{"Sepal.Length":5.9,"Sepal.Width":3,"Petal.Length":5.1,"Petal.Width":1.8,"Species":"virginica"}] ' \
-u zach@datarobot.com:TOKEN \
https://cfds-ccm-prod.orm.datarobot.com/predApi/v1.0/5bc0b538b64ee91f026dbef7/5bc0b566566ad60176107ff1/predict


curl -H "Content-Type: application/json", -H "datarobot-key: " -X POST --data '[{"Sepal.Length":5.9,"Sepal.Width":3,"Petal.Length":5.1,"Petal.Width":1.8,"Species":"virginica"}] ' -u zach@datarbot.com: https://cfds-ccm-prod.orm.datarobot.com/predApi/v1.0/5bc0b538b64ee91f026dbef7/5bc0b566566ad60176107ff1/predict



curl -i -X POST "cfds-ccm-prod.orm.datarobot.com/predApi/v1.0/5bc0b538b64ee91f026dbef7/5bc0b566566ad60176107ff1/predict" -u zach@datarbot.com: -F file=@/Users/zachary/datasets/iris_int_in_species.csv



curl -H "Content-Type: application/json", -H "datarobot-key: " -X POST --data '[{"Sepal.Length":5.9,"Sepal.Width":3,"Petal.Length":5.1,"Petal.Width":1.8,"Species":"virginica"}]' -u zach@datarbot.com:TOKEN https://cfds-ccm-prod.orm.datarobot.com/predApi/v1.0/5bc0b538b64ee91f026dbef7/5bc0b566566ad60176107ff1/predict