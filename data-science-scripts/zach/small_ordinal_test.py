import pandas as pd
from ModelingMachine.engine.tasks2.ordinal_encoder import OrdinalEncoder
df_train = pd.DataFrame(data={'Holiday (actual)': ['Yes', 'No', 'Yes', 'No', 'other']})
df_test = pd.DataFrame(data={'Holiday (actual)': [np.nan, 'NEW', 'Yes', 'No']})
ord_enc = OrdinalEncoder(columns=['Holiday (actual)'], min_support=2)
encoded_arr = ord_enc.fit_transform(df_train)
print(encoded_arr)
print(ord_enc.transform(df_test))
