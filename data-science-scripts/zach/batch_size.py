from tests.ModelingMachine.metablueprint.test_MBSeriesTwelve import MBSeriesTwelve_V4_test
from tests.ModelingMachine.metablueprint.test_MBSeriesTwelve import DictToClass

m_params = {
    'rtype': 'Binary',
    'metric': 'LogLoss',
    'metric_for_models': 'LogLoss',
    'data_size': 436057899340,
    'nrows': 1302823907,
    'x_num': 26,
    'x_cat': 13,
    'x_txt': 0,
    'x_img': 0,
}

m_params = DictToClass(m_params)
mbp = MBSeriesTwelve_V4_test()
max_batch_size = mbp.get_keras_max_batch_size(m_params=m_params)
print(max_batch_size)
per_row = mbp.get_keras_batch_size_per_row(m_params=m_params)
print(per_row)
print(per_row[0] * m_params.nrows)
print(per_row[0] * m_params.nrows * 0.025)
print(mbp.get_keras_model(m_params))
