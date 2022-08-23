from datetime import datetime
import numpy as np
from ModelingMachine.engine.tasks2.xgb2 import XGBoostClassifier


def test_time():
    X = np.load('/tmp/X.np')
    y = np.load('/tmp/Y.np')

    model = XGBoostClassifier(
        loss='softprob',
        learning_rate=0.1,
        n_estimators=500,
        colsample_bytree=.5,
        random_state=1234,
        missing_value=-9999.0
    )

    t1 = datetime.now()
    model.fit(X, y)
    t2 = datetime.now()
    time_elapsed = (t2 - t1).total_seconds()
    print(time_elapsed)

    assert time_elapsed < 10
