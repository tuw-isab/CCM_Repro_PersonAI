import json

import numpy as np
import pandas


class NumpyAndPandasEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.float64):
            return float(obj)
        if isinstance(obj, pandas.DataFrame):
            return obj.to_dict(orient="list")
        if isinstance(obj, pandas.Series):
            return obj.to_frame().to_dict(orient="list")
        return json.JSONEncoder.default(self, obj)


def get_pipeline_params(clf):
    params = clf.get_params(deep=True)
    params["steps"] = [(x[0], str(x[1])) for x in params["steps"]]
    for step in params["steps"]:
        params.pop(step[0])
    return params
