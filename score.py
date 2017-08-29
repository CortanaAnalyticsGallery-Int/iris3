#from azureml.datacollector import ModelDataCollector
import pickle
import sys
import os
import pandas as pd

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from azureml.dataprep.package import run
from azure.ml.api.schema.dataTypes import DataTypes
from azure.ml.api.schema.sampleDefinition import SampleDefinition
import azure.ml.api.realtime.services as amlo16n

def init():
    global clf2, inputs_dc, prediction_dc
    # load the model back from the 'outputs' folder into memory
    print("Import the model from model.pkl")
    f2 = open('./model.pkl', 'rb')
    clf2 = pickle.load(f2)
    inputs_dc = ModelDataCollector(clf2,identifier="inputs")
    prediction_dc = ModelDataCollector(clf2, identifier="prediction")

def run(npa):
    global clf2, inputs_dc, prediction_dc
    if isinstance(npa, str):
        print("convert string to array")
        finalarr = np.array(np.array(list(npa)))
    else:
        finalarr = npa
    print(finalarr.shape)
    pred = clf2.predict(npa)
    retdf = pd.DataFrame(data={"Scored Labels":np.squeeze(np.reshape(pred, newshape= [-1,1]), axis=1)})

    inputs_dc.collect(npa)
    prediction_dc.collect(pred)
    return str(retdf)



def main():
    init()
    # predict a new sample
    X_new = [[3.0, 3.6, 1.3, 0.25]]
    n=40
    random_state = np.random.RandomState(0)
    # add random features to match the training data
    X_new_with_random_features = np.c_[X_new, random_state.randn(1, n)]

    print(run(X_new_with_random_features))
    print("Calling prepare schema")

    inputs = {"npa": SampleDefinition(DataTypes.NUMPY, X_new_with_random_features)}

    amlo16n.generate_schema(inputs=inputs,
                            filepath="outputs/schema.json",
                            run_func=run)

    #amlo16n.generate_main(user_file="score.py", schema_file="outputs/schema.json",
    #                      main_file_name="outputs/main.py")

    print("End of prepare schema")

if __name__ == "__main__":
    main()