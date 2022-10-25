
import requests
import pandas as pd

caenes = pd.read_feather("../src/data/split_train_test/test.feather")


glosa = list(caenes.glosa_caenes[0:10])

url = "http://10.91.160.65:9292/predict"

data = {
    "text" : glosa,
    "classification" : "ciuo",
    "digits" : 2
}
 
response = requests.post(url, json=data)
 
print("Status Code", response.status_code)
print("JSON Response ", response.json())
