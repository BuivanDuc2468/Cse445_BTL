from chefboost import Chefboost as chef
import pandas as pd
import numpy as np


df = pd.read_csv("play.txt")
config = {'algorithm': 'C4.5'}

model = chef.fit(df.copy(), config)


for index,instance in df.iterrows():
    prediction = chef.predict(model,instance)
    actual = instance['Decision']
    print(actual , " - " , prediction)
