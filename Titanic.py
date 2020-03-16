import numpy as np
import pandas as pd
import os

import os
#for dirname, _, filenames in os.walk('.'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

train_data = pd.read_csv("./input/train.csv")
#print(train_data.head())

test_data = pd.read_csv("./input/test.csv")
#print(test_data.head())

women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = round(sum(women)/len(women),2)
print("% of women who survived:", rate_women,"%")

men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = round(sum(men)/len(men),2)
print("% of men who survived:", rate_men,"%")

survivors = train_data["Survived"]
rate_survivors = round(sum(survivors)/len(survivors), 2)
print("% of survivors in total:", rate_survivors, "%")