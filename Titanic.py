import numpy as np
import pandas as pd
import os

import os
for dirname, _, filenames in os.walk('.'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_data = pd.read_csv("./input/train.csv")
print(train_data.head())

test_data = pd.read_csv("./input/test.csv")
print(test_data.head())

women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)

men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)