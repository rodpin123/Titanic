import numpy as np
import pandas as pd
import os

from sklearn.ensemble import RandomForestClassifier

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



y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('C:\my_submission.csv', index=False)
print("Your submission was successfully saved!")