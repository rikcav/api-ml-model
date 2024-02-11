import numpy as np
import pandas as pd
from pandas.core.common import random_state

data = pd.read_csv("./dataset.csv")
data_visual = pd.read_csv("./dataset.csv")


# Machine Learning
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df = pd.DataFrame(data)
df["Gender"] = le.fit_transform(df["Gender"])


# Feature Scaling
from sklearn.preprocessing import StandardScaler

std_sc = StandardScaler()
df.iloc[:, 0:-1] = std_sc.fit_transform(df.iloc[:, 0:-1])

# Split data into train and test
from sklearn.model_selection import train_test_split

x = df.drop("Index", axis=1)
y = df["Index"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)


# Ensemle Learning
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=200, criterion="entropy", random_state=0)
model.fit(x_train, y_train)


# Prediction results and metrics
from sklearn.metrics import confusion_matrix, accuracy_score

y_pred = model.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
acc_sc = accuracy_score(y_test, y_pred)
print(cm)
print(acc_sc)


# Optimal number of trees in Random Forest
def trees_in_forest_vs_acc(
    trees, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test
):
    rfc = RandomForestClassifier(
        n_estimators=trees, criterion="entropy", random_state=0
    )
    rfc.fit(x_train, y_train)
    y_pred_rfc = rfc.predict(x_test)
    acc = accuracy_score(y_test, y_pred_rfc)
    return acc


trees_list_for_randomForest = [100, 200, 300, 500, 700, 1000]
acc_scores_for_trees_RFC = []
for x in trees_list_for_randomForest:
    acc_scores_for_trees_RFC.append(trees_in_forest_vs_acc(x))

print(acc_scores_for_trees_RFC)


# Test health status
def predict_mpg(config, model):
    if type(config) == dict:
        df = pd.DataFrame(config)
    else:
        df = config

    y_pred = model.predict(df)

    if y_pred == 0:
        return "Extremely Weak"
    elif y_pred == 1:
        return "Weak"
    elif y_pred == 2:
        return "Normal"
    elif y_pred == 3:
        return "Overweight"
    elif y_pred == 4:
        return "Obesity"
    elif y_pred == 5:
        return "Extreme Obesity"


config = {"Gender": [1], "Height": [177], "Weight": [188]}

predict_mpg(config, model)


# Create model file
import pickle


pkl_filename = "model.pkl"
with open(pkl_filename, "wb") as file:
    pickle.dump(model, file)


with open(pkl_filename, "rb") as file:
    pickle_model = pickle.load(file)


score = pickle_model.score(x_test, y_test)
print(score)

y_pred_pkl = pickle_model.predict(x_test)


pkl_filename = "model.pkl"
with open(pkl_filename, "rb") as f_in:
    model = pickle.load(f_in)

predictValue = predict_mpg(config, model)
print(predictValue)
