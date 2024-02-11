from joblib.externals.cloudpickle import parametrized_type_hint_getinitargs
import numpy as np
import pandas as pd
from pandas.core.common import random_state

# Dataset importing
data = pd.read_csv("./dataset.csv")
data_visual = pd.read_csv("./dataset.csv")


# Machine Learning
from sklearn.preprocessing import LabelEncoder

df = pd.DataFrame(data)
df["Gender"] = LabelEncoder().fit_transform(df["Gender"])


# Feature Scaling
from sklearn.preprocessing import StandardScaler

df.iloc[:, 0:-1] = StandardScaler().fit_transform(df.iloc[:, 0:-1])


# Split data into train and test
from sklearn.model_selection import train_test_split, GridSearchCV

x = df.drop("Index", axis=1)
y = df["Index"]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=0
)


# Hyperparameter tuning using Grid Search
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["auto", "sqrt"],
}


from sklearn.ensemble import RandomForestClassifier


rfc = RandomForestClassifier(random_state=0)
grid_search = GridSearchCV(
    estimator=rfc,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    verbose=2,
    n_jobs=-1,
)
grid_search.fit(x_train, y_train)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_


# Best model evaluation
from sklearn.metrics import confusion_matrix, accuracy_score


print(f"Best parameters: {best_params}")

y_pred = best_model.predict(x_test)
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))


# Save best model to a file
import pickle


pkl_filename = "model.pkl"
with open(pkl_filename, "wb") as file:
    pickle.dump(best_model, file)
