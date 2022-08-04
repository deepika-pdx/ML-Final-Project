import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

rawdata = pd.read_csv('Diabetics_Prediction_Dataset/diabetes.csv')
df = rawdata[['diabetes']].copy()
df["diabetes"].replace(["No diabetes", "Diabetes"], [0, 1], inplace=True)
dummies = pd.get_dummies(rawdata.gender)
df = pd.concat([df, dummies], axis='columns')
colsToNormalize = ["cholesterol", "glucose", "hdl_chol", "chol_hdl_ratio", "age", "height", "weight",
                   "bmi", "systolic_bp", "diastolic_bp", "waist", "hip", "waist_hip_ratio"]
for i in range(0, len(colsToNormalize)):
    df[colsToNormalize[i]] = (rawdata[colsToNormalize[i]] - rawdata[colsToNormalize[i]].mean()) / rawdata[
        colsToNormalize[i]].std(ddof=False)

df["male"] = (df["male"] - df["male"].mean()) / df["male"].std(ddof=False)
df["female"] = (df["female"] - df["female"].mean()) / df["female"].std(ddof=False)

train, test = train_test_split(df, test_size=0.2)

Y_col = 'diabetes'
X_cols = df.loc[:, df.columns != Y_col].columns
X_train, X_test, y_train, y_test = train_test_split(df[X_cols], df[Y_col], test_size=0.2)

print(len(X_train))
print(len(X_train.columns))
print(len(X_test))
print(len(X_test.columns))

model = MLPClassifier()
model.fit(X_train, y_train)
print(model)

expected_y = y_test
predicted_y = model.predict(X_test)

print(metrics.classification_report(expected_y, predicted_y))
print(metrics.confusion_matrix(expected_y, predicted_y))
