import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
import imblearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from imblearn.combine import SMOTETomek
import warnings
warnings.filterwarnings("ignore", category=UserWarning)



data = pd.read_csv('C:/Users/pc/Desktop/project/Tamil/dataset/train_u6lujuX_CVtuZ9i.csv')
print(data)
print(data.info())
print(data.isnull().sum())

data['Gender'] = data['Gender'].fillna(data['Gender'].mode()[0])
data['Married'] = data['Married'].fillna(data['Married'].mode()[0])
data['Dependents'] = data['Dependents'].str.replace('+', '', regex=False)
data['Dependents'] = data['Dependents'].fillna(data['Dependents'].mode()[0])
data['Self_Employed'] = data['Self_Employed'].fillna(data['Self_Employed'].mode()[0])
data['LoanAmount'] = data['LoanAmount'].fillna(data['LoanAmount'].mode()[0])
data['Loan_Amount_Term'] = data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0])
data['Credit_History'] = data['Credit_History'].fillna(data['Credit_History'].mode()[0])

data['Gender'] = data['Gender'].astype('category')
data['Gender'] = data['Gender'].replace({'Male': 0, 'Female': 1}).astype('int64')
data['Married'] = data['Married'].replace({'No': 0, 'Yes': 1}).astype('int64')
data['Dependents'] = data['Dependents'].astype('int64')
data['Self_Employed'] = data['Self_Employed'].map({'Yes': 1, 'No': 0})
data['Self_Employed'] = data['Self_Employed'].astype('int64')
data['CoapplicantIncome'] = data['CoapplicantIncome'].astype('int64')
data['LoanAmount'] = data['LoanAmount'].astype('int64')
data['Loan_Amount_Term'] = data['Loan_Amount_Term'].astype('int64')
data['Credit_History'] = data['Credit_History'].astype('int64')




le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = le.fit_transform(data[col].astype(str))

# Split the data into X and y
y = data['Loan_Status']
X = data.drop('Loan_Status', axis=1)

# Apply SMOTETomek
smote = SMOTETomek(sampling_strategy=0.90)
X_bal, y_bal = smote.fit_resample(X, y)

# Print the class distribution before and after SMOTETomek
print("Before SMOTETomek: ", y.value_counts())
print("After SMOTETomek: ", y_bal.value_counts())


plt.figure(figsize=(12,5))
plt.subplot(121)
sns.histplot(data['ApplicantIncome'], color='r')
plt.title('Applicant Income Distribution')
plt.subplot(122)
sns.histplot(data['Credit_History'])
plt.title('Credit History Distribution')
plt.show()

plt.figure(figsize=(10,5))
sns.countplot(x='Gender', hue='Loan_Status', data=data)
plt.title('Gender vs Loan Status')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(20,5))
plt.subplot(131)
sns.countplot(x=data['Married'], hue=data['Gender'])
plt.subplot(132)
sns.countplot(x="Self_Employed", hue="Education", data=data)
plt.subplot(133)
sns.countplot(x='Property_Area', hue='Loan_Amount_Term', data=data)
plt.show()


sns.stripplot(x='Gender', y='ApplicantIncome', data=data, hue='Loan_Status', jitter=True)
plt.show()


sc = StandardScaler()
names = list(X_bal.columns)
X_bal = sc.fit_transform(X_bal)
X_bal = pd.DataFrame(X_bal, columns=names)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.25, random_state=42)
X_train.head()

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report

def decisionTree(X_train, X_test, y_train, y_test):
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    print('*** DecisionTreeClassifier ***')
    print('Confusion matrix')
    print(confusion_matrix(y_test, y_pred))
    print('Classification report')
    print(classification_report(y_test, y_pred))
    return dt

# Retrain the model with only 11 features
dt = DecisionTreeClassifier(max_features=11)
dt.fit(X_train, y_train)

# Make prediction with 11 features
dt.predict([[1,1,0,1,1,4276,1542,145,240,0,0,1]])

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

def randomForest(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print('*** RandomForestClassifier ***')
    print('Confusion matrix')
    print(confusion_matrix(y_test, y_pred))
    print('Classification report')
    print(classification_report(y_test, y_pred))
    return rf
    
rf = randomForest(X_train, X_test, y_train, y_test)
rf.predict([[1,1,0,1,1,4276,1542,145,240,0,0,1]])


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

def KNN(X_train, X_test, y_train, y_test):
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print('*** KNeighborsClassifier ***')
    print('Confusion matrix')
    print(confusion_matrix(y_test, y_pred))
    print('Classification report')
    print(classification_report(y_test, y_pred))
    return knn

from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report
import xgboost

def xgboost(X_train, X_test, y_train, y_test):
    xg=GradientBoostingClassifier()
    xg.fit(X_train,y_train)
    yPred=xg.predict(X_test)
    print('***GradientBoostingClassifier***')
    print('Confusion matrix')
    print(confusion_matrix(y_test,yPred))
    print('classification report')
    print(classification_report(y_test,yPred))
    return xg


import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
classifier = Sequential()
classifier.add(Dense(units=100, activation='relu', input_dim=12))
classifier.add(Dense(units=50, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_history = classifier.fit(X_train, y_train, batch_size=100, validation_split=0.2, epochs=100)


dt = decisionTree(X_train, X_test, y_train, y_test)
# Make prediction with 11 features
dt.predict([[1,1,0,1,1,4276,1542,145,240,0,0,1]])
rf = randomForest(X_train, X_test, y_train, y_test)
rf.predict([[1,1,0,1,1,4276,1542,145,240,0,0,1]])
knn = KNN(X_train, X_test, y_train, y_test)
knn.predict([[1,1,0,1,1,4276,1542,145,240,0,0,1]])
xg = xgboost(X_train, X_test, y_train, y_test)
xg.predict([[1,1,0,1,1,4276,1542,145,240,0,0,1]])

classifier.save("loan.h5")

y_pred = classifier.predict(X_test)

print(y_pred)

y_pred = (y_pred > 0.5)

print(y_pred)

unsqueezed_text = y_pred.squeeze()

print(unsqueezed_text)

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

def predict_exit(sample_value, sc, model):
    sample_value = np.array(sample_value)
    sample_value = sample_value.reshape(1, -1)
    sample_value = sc.transform(sample_value)
    # add return statement to return the predicted value
    return model.predict(sample_value)

sample_value = [[1, 1, 0, 1, 1, 4276, 1542, 145, 240, 0, 0]]

# Define and train a logistic regression model
X_train = X_bal.iloc[:, :-1]  # exclude the last column
y_train = y_bal
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
model = LogisticRegression()
model.fit(X_train, y_train)

# Call predict_exit and store the predicted value
prediction = predict_exit(sample_value, sc, model)

# Print the prediction
if prediction > 0.5:
    print('Prediction: High chance of loan approval!')
else:
    print('Prediction: Low chance of loan approval.')

def compareModels(X_train,X_test,y_train,y_test):
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train, y_train)
    try:
        y_pred = decision_tree.predict(X_test)
    except:
        y_pred = np.zeros(len(y_test))
        print("An error occurred while predicting values.")
    print("Decision Tree Model")
    print("Accuracy Score:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:", confusion_matrix(y_test, y_pred))
    print("Classification Report:", classification_report(y_test, y_pred))
    print('-'*100)


    random_forest = RandomForestClassifier()
    
    random_forest.fit(X_train, y_train)
    try:
        y_pred = random_forest.predict(X_test)
    except:
        y_pred = np.zeros(len(y_test))
        print("An error occurred while predicting values.")
    y_pred = random_forest.predict(X_test)
    print("Random Forest Model")
    print("Accuracy Score:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:", confusion_matrix(y_test, y_pred))
    print("Classification Report:", classification_report(y_test, y_pred))
    print('_'*100)

    xgb = XGBClassifier()
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    print("XGB Model")
    print("Accuracy Score:", accuracy_score(y_pred, y_test))
    print("Confusion Matrix:", confusion_matrix(y_test, y_pred))
    print("Classification Report:", classification_report(y_test, y_pred))
    print('-'*100)

    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print("KNN Model")
    print("Accuracy Score:", accuracy_score(y_pred, y_test))
    print("Confusion Matrix:", confusion_matrix(y_test, y_pred))
    print("Classification Report:", classification_report(y_test, y_pred))
    print('-'*100)


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


f1 = f1_score(y_test, y_pred, average='weighted')
print("F1 Score:", f1)

cv = cross_val_score(rf, X_bal, y_bal, cv=5)
print("Cross Validation Score:", np.mean(cv))


pickle.dump(model,open('rdf.pkl','wb'))
