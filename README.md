import pandas as pd

url = "heart_disease_health_indicators_BRFSS2015.csv"
HeartDisease = pd.read_csv(url)
HeartDisease
HeartDisease.columns

HeartDisease.isna().sum() # check for missing value
HeartDisease = HeartDisease.dropna() # remove all missing column

import matplotlib.pyplot as plt
import seaborn as sns

HeartDisease["BMI"].hist(bins = 40)
plt.title("Distribution of Body Mass Index")
plt.xlabel("BMI")
plt.ylabel("Frequency")

sns.catplot(x="Age", data = HeartDisease, kind = "count")
plt.title("Number of Respondents per Age Category")
plt.xlabel("Age")
plt.ylabel("Frequency")

HeartDisease["MentHlth"].hist(bins = 20)
plt.title("Distribution of Poor Mental Health")
plt.xlabel("Number of Days of Poor Mental Health (0-30)")
plt.ylabel("Number of Respondents")

HeartDisease["PhysHlth"].hist(bins = 15)
plt.title("Distribution of Poor Physical Health Days")
plt.xlabel("Number of Days of Poor Physical Health (0-30)")
plt.ylabel("Number of Respondents")

sns.catplot(x = "PhysActivity", y = "BMI", data = HeartDisease, kind = "box")
plt.title("How Exercise Habits Impact Body Mass Index")
plt.xlabel("Physical Activity")#(0 = No, 1 = Yes)
plt.ylabel("Distribution of Body Mass Index")

sns.catplot(x="Age", y="GenHlth", data=HeartDisease, kind = "bar")
plt.title("How Age Affects General Health Ratings")
plt.xlabel("Age")
plt.ylabel("General Health")

# Classification models:
x = HeartDisease[["BMI", "Age", "MentHlth", "PhysHlth"]]
y = HeartDisease["HeartDiseaseorAttack"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 40)

# k-nearest neighbor classification
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(x_train, y_train)
y_test_preds = knn.predict(x_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_test_preds)
cm
tn = cm[0][0]
fp = cm[0][1]
fn = cm[1][0]
tp = cm[1][1]
sensitivity = tp/(tp + fn)
sensitivity
specificity = tn/(tn + fp)
specificity

knn3 = KNeighborsClassifier(n_neighbors = 3)
knn3.fit(x_train, y_train)
y_test_pred3 = knn3.predict(x_test)
cm3 = confusion_matrix(y_test,y_test_pred3)
cm3
tn3 = cm3[0][0]
fp3 = cm3[0][1]
fn3 = cm3[1][0]
tp3 = cm3[1][1]
sensitivity3 = tp3/(tp3 + fn3)
sensitivity3
specificity3 = tn3/(tn3 + fp3)
specificity3

knn15 = KNeighborsClassifier(n_neighbors = 15)
knn15.fit(x_train, y_train)
y_test_pred15 = knn15.predict(x_test)
cm15 = confusion_matrix(y_test,y_test_pred15)
cm15
tn15 = cm15[0][0]
fp15 = cm15[0][1]
fn15 = cm15[1][0]
tp15 = cm15[1][1]
specificity15 = tn15/(tn15 + fp15)
specificity15
sensitivity15 = tp15/(tp15 + fn15)
sensitivity15

# logistic Regression
import statsmodels.formula.api as smf
logit_model = smf.logit('HeartDiseaseorAttack ~ BMI + Age + MentHlth + PhysHlth',HeartDisease).fit()
logit_model.summary()
cm_log = logit_model.pred_table()
cm_log
tn_log = cm_log[0][0]
fp_log = cm_log[0][1]
fn_log = cm_log[1][0]
tp_log = cm_log[1][1]
sensitivity_log = tp_log/(tp_log + fn_log)
sensitivity_log
specificity_log = tn_log/(tn_log + fp_log)
specificity_log

cm_log7 = logit_model.pred_table(0.7)
cm_log7
tn_log7 = cm_log7[0][0]
fp_log7 = cm_log7[0][1]
fn_log7 = cm_log7[1][0]
tp_log7 = cm_log7[1][1]
sensitivity_log7 = tp_log7/(tp_log7 + fn_log7)
sensitivity_log7
specificity_log7 = tn_log7/(tn_log7 + fp_log7)
specificity_log7

cm_log3 = logit_model.pred_table(0.3)
cm_log3
tn_log3 = cm_log3[0][0]
fp_log3 = cm_log3[0][1]
fn_log3 = cm_log3[1][0]
tp_log3 = cm_log3[1][1]
sensitivity_log3 = tp_log3/(tp_log3 + fn_log3)
sensitivity_log3
specificity_log3 = tn_log3/(tn_log3 + fp_log3)
specificity_log3

# DecisionTreeClassification
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth = 2)
tree.fit(x_train,y_train)
y_test_predd = tree.predict(x_test)
from sklearn.metrics import confusion_matrix
cm_d = confusion_matrix(y_test,y_test_predd)
cm_d 
tn_d = cm_d[0][0]
fp_d = cm_d[0][1]
fn_d = cm_d[1][0]
tp_d = cm_d[1][1]
sensitivity_d = tp_d/(tp_d + fn_d)
sensitivity_d
specificity_d = tn_d/(tn_d + fp_d)
specificity_d

tree8 = DecisionTreeClassifier(max_depth = 8)
tree8.fit(x_train,y_train)
y_test_predd8 = tree8.predict(x_test)
cm_d8 = confusion_matrix(y_test,y_test_predd8)
cm_d8
tn_d8 = cm_d8[0][0]
fp_d8 = cm_d8[0][1]
fn_d8 = cm_d8[1][0]
tp_d8 = cm_d8[1][1]
sensitivity_d8 = tp_d8/(tp_d8 + fn_d8)
sensitivity_d8
specificity_d8 = tn_d8/(tn_d8 + fp_d8)
specificity_d8

tree4 = DecisionTreeClassifier(max_depth = 4)
tree4.fit(x_train,y_train)
y_test_predd4 = tree4.predict(x_test)
cm_d4 = confusion_matrix(y_test,y_test_predd4)
cm_d4
tn_d4 = cm_d4[0][0]
fp_d4 = cm_d4[0][1]
fn_d4 = cm_d4[1][0]
tp_d4 = cm_d4[1][1]
sensitivity_d4 = tp_d4/(tp_d4 + fn_d4)
sensitivity_d4
specificity_d4 = tn_d4/(tn_d4 + fp_d4)
specificity_d4


g = sns.relplot(x="PhysHlth", y="MentHlth", hue="HeartDiseaseorAttack",data=HeartDisease)
g.set_axis_labels("Poor Physical Health Days", "Poor Mental Health Days")
g.fig.suptitle("Relationship Between Physical and Mental Health")

j = sns.jointplot(x="BMI", y="PhysHlth", data=HeartDisease, kind="scatter")
j.set_axis_labels("Body Mass Index", "Poor Physical Health Days")
j.fig.suptitle("Relationship Between BMI and Physical Health")

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
X = HeartDisease[["BMI", "Age", "MentHlth"]]
y = HeartDisease["PhysHlth"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
import statsmodels.formula.api as smf
model = smf.ols('PhysHlth ~ BMI + Age + MentHlth', data=HeartDisease).fit()
model.summary()
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
mse

model1 = smf.ols('PhysHlth ~ BMI + MentHlth', data=HeartDisease).fit()
model1.summary()
X1 = HeartDisease[["BMI", "MentHlth"]]
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y, test_size=0.2, random_state=40)
lr.fit(X1_train, y1_train)
y1_pred = lr.predict(X1_test)
mse1 = mean_squared_error(y1_test, y1_pred)
mse1

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
features = ["BMI", "Age", "MentHlth", "PhysHlth"]
X = HeartDisease[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
pca_c = pca.fit_transform(X_scaled)
pca_c
pca_df = pd.DataFrame(pca_c, columns=["PC1", "PC2"])
pca_df["HeartDisease"] = HeartDisease["HeartDiseaseorAttack"]
sns.relplot(x="PC1", y="PC2", hue="HeartDisease", data=pca_df)
plt.title("PCA of Health Indicators")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

features1 = ["BMI", "PhysHlth", "MentHlth"]
X1 = HeartDisease[features1]
X1_scaled = scaler.fit_transform(X1)
components1 = pca.fit_transform(X1_scaled)
components1
pca_df1 = pd.DataFrame(components1, columns=["PC1", "PC2"])
pca_df1["HeartDisease"] = HeartDisease["HeartDiseaseorAttack"]
sns.scatterplot(x="PC1", y="PC2", hue="HeartDisease",data=pca_df1, alpha=0.6)
plt.title("PCA of Selected Health Indicators")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

sns.boxplot(x="HeartDiseaseorAttack", y="BMI", data=HeartDisease)
plt.title("BMI by Heart Disease Status")
plt.xlabel("Heart Disease or Attack")
plt.ylabel("BMI")

HD = HeartDisease[HeartDisease["HeartDiseaseorAttack"] == 1]["BMI"]
no_HD = HeartDisease[HeartDisease["HeartDiseaseorAttack"] == 0]["BMI"]
observed_stat = HD.mean() - no_HD.mean()
observed_stat

import numpy as np
all_bmi = HeartDisease["BMI"].values
labels = HeartDisease["HeartDiseaseorAttack"].values

inertias_list = []
for i in range(1000):
    shuffled = np.random.permutation(labels)
    group1 = all_bmi[shuffled == 1]
    group0 = all_bmi[shuffled == 0]
    stat = group1.mean() - group0.mean()
    inertias_list.append(stat)
p_value = np.mean(np.abs(simulated_stats) >= abs(observed_stat))
p_value
















