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

