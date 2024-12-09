import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("assets/general_data.csv", sep=",")
data = data.dropna()
data.drop(['EmployeeCount', 'EmployeeID', 'StandardHours', 'Over18', 'Gender', 'MaritalStatus', 'EducationField'],
          axis=1, inplace=True)

# %%
from sklearn.preprocessing import LabelEncoder

label_encoder_y = LabelEncoder()
data['Attrition'] = label_encoder_y.fit_transform(data['Attrition'])

# %%

corr_cols = data[['Attrition', 'Age', 'DistanceFromHome',
                  'Education', 'JobLevel', 'MonthlyIncome',
                  'NumCompaniesWorked', 'PercentSalaryHike', 'StockOptionLevel',
                  'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany',
                  'YearsSinceLastPromotion', 'YearsWithCurrManager', 'PerformanceRating',
                  'JobInvolvement', 'EnvironmentSatisfaction', 'JobSatisfaction',
                  'WorkLifeBalance']]

corr = corr_cols.corr()
plt.figure(figsize=(18, 10))
sns.heatmap(corr, annot=True)
plt.show()

# %%
# one hot encode the following fields
print(data['BusinessTravel'].unique())
print(data['Department'].unique())
print(data['JobRole'].unique())

# %%
data_encoded = pd.get_dummies(data, columns=['BusinessTravel', 'Department', 'JobRole'], drop_first=True, dtype=int)
# data_encoded.rename(columns={"BusinessTravel_Non-Travel": "BusinessTravel_None"}, inplace=True)

# %%
data_encoded[['Age_sd', 'DistanceFromHome_sd', 'MonthlyIncome_sd', 'NumCompaniesWorked_sd', 'PercentSalaryHike_sd',
              'TotalWorkingYears_sd',
              'TrainingTimesLastYear_sd', 'YearsAtCompany_sd', 'YearsSinceLastPromotion_sd', 'YearsWithCurrManager_sd',
              'PerformanceRating_sd', 'JobInvolvement_sd', 'EnvironmentSatisfaction_sd', 'JobSatisfaction_sd',
              'WorkLifeBalance_sd']] = (
    StandardScaler().fit_transform(
        data_encoded[['Age', 'DistanceFromHome', 'MonthlyIncome', 'NumCompaniesWorked', 'PercentSalaryHike',
                      'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany', 'YearsSinceLastPromotion',
                      'YearsWithCurrManager', 'PerformanceRating', 'JobInvolvement', 'EnvironmentSatisfaction',
                      'JobSatisfaction', 'WorkLifeBalance']]))


# %%
# Use the function below to create a training [70%], validation [15%], and test set [15%].
def train_validate_test_split(df, train_percent=.7, validate_percent=.15, seed=None):
    # reset the index
    df = df.reset_index(drop=True)

    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end

    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    return train, validate, test


train, validate, test = train_validate_test_split(data_encoded)
# %%
# set the smote up
y = train['Attrition']
X = train.drop(['Attrition'], axis=1)
sm = SMOTE(random_state=42)
X_sm, y_sm = sm.fit_resample(X, y)
# Notice that the shape of X before and after rebalancing.
# The data frame has grown to include more cases of where employee attrition = 1.
print(f'''Shape of X before SMOTE: {X.shape}
Shape of X after SMOTE: {X_sm.shape}''')
y_sm.mean()
# Remerge the data
tr_sm = pd.concat([y_sm, X_sm], axis=1)

tr_sm.to_csv(r'assets/train.csv')
validate.to_csv(r'assets/validate.csv')
test.to_csv(r'assets/test.csv')

# %%
# start model training

# Import Libraries
import statsmodels.formula.api as sm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, roc_curve, \
    ConfusionMatrixDisplay

# Read in the data files
train = pd.read_csv("assets/train.csv", sep=",")
validate = pd.read_csv("assets/validate.csv", sep=",")
test = pd.read_csv("assets/test.csv", sep=",")

# %%
# Drop an extraneous column and determine the shape of the train data frame.
train = train.drop(['Unnamed: 0'], axis=1)
test = test.drop(['Unnamed: 0'], axis=1)
validate = validate.drop(['Unnamed: 0'], axis=1)

# %%
# Build a logistic regression model
model_og = sm.logit(formula="Attrition ~ Age_sd + DistanceFromHome_sd + Education + JobLevel + "
                            "MonthlyIncome_sd + NumCompaniesWorked_sd + PercentSalaryHike_sd + "
                            "StockOptionLevel + TotalWorkingYears_sd + TrainingTimesLastYear_sd + "
                            "YearsAtCompany_sd + YearsSinceLastPromotion_sd + YearsWithCurrManager_sd + "
                            "PerformanceRating_sd + JobInvolvement_sd + EnvironmentSatisfaction_sd + "
                            "JobSatisfaction_sd + WorkLifeBalance_sd + "
                            "BusinessTravel_Travel_Frequently + BusinessTravel_Travel_Rarely + "
                            "Department_Research_Development + "
                            "Department_Sales + "
                            "JobRole_Human_Resources + JobRole_Laboratory_Technician + JobRole_Manager + "
                            "JobRole_Manufacturing_Director + JobRole_Research_Director + "
                            "JobRole_Research_Scientist + JobRole_Sales_Executive + "
                            "JobRole_Sales_Representative", data=train).fit()

#%%
model_cat = sm.logit(formula="Attrition ~ Age_sd + DistanceFromHome_sd + Education + JobLevel + "
                            "MonthlyIncome_sd + NumCompaniesWorked_sd + PercentSalaryHike_sd + "
                            "StockOptionLevel + TotalWorkingYears_sd + TrainingTimesLastYear_sd + "
                            "YearsAtCompany_sd + YearsSinceLastPromotion_sd + YearsWithCurrManager_sd + "
                            "PerformanceRating_sd + JobInvolvement_sd + EnvironmentSatisfaction_sd + "
                            "JobSatisfaction_sd + WorkLifeBalance_sd + "
                            "BusinessTravel_Travel_Frequently + BusinessTravel_Travel_Rarely + "
                            "Department_Research_Development + "
                            "Department_Sales", data=train).fit()

#%%
model_quant = sm.logit(formula="Attrition ~ Age_sd + DistanceFromHome_sd + Education + JobLevel + "
                               "MonthlyIncome_sd + NumCompaniesWorked_sd + PercentSalaryHike_sd + "
                               "StockOptionLevel + TotalWorkingYears_sd + TrainingTimesLastYear_sd + "
                               "YearsAtCompany_sd + YearsSinceLastPromotion_sd + YearsWithCurrManager_sd + "
                               "PerformanceRating_sd + JobInvolvement_sd + EnvironmentSatisfaction_sd + "
                               "JobSatisfaction_sd + WorkLifeBalance_sd", data=train).fit()

#%%
model_relevant = sm.logit(formula="Attrition ~ DistanceFromHome_sd + MonthlyIncome_sd + PercentSalaryHike_sd + "
                                  "TrainingTimesLastYear_sd + YearsSinceLastPromotion_sd + EnvironmentSatisfaction_sd + "
                                  "JobSatisfaction_sd + WorkLifeBalance_sd", data=train).fit()

# %%
# Print the summary of the model
print(model_og.summary())
print(model_cat.summary())
print(model_quant.summary())
print(model_relevant.summary())


#%%
# Predict the values of the train and test data
train['pred'] = model_quant.predict(train)
validate['pred'] = model_quant.predict(validate)
test['pred'] = model_quant.predict(test)

#%%
# Create a confusion matrix
y_pred = model_quant.predict(test)

# Y-pred are the probabilities that each row (employee) in the test set will attrite.
# To produce a confusion matrix work we need to change that those probabilities to binary (0,1) values.
# This can be done by rounding the y_pred values
prediction = list(map(round, y_pred))
sns.displot(y_pred, kde=False, rug=False)
plt.show()

#%%
cm = confusion_matrix(test['Attrition'], prediction)
print ("Confusion Matrix : \n", cm)

# We also can use sklearn to calculate the overall accuracy of the model.  A flawed, but helpful metric.
print('Test accuracy = ', accuracy_score(test['Attrition'], prediction))

