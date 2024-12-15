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
model_final = sm.logit(formula="Attrition ~ Age_sd + DistanceFromHome_sd + Education + "
                               "NumCompaniesWorked_sd + PercentSalaryHike_sd + "
                               "StockOptionLevel + TotalWorkingYears_sd + TrainingTimesLastYear_sd + "
                               "PerformanceRating_sd + JobInvolvement_sd + EnvironmentSatisfaction_sd + "
                               "JobSatisfaction_sd + WorkLifeBalance_sd", data=train).fit()


#%%
# Predict the values of the train and test data
train['pred'] = model_final.predict(train)
validate['pred'] = model_final.predict(validate)
test['pred'] = model_final.predict(test)


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

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()


#%%
auc=roc_auc_score(test['Attrition'],y_pred )
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(test['Attrition'],  y_pred)

# Create ROC curve
plt.plot(fpr, tpr, label='ROC curve (area = %.2f)' %auc)
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random guess')
plt.title('ROC curve')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.grid()
plt.legend()
plt.show()

#%%

data_ns = data.drop(['BusinessTravel', 'Department', 'JobRole'], axis=1)

#%%
train_, validate_, test_ = train_validate_test_split(data_ns)

y = train_['Attrition']
X = train_.drop(['Attrition'], axis=1)
sm = SMOTE(random_state=42)
X_sm, y_sm = sm.fit_resample(X, y)
# Notice that the shape of X before and after rebalancing.
# The data frame has grown to include more cases of where employee attrition = 1.
print(f'''Shape of X before SMOTE: {X.shape}
Shape of X after SMOTE: {X_sm.shape}''')
y_sm.mean()
# Remerge the data
tr_sm_ = pd.concat([y_sm, X_sm], axis=1)

tr_sm_.to_csv(r'assets/train_.csv')
validate_.to_csv(r'assets/validate_.csv')
test_.to_csv(r'assets/test_.csv')

#%%
train_ = pd.read_csv("assets/train_.csv", sep=",")
validate_ = pd.read_csv("assets/validate_.csv", sep=",")
test_ = pd.read_csv("assets/test_.csv", sep=",")

#%%
train_ = train_.drop(['Unnamed: 0'], axis=1)
test_ = test_.drop(['Unnamed: 0'], axis=1)
validate_ = validate_.drop(['Unnamed: 0'], axis=1)

#%%
import statsmodels.formula.api as sm
model_final_ = sm.logit(formula="Attrition ~ Age + DistanceFromHome + Education + "
                               "NumCompaniesWorked + PercentSalaryHike + "
                               "StockOptionLevel + TotalWorkingYears + TrainingTimesLastYear + "
                               "PerformanceRating + JobInvolvement + EnvironmentSatisfaction + "
                               "JobSatisfaction + WorkLifeBalance", data=train_).fit()

#%%
# get the mean value of data_ns
mean_values = data_ns.mean()

# Calculate the log odds of the mean value
baseline_log_odds = model_final_.predict(mean_values)

# Exponentiate the log odds to get the baseline odds
baseline_odds = np.exp(baseline_log_odds)

# Calculate the baseline probability
baseline_probability = baseline_odds / (1 + baseline_odds)

print(f'Baseline Probability: {baseline_probability}')

#%%
data_copy = data_ns.copy()

data_copy['MonthlyIncome'] = data_copy['MonthlyIncome'] + 7500

# Calculate the log odds of the mean value
new_log_odds = model_final_.predict(data_copy.mean())

# Exponentiate the log odds to get the new odds
new_odds = np.exp(new_log_odds)

# Calculate the new probability
new_probability = new_odds / (1 + new_odds)

probability_difference = new_probability - baseline_probability

# Estimate the number of employees that will not attrite
num_employees = 3965  # Replace with actual number of employees at Acme Aroma
employees_saved = probability_difference * num_employees

# Estimate the cost savings per employee (assume a range of 50-75% of salary)
average_salary = 65104  # Replace with actual average salary
cost_savings_min = employees_saved * 0.50 * average_salary
cost_savings_max = employees_saved * 0.75 * average_salary

# Print results
print(f"Baseline probability: {baseline_probability[0]:.4f}")
print(f"New probability after initiative: {new_probability[0]:.4f}")
print(f"Employees saved due to initiative: {employees_saved[0]:.0f}")
print(f"Estimated cost savings (50% salary): ${cost_savings_min[0]:.2f}")
print(f"Estimated cost savings (75% salary): ${cost_savings_max[0]:.2f}")
