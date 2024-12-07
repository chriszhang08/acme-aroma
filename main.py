#%%'
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import OneHotEncoder

#%%
# import costs.csv
costs = pd.read_csv('assets/costs.csv')

#%%
# plot employee turnover over the years
fig, ax = plt.subplots(figsize=(12,8))
sns.lineplot(data=costs, x="Year", y="Employee_Turnover")
plt.xlabel("Year")
plt.ylabel("Employee Turnover")
plt.title("Employee Turnover 2012-2022")

plt.show()

#%%
# load training data
training_data = pd.read_csv('assets/train_SM.csv')
