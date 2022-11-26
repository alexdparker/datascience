
#%%
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px 
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import statsmodels.api as sm    # for QQ plot
import statsmodels.formula.api as smf
import statsmodels.stats.outliers_influence as st_inf    # for Cook's distance
import scipy

#%%

### Load in some datasets

# %%
tips = sns.load_dataset('tips')
flights = sns.load_dataset('flights')

#%% [markdown]
## Tips Dataset Analysis
### Adding in some new columns / features

#%%

### Potential target variables

# tip percentage
tips['tip_percent'] = (tips.tip / tips.total_bill)*100
tips['tp_zscore'] = (tips['tip_percent'] - tips['tip_percent'].mean())/ tips['tip_percent'].std()

tips['tp_adj'] = tips['tip_percent']             # replacing outliers of dep var with the mean
tips.loc[tips['tp_zscore'] > 3, 'tp_adj'] = tips['tip_percent'].mean()

# Whether the tip percent was above or below 20%
tips['percent_above_20'] = False
tips.loc[tips['tip_percent'] >= 20,'percent_above_20'] = True

### potentially useful features

# weekend flag
tips['weekend'] = False 
tips.loc[(tips['day'].isin(['Sat','Sun'])) | ((tips['day'] == 'Fri') & (tips['time'] == 'Dinner')), 'weekend'] = True

# bucketing total bill into deciles
tips['bill_decile'] = pd.qcut(tips.total_bill, q = 10, labels= np.arange(1,11,1))

# size of bill relative to party size
tips['price_per_head'] = tips['total_bill']/tips['size']

# creating group types
labels = ['solo','couple','small_group','large_group']
tips['group_type'] = pd.cut(tips['size'], bins = [0,1.1,2.1,4.1,100.1], include_lowest = True, labels = labels, ordered = True)

# Reciprocal of total_bill
tips['tb_reciprocal'] = 1/tips['total_bill']

# Square of total bill
tips['tb_squared'] = tips['total_bill']*tips['total_bill']

# Convert smoker to boolean
tips['smoker'] = tips['smoker'].map({'Yes': True, 'No': False})


#%% [markdown]
### EDA

#%%
tips.sample(10)

#%%
# BOx plot for the target variable: tip percent
px.box(tips, x = 'tip_percent').show()

#%% 

# Heatmap
corr = tips.corr()
px.imshow(corr, text_auto= True).show('svg')

#%%

# Generate a histogram for all numerical columns
numerical_columns = tips.select_dtypes(np.number).head().columns
for i in numerical_columns:
        px.histogram(tips, 
                x = i, 
                height = 500, 
                width = 500,
                title = i).show('svg')



#%%

### A few choice scatter plots
px.scatter(tips, 
        x = 'total_bill', 
        y = 'tip_percent',
        color = 'group_type',
        hover_data=['sex','smoker','day','time'],
        title = "Relationship between total bill and tip percent",
        height = 800, 
        width  = 1000).show('vscode')

px.scatter(tips, 
        x = 'size', 
        y = 'tip_percent',
        color = 'smoker',
        hover_data=['sex','smoker','day','time'],
        title = "Relationship between party size and tip percent",
        height = 800, 
        width  = 1000).show('vscode')


#%%

### A few choice box plots
px.box(tips, 
    x = 'weekend',              
    y = 'tp_adj', 
    color = 'time',                                  
    height = 800,
    width = 1400 
    ).show()

#%%
px.scatter(tips, 
        x = 'size', 
        y = 'tp_adj', 
        color = 'smoker', 
        height = 800, 
        width  =1000,
        trendline= 'ols').show()
#%% [markdown] 

### Predicting Tip Percentage

#%%
from sklearn.model_selection import train_test_split
train, test = train_test_split(tips, train_size=0.75, random_state=456)

#%%
### Run a few OLS models

# run a model with all the features in the original dataset
model1 = smf.ols(formula= 'tip_percent ~ total_bill + size + C(sex) + C(smoker) + C(day) + C(time)', data = train)
res1 = model1.fit()

# model 2: usiong the reciprocal of total bill 
model2 = smf.ols(formula= 'tip_percent ~ tb_reciprocal + size+ C(sex) + C(smoker) + C(day) + C(time)', data = train)
res2 = model2.fit()

# model3 elminating outliers in the dep variables (mean replacement)
model3 = smf.ols(formula= 'tp_adj ~ tb_reciprocal + size + C(sex) + C(smoker) + C(day) + C(time)', data = train)
res3 = model3.fit()

# model4 elimninating some variables and addin an interaction
model4 = smf.ols(formula= 'tp_adj ~ tb_reciprocal + np.reciprocal(size)', data = train)
res4 = model4.fit()

#%%

### Return model performance stats (on the training data)
models = [res1,res2,res3,res4]

#for i in models:
#        print(i.summary())

# Training Data Stats
for i in models:
        print("R Squared: "," ",i.rsquared)

for i in models:
        print("Mean Absolute Error: "," ",np.abs(i.resid).mean())

for i in models:
        print("Mean Squared Error: "," ",np.square(i.resid).mean())

#%%

### Percentage Error Stats

# In sample
for i in models:
        print("MPE: "," ", np.mean(i.resid/train.tip_percent))

for i in models:
        print("MAPE: "," ", np.mean(np.abs(i.resid)/train.tip_percent))


# Out opf sample
for i in models:
        print("MPE (Out of Sample): "," ", np.mean((i.predict(test) - test.tip_percent)/test.tip_percent))

for i in models:
        print("MAPE (Out of Sample): "," ", np.mean(np.abs(i.predict(test) - test.tip_percent)/test.tip_percent))






#%%
# save cooks distance for 
model1_cd = st_inf.OLSInfluence(res1).summary_frame()['cooks_d']

# %%

# basic plots of model 1 performance (in sample)
px.scatter(train, 
        x = 'size', 
        y = 'model1_resid', 
        color = 'smoker', 
        size = 'total_bill',
        hover_data=['tip_percent','time','day'],
        height = 800,
        width = 800).show()

# model 1 QQ plot 
sns.kdeplot(train.model1_resid)
fig = sm.qqplot(train.model1_resid, line = 's')
plt.show()

#%%

# Algorithms
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB


# %%

titanic = sns.load_dataset('titanic')
print(titanic.info())
titanic.head()

#%%

### this reduces the memory usage by more than half
titanic = (titanic.drop(['pclass','embarked','who','deck','alive'], axis = 1)
                        .assign(age = titanic['age'].fillna(titanic['age'].median()),
                                adult_male = titanic['adult_male']*1,
                                alone = titanic['alone']*1
                                )
                        .astype({
                                'sex':'category',
                                'age':'float32',
                                'sibsp':'int32',
                                'parch':'int32',
                                'fare':'float32',
                                'embark_town':'category'
                                })
                        .dropna(axis = 0, how = 'any')
                )

print(titanic.info())


#%%
titanic_prepped = pd.get_dummies(titanic)

test, train = train_test_split(titanic_prepped, train_size = .75, random_state = 456)

train.head()

#%%

X_train =  train.drop('survived', axis = 1)
Y_train = train['survived']
X_test = test.drop('survived', axis = 1)

#%%


### Decision Tree
decision_tree = DecisionTreeClassifier() 
decision_tree.fit(X_train, Y_train)  
Y_pred = decision_tree.predict(X_test)  
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

#%%

### Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)

Y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(random_forest.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances.head(15)

#%%

# KNN 
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)  
Y_pred = knn.predict(X_test)  
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)












# %%
