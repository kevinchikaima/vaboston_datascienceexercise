import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Written by Kevin Ma
# Code is based on https://365datascience.com/tutorials/python-tutorials/predictive-model-python/

# Read the two CSV files
clinical_df = pd.read_csv('clinical.csv')
genetic_df = pd.read_csv('genomics.csv')

# Due to imcomplete data, removing columns T, N, M, Tumor.Size in calculation
clinical_df.drop(labels=['T','N','M','Tumor.Size'],axis=1,inplace=True)

# Replacing strings for labels and data cleanup
clinical_df['Outcome'].replace(['Alive','Dead'], [1, 0], inplace=True)
# clinical_df['Stage'].replace(['IA','IB','IIA','IIB','IIIA','IIIB','IV','IVB'], [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5], inplace=True)
clinical_df['Stage'].replace(['1B'], ['IB'], inplace=True)
# clinical_df['Histology'].replace(['Squamous cell carcinoma','Adenocarcinoma','Large-cell carcinoma'], [1, 2, 3], inplace=True)
# clinical_df['Primary.Site'].replace(['Left Lower Lobe','Right Lower Lobe','Left Upper Lobe','Right Upper Lobe','Left Hilar','Right Hilar','Right Middle Lobe','Both Lung'], [1, 2, 3, 4, 5, 6, 7, 8], inplace=True)
clinical_df['Primary.Site'].replace(['Righ Upper Lobe'], ['Right Upper Lobe'], inplace=True)

# Replacing categorial labels with dummies
stage = pd.get_dummies(clinical_df['Stage'],drop_first=True)
histo = pd.get_dummies(clinical_df['Histology'],drop_first=True)
primary_site = pd.get_dummies(clinical_df['Primary.Site'],drop_first=True)
clinical_df.drop(columns=['Stage','Histology','Primary.Site'],inplace=True)
clinical_df = pd.concat([clinical_df,stage,histo,primary_site],axis=1)

# I am defining one-year survival as "Is alive at follow up" or "died after 12 months"
clinical_df['OneYearSurvival'] = np.where(clinical_df['Survival.Months'] > 11,1,0)
clinical_df.loc[clinical_df['Outcome'] == 1, 'OneYearSurvival'] = 1

clinical_df.info()
clinical_df.to_csv('output.csv')

# Feature selection
# Total number of features: 9
# two dependent variables: Survival (alive or dead) and months
# Select 3 best 

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
k = 3
X = clinical_df.iloc[:,3:-2] # features
Y = clinical_df.iloc[:,-1] # survival

best_features = SelectKBest(score_func=chi2, k=k)
fit=best_features.fit(X,Y)

df_scores = pd.DataFrame(fit.scores_)
df_columns = pd.DataFrame(X.columns)
features_scores= pd.concat([df_columns, df_scores], axis=1)
features_scores.columns= ['Features', 'Score']
features_scores.sort_values(by=['Score'])
features_scores.to_csv('features_scores.csv')

# From the feature outputs, we find that Num.Primaries, Radiation, Grade are most predictive of the continuous variables
clinical_df.drop(columns=['Age','Num.Mutations'],inplace = True)
X_new = clinical_df.iloc[:,3:-2] # features

# Prepare training/testing split
X_train, X_test, y_train, y_test = train_test_split(X_new,Y,test_size=0.4,random_state=1) # default split of 0.6/0.4

# Build Logistic Regression model
logreg= LogisticRegression()
logreg.fit(X_train,y_train)

# run prediction
y_pred=logreg.predict(X_test)

from sklearn import metrics
from sklearn.metrics import classification_report
print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))
print('Recall: ',metrics.recall_score(y_test, y_pred, zero_division=1))
print('Precision:',metrics.precision_score(y_test, y_pred, zero_division=1))
print('CL Report:',metrics.classification_report(y_test, y_pred, zero_division=1))

# Plot ROC curve
y_pred_proba= logreg.predict_proba(X_test)[::,1]
false_positive_rate, true_positive_rate, _ = metrics.roc_curve(y_test, y_pred_proba)
auc= metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(false_positive_rate, true_positive_rate,label="AUC="+str(auc))
plt.title('ROC Curve')
plt.ylabel('True Positive Rate')
plt.xlabel('false Positive Rate')
plt.legend(loc=4)
plt.savefig('roc.png')