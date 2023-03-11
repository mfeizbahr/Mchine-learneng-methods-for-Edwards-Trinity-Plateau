# Load Libraries
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from itertools import chain
from matplotlib import pyplot as plt


# CHange working directory
dir = 'C:/pr2/'
os.chdir(dir)

# Read the dataset
a = pd.read_csv('FINAL.csv') # read our dataset
tex_mapping = {"Clay": 1, "Clay loam": 2, "Fine sandy loam": 3, "Loam": 4, "Loamy sand": 5, "Sandy loam": 6, "Sandy clay loam": 7, "Silt loam": 8, "Silty clay": 9, "Silty clay loam": 10  }
# Create the new tex_index column based on chtexture_texcl column and mapping
a["tex_index"] = a["chtexture_texcl"].map(tex_mapping)


features = ['TEMPERATURE_CELSIUS', 'PH', 'ALKALINITY',  'CALCIUM_DIS', 'MAGNESIUM_DIS', 'SODIUM_%',  'claytotal_l',  'tex_index' ] # INPUT DATA FEATURES
X = a[features] # DATAFRAME OF INPUT FEATURES
#SVCYR2 = a['SVCYR']  # Add SVCYR square to the dataset
#SVCYR2 = np.power(SVCYR2,2)
#X.insert(1,"SVCYR2",SVCYR2)  # Insert the data frame
Y = a['Y_value'] # ADD IT TO THE INPUT FEATURE DATAFRAME

# Split into training and testing data
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.30,
                                               random_state=10)

# instantiate the model object (using the default parameters)
logreg = LogisticRegression(C=10**9) # Large C means no regularization

# fit the model with data
logreg.fit(X_train,y_train)

# Make Predictions
y_pred=logreg.predict(X_test) # Make Predictions
yprob = logreg.predict_proba(X_test) #test output probabilities
zz = pd.DataFrame(yprob)
zz.to_csv('probs.csv')

# Get the parameters
logreg.get_params()
logreg.coef_
logreg.intercept_

# Write the data to a file
keys = list(X.columns)
keys.append('Intercept')

vals = logreg.coef_.tolist()
vals = list(chain.from_iterable(vals))
intcept = float(logreg.intercept_)
vals.append(intcept)

par_dict = dict(zip(keys,vals))
with open('pars.txt','w') as data: 
      data.write(str(par_dict))

# Create a confusion Matrix and plot it
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix # y_test is going be rows (obs), y_pred (predicted) are cols
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cnf_matrix)
disp.plot()
plt.show()

# Evaluate usng accuracy, precision, recall
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)) # overall accuracy
print("Precision:",metrics.precision_score(y_test, y_pred)) # predicting 0 (Sat)
print("Recall:",metrics.recall_score(y_test, y_pred)) # predicting 1 (unsat)

# ROC Curve
y_pred_proba = logreg.predict_proba(X_test)[::,1]
#y_pred_probwa = logreg.predict_proba(X)[::,1]


#a['y_pred_probwa'] = y_pred_probwa[a['StateWellNumber'].index]

# save updated a table to a new CSV file
#a.to_csv('a_with_y_pred_probwa.csv', index=False)

fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(round(auc,4)))
plt.legend(loc=4)
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
plt.grid()
plt.show()

X_train.hist(figsize=(20,20))
plt.show()

 

# Create histograms with density plots and x-limits for multiple variables
fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(20, 12))
colors = ['teal', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'blue']

for i, var in enumerate(features):
    row, col = divmod(i, 5)
    ax = axs[row, col]
    ax.hist(X_train[var], bins=15, density=True, stacked=True, color=colors[i % len(colors)], alpha=0.6)
    X_train[var].plot(kind='density', color=colors[i % len(colors)])
    ax.set(xlabel=var)
    ax.set_xlim(-10, 85)

plt.show()


import seaborn as sns

# Compute the correlation matrix
corr_matrix = X_train.corr()

# Plot the heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()



# Compute the correlation matrix
corr_matrix = X_train.corr()

# Set up the matplotlib figure
fig, ax = plt.subplots(figsize=(10, 8))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=.3, center=0,
            annot=True, square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()




# Calculate the correlation matrix
corr = X.corr()

# Plot the correlation matrix as a heatmap
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()
