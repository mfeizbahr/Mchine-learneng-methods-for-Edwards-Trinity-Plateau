import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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
Y = a['Y_value']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train the Naive Bayes classifier
gnb = GaussianNB()
gnb.fit(X_train, Y_train)

# Predict the test data
Y_pred = gnb.predict(X_test)
Y_pred_all = gnb.predict(X)
a['Y_pred_all'] = Y_pred_all[a['StateWellNumber'].index]
a.to_csv('Y_pred_all_NBC.csv', index=False)

#y_pred_proba_NBC = gnb.predict_proba(X)[::,1]


#a['y_pred_proba_NBC'] = y_pred_proba_NBC[a['StateWellNumber'].index]

# save updated a table to a new CSV file
#a.to_csv('a_with_y_pred_probwa_NBC.csv', index=False)
 
# Print the confusion matrix
print(confusion_matrix(Y_test, Y_pred))

# Calculate the confusion matrix
cm = confusion_matrix(Y_test, Y_pred)

# Plot the confusion matrix as a heatmap
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

from sklearn import metrics

# Evaluate usng accuracy, precision, recall
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred)) # overall accuracy
print("Precision:",metrics.precision_score(Y_test, Y_pred)) # predicting 0 (Sat)
print("Recall:",metrics.recall_score(Y_test, Y_pred)) # predicting 1 (unsat)