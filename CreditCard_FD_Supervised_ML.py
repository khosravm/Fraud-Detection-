#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 17:43:52 2023

@author: khosravm
"""
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,normalize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA

from keras.models import Model
from keras.layers import Input, Dense
from keras import regularizers, Sequential

RANDOM_SEED = 42
LABELS = ["Normal", "Fraud"]

# *****************************************************************************
# Read Input: credit card transactions that occurred during a period of 2 days, 
# with 492 frauds out of 284,807 transactions.
# *****************************************************************************
df = pd.read_csv("creditcard.csv")
df['Class'] = df['Class'].map({"'0'": 0, "'1'": 1})  # remove double "
# print(df.head())
# print(df.shape)  # (284807, 31)
# print(df.isnull().values.any()) # False
# *****************************************************************************
# Checking No. of records of each class (Fraud and Non-Fraud)
# *****************************************************************************
count_classes = pd.value_counts(df['Class'], sort = True)
count_classes.plot(kind = 'bar', rot=0)
plt.title("Transaction class distribution")
plt.xticks(range(2), LABELS)
plt.xlabel("Class")
plt.ylabel("Frequency")

# *****************************************************************************
#                                  Data Preparation
# *****************************************************************************
## Data for Models without rep.Learning

df1      = df.sample(frac=1) # randomize the whole dataset

data1    = df1.drop(["Time","Class"],axis=1)
data_arr = data1.values

labels     = pd.DataFrame(df1[["Class"]])
labels_arr = labels.values

X_tr,X_test,Y_tr,Y_test = train_test_split(data_arr,labels_arr,test_size=0.25)
X_tr   = normalize(X_tr)
X_test = normalize(X_test)


## Data for Models with rep.Learning

frauds = df[df.Class == 1]
normal = df[df.Class == 0]
# print("Frauds: ",frauds.shape)
# print("Normals: ",normal.shape)

# Describe Fraud and Non-Fraud Transactions
print("=============================================")
print("Frauds: ",frauds.Amount.describe())
print("=============================================")
print("Normals: ",normal.Amount.describe())
print("")

# Graphical representation
# Plotting Amount per transaction
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Amount per transaction by class')

bins = 50

ax1.hist(frauds.Amount, bins = bins)
ax1.set_title('Fraud')

ax2.hist(normal.Amount, bins = bins)
ax2.set_title('Normal')

plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')
plt.xlim((0, 20000))
plt.yscale('log')
plt.show()

# Plotting time of transaction (check for correlations)
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Time of transaction vs Amount by class')

ax1.scatter(frauds.Time, frauds.Amount)
ax1.set_title('Fraud')

ax2.scatter(normal.Time, normal.Amount)
ax2.set_title('Normal')

plt.xlabel('Time (in Seconds)')
plt.ylabel('Amount')
plt.show()

# The time does not seem to be a crucial feature in distinguishing normal vs 
# fraud cases, as a result:
data = df.drop(['Time'], axis=1)

# Scaling: The numerical amount in fraud and normal cases differ highly
data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))


# *****************************************************************************
#                   Building Autoencoders Model
# *****************************************************************************
# Data 
non_fraud = data[data['Class'] == 0] #.sample(1000)
fraud     = data[data['Class'] == 1]

df = non_fraud.append(fraud).sample(frac=1).reset_index(drop=True)
X  = df.drop(['Class'], axis = 1).values
Y  = df["Class"].values

## Create the model  **********************************************************

# Autoencoder Model
## Input layer
input_layer = Input(shape=(X.shape[1],))

## encoding part
encoded = Dense(100, activation='tanh', activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoded = Dense(50, activation='relu')(encoded)

## decoding part
decoded = Dense(50, activation='tanh')(encoded)
decoded = Dense(100, activation='tanh')(decoded)

## output layer
output_layer = Dense(X.shape[1], activation='relu')(decoded)

autoencoder = Model(input_layer, output_layer)
## Compile the model  **********************************************************
autoencoder.compile(optimizer="adadelta", loss="mse")

# *****************************************************************************                         
#   Training the Autoencoder **************************************************

# Scaling the values

x = data.drop(["Class"], axis=1).values
y = data["Class"].values

x_scale = MinMaxScaler().fit_transform(x)
x_norm, x_fraud = x_scale[y == 0], x_scale[y == 1]

autoencoder.fit(x_norm[0:2000], x_norm[0:2000], 
                batch_size = 256, epochs = 10, 
                shuffle = True, validation_split = 0.25);

# *****************************************************************************                          
#  Hidden Representation ******************************************************

hidden_representation = Sequential()
hidden_representation.add(autoencoder.layers[0])
hidden_representation.add(autoencoder.layers[1])
hidden_representation.add(autoencoder.layers[2])

# Encoder Prediction (Input representation)
norm_hid_rep  = hidden_representation.predict(x_norm[:3000])
fraud_hid_rep = hidden_representation.predict(x_fraud)

# Getting the representation Data
rep_x = np.append(norm_hid_rep, fraud_hid_rep, axis = 0)

y_n   = np.zeros(norm_hid_rep.shape[0])
y_f   = np.ones(fraud_hid_rep.shape[0])
rep_y = np.append(y_n, y_f)



# Visualize these encoded representations ************************************* 
# Visulize in 2D, to get a first glance of the classification problem
random_seed = 0
np.random.seed(random_seed)

# Dimensionality reduction: t-SNE and PCA
def tsne_plot(x, y):
    tsne = TSNE(n_components=2, random_state=random_seed) 
    
    X_tsne = tsne.fit_transform(x)

    plt.figure(figsize=(8, 8))
    plt.scatter(X_tsne[np.where(y == 0), 0], X_tsne[np.where(y == 0), 1], color='g', label='Non Fraud')
    plt.scatter(X_tsne[np.where(y == 1), 0], X_tsne[np.where(y == 1), 1], color='r', label='Fraud')

    plt.legend(loc='best')
    plt.show()
    
# def pca_plot(x, y):
#     pca = PCA(n_components=2, random_state=random_seed)
#     X_pca = pca.fit_transform(x)
    
#     plt.figure(figsize=(8, 8))
#     plt.scatter(X_pca[np.where(y == 0), 0], X_pca[np.where(y == 0), 1], color='g', label='Non Fraud')
#     plt.scatter(X_pca[np.where(y == 1), 0], X_pca[np.where(y == 1), 1], color='r', label='Fraud')

#     plt.legend(loc='best')
#     plt.show()
    
tsne_plot(rep_x,rep_y)
# pca_plot(rep_x,rep_y)

# *****************************************************************************
#               Logistic Regression Model (with Rep.Learning)
# *****************************************************************************
#Splitting Data
train_x, val_x, train_y, val_y = train_test_split(rep_x, rep_y, test_size=0.25)

LR_clf    = LogisticRegression(solver="lbfgs").fit(train_x, train_y)
LR_pred_y = LR_clf.predict(val_x)


#calculating confusion matrix for LR
cm_LR   = confusion_matrix(val_y,LR_pred_y)
disp_LR = ConfusionMatrixDisplay(confusion_matrix=cm_LR)

print("=====================================================")
print ("Classification Report: ")
print ("")
print (classification_report(val_y,LR_pred_y))

# *****************************************************************************
#                            Logistic Regression Model
# *****************************************************************************

LR1_clf         = LogisticRegression(solver="lbfgs").fit(X_tr, Y_tr)
LR1_predicted_Y = LR1_clf.predict(X_test)


#calculating confusion matrix for knn
cm_LR1   = confusion_matrix(Y_test,LR1_predicted_Y)
disp_LR1 = ConfusionMatrixDisplay(confusion_matrix=cm_LR1)

fig, (ax1, ax2)  = plt.subplots(2,1)
disp_LR.plot(ax  = ax1)
disp_LR1.plot(ax = ax2)
plt.show()

print("=====================================================")
print ("Classification Report: ")
print ("")
print (classification_report(Y_test, LR1_predicted_Y))

# *****************************************************************************
#          k_nearest_neighbours (k-NN) Classifier (with Rep.Learning)
# *****************************************************************************

knn = KNeighborsClassifier(n_neighbors=5,algorithm="kd_tree",n_jobs=-1)
knn.fit(train_x,train_y.ravel())
knn_pred_y = knn.predict(val_x)

#calculating confusion matrix for knn
cm_knn   = confusion_matrix(val_y,knn_pred_y)
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn)


print("=====================================================")
print ("Classification Report: ")
print ("")
print (classification_report(val_y,knn_pred_y))

# *****************************************************************************
#                       k_nearest_neighbours (k-NN) Classifier
# *****************************************************************************

knn1 = KNeighborsClassifier(n_neighbors=5,algorithm="kd_tree",n_jobs=-1)
knn1.fit(X_tr,Y_tr.ravel())
knn1_predicted_Y = knn1.predict(X_test)

#calculating confusion matrix for knn
cm_knn1 = confusion_matrix(Y_test,knn1_predicted_Y)
disp_knn1 = ConfusionMatrixDisplay(confusion_matrix=cm_knn1)

fig, (ax1, ax2)   = plt.subplots(2,1)
disp_knn.plot(ax  = ax1)
disp_knn1.plot(ax = ax2)
plt.show()

print("=====================================================")
print ("Classification Report: ")
print ("")
print (classification_report(Y_test,knn1_predicted_Y))

# *****************************************************************************
#                 Decision Tree Classifier (with Rep.Learning)
# *****************************************************************************

dec_tree_model = DecisionTreeClassifier()
dec_tree_model.fit(train_x,train_y)

DT_pred_y = dec_tree_model.predict(val_x)

#calculating confusion matrix for DT
cm_DT   = confusion_matrix(val_y,DT_pred_y)
disp_DT = ConfusionMatrixDisplay(confusion_matrix=cm_DT)



print("=====================================================")
print ("")
print ("Classification Report: ")
print (classification_report(val_y, DT_pred_y))

print ("")
print (f"Accuracy Score: {accuracy_score(val_y, DT_pred_y)*100:.3} %")

# *****************************************************************************
#                            Decision Tree Classifier
# *****************************************************************************

dec_tree_model1 = DecisionTreeClassifier()
dec_tree_model1.fit(X_tr, Y_tr)

DT1_predicted_Y = dec_tree_model1.predict(X_test)

#calculating confusion matrix for DT
cm_DT1   = confusion_matrix(Y_test, DT1_predicted_Y)
disp_DT1 = ConfusionMatrixDisplay(confusion_matrix=cm_DT1)

fig, (ax1, ax2)   = plt.subplots(2,1)
disp_DT.plot(ax  = ax1)
disp_DT1.plot(ax = ax2)
plt.show()

print("=====================================================")
print ("")
print ("Classification Report: ")
print (classification_report(Y_test, DT1_predicted_Y))

print ("")
print (f"Accuracy Score: {accuracy_score(Y_test, DT1_predicted_Y)*100:.3} %")


# *****************************************************************************
#                            ROC Curve 
# *****************************************************************************

# Use model to predict probability that given y value is 1
y_pred_proba_LR   = LR_clf.predict_proba(val_x)[::,1]
y_pred_proba_knn  = knn.predict_proba(val_x)[::,1]
y_pred_proba_DT   = dec_tree_model.predict_proba(val_x)[::,1]

y_pred_proba_LR1  = LR1_clf.predict_proba(X_test)[::,1]
y_pred_proba_knn1 = knn1.predict_proba(X_test)[::,1]
y_pred_proba_DT1  = dec_tree_model1.predict_proba(X_test)[::,1]

# Calculate AUC of model
lr_auc   = roc_auc_score(val_y, y_pred_proba_LR)
knn_auc  = roc_auc_score(val_y, y_pred_proba_knn)
DT_auc   = roc_auc_score(val_y, y_pred_proba_DT)

lr1_auc  = roc_auc_score(Y_test, y_pred_proba_LR1)
knn1_auc = roc_auc_score(Y_test, y_pred_proba_knn1)
DT1_auc  = roc_auc_score(Y_test, y_pred_proba_DT1)

# Summarize scores
print('Logistic: ROC AUC=%.3f' % (lr_auc))
print('Knn: ROC AUC=%.3f' % (knn_auc))
print('Decision Tree: ROC AUC=%.3f' % (DT_auc))

print('Logistic: ROC AUC=%.3f' % (lr1_auc))
print('Knn: ROC AUC=%.3f' % (knn1_auc))
print('Decision Tree: ROC AUC=%.3f' % (DT1_auc))

# calculate roc curves
lr_fpr, lr_tpr, _ = roc_curve(val_y, y_pred_proba_LR)
kn_fpr, kn_tpr, _ = roc_curve(val_y, y_pred_proba_knn)
dt_fpr, dt_tpr, _ = roc_curve(val_y, y_pred_proba_DT)

lr_fpr1, lr_tpr1, _ = roc_curve(Y_test, y_pred_proba_LR1)
kn_fpr1, kn_tpr1, _ = roc_curve(Y_test, y_pred_proba_knn1)
dt_fpr1, dt_tpr1, _ = roc_curve(Y_test, y_pred_proba_DT1)

# Plot the roc curve for the model
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic with RL')
plt.plot(kn_fpr, kn_tpr, marker='+', label='Knn with RL')
plt.plot(dt_fpr, dt_tpr, marker='*', label='Decision Tree with RL')

plt.plot(lr_fpr1, lr_tpr1, 'rh', label='Logistic')
plt.plot(kn_fpr1, kn_tpr1, 'kd', label='Knn')
plt.plot(dt_fpr1, dt_tpr1, 'bo', label='Decision Tree')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

