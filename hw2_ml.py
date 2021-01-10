import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


missing_values = ["n/a", "na", "--", " ?","?"]

data = pd.read_csv('data/dataset.txt', na_values=missing_values, header=None)

#print(data.tail)

from sklearn.neighbors import KNeighborsClassifier
import array

cm=[]
recall = 0
precision = 0
precisions = []
recalls = []
accs = []
feat_count = []
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

def clear_lists():
    precisions.clear()
    recalls.clear()
    accs.clear()
    feat_count.clear()


#use 5 nearest neighbor (k = 5) and Euclidean distance to implement k-NN classifier.
def KNN(X_train, y_train, X_valid, y_valid):
    global recall
    global precision
    global knn
    knn = KNeighborsClassifier(n_neighbors=5,metric='euclidean')
    knn.fit(X_train,y_train)
    result = knn.predict(X_valid)
   # print("KNN Results")
   # print(result)
    cm = confusion_matrix(y_valid,result)
    print("Confusion matrix")
    print(cm)
    recall = recall_score(y_valid, result)
    recalls.append(recall)
    precision = precision_score(y_valid, result)
    precisions.append(precision)
    accuracy = accuracy_score(y_valid, result)
    accs.append(accuracy)
    feat_count.append(X_valid.shape[1])
    print("Accuracy")
    print(accuracy)
    return (accuracy)


# Use random half of the dataset for training and other half for validation by preserving the distribution of the classes in the original dataset.
#1) Feed the original dataset without any dimensionality reduction as input to k-NN.

from sklearn.model_selection import train_test_split
y =data[data.columns[-1]]
print(y.values)

X = data.drop(data.columns[-1], axis=1, inplace=False)
print(X.tail)

X_train, X_valid, y_train, y_valid = train_test_split(X.values,y.values, test_size=0.5, random_state=42)

print("1) Feed the original dataset without any dimensionality reduction as input to k-NN.")

KNN(X_train,y_train,X_valid,y_valid)

#2) Feature extraction: Use PCA to reduce dimensionality to m, followed by k-NN. Try for different values of m corresponding to proportion of variance of 0.80, 0.81, 0.82, ...., 0.99

from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Reduce dimension to m with PCA
def pca(m):
    pca = make_pipeline(StandardScaler(),
                        PCA(n_components=m, random_state=42, svd_solver='full'))
    pca.fit(X_train, y_train)
    pca.fit(X_valid, y_valid)
    train_X= pca.transform(X_train)
    valid_X= pca.transform(X_valid)
    #print(pca.named_steps['pca'].explained_variance_ratio_)
    return train_X,valid_X

range_ay = np.arange(0.80, 1.0, 0.01)
best_acc = 0
best_m = 0
print(
    "Feature extraction: Use PCA to reduce dimensionality to m, followed by k-NN. Try for different values of m corresponding to proportion of variance of 0.80, 0.81, 0.82, ...., 0.99")
clear_lists()
for m in list(range_ay):
    train_X,valid_X = pca(m)
    print("Reduced to m =" + '{:f}'.format(m))
    acc = KNN(train_X,y_train,valid_X, y_valid)
    print(valid_X.shape[1])
    if acc > best_acc:
        best_acc = acc
        best_m = m
        feat_c = valid_X.shape[1]

print("Our best accuracy = " + "{:f}".format(best_acc))
print("m was = " + "{:f}".format(best_m))
print("Feature Count " + "{:f}".format(feat_c))

#%%
# 2) Plot the accuracy, precision, and recall of 'PCA+k-NN' and 'forward selection with k-NN' versus m (i.e. number of features in the reduced subset).
fig, ax = plt.subplots(figsize=(6,6))

ax.plot(feat_count, recalls, label='Recall')
ax.plot(feat_count, accs, label='Precision')
ax.plot(feat_count, precisions, label='accuracy')
ax.set_xlabel('Feature Count')
ax.set_ylabel('Value')
ax.legend(loc='center left')
fig.show()
f_1 = feat_count
r_1 = recalls
p_1 = precisions
a_1 = accs
#%% plot the data for m=2.
clear_lists()
train_X, valid_X = pca(2)
KNN(train_X, y_train, valid_X, y_valid)
print(precision)
print(recall)
from mlxtend.plotting import plot_decision_regions
fig_1, ax1 = plt.subplots(figsize=(6,6))
fig_1 = plot_decision_regions(train_X, y_train, clf=knn, legend=2)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title('PCA-Knn')
plt.show()

# Feature Selection: Use forward selection to reduce dimensionality to m using k-NN as predictor. Train the model for each m between 1 and 57.

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.pipeline import Pipeline
import time


knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
feature_names = X.columns
best_acc = 0
print("# Feature Selection: Use forward selection to reduce dimensionality to m using k-NN as predictor.")
X_train, X_valid, y_train, y_valid = train_test_split(X,y, test_size=0.5, random_state=42)
clear_lists()
#PLEASE BE AWARE THAT CODE RUNS ON ALL AVAILABLE CPU (N_JOBS = -1)
def sfs_knn(m):
    global precision
    global recall
    global sfs1
    sfs1 = SFS(knn,
                   k_features=m,
                   forward=True,
                   floating=False,
                   scoring='accuracy',
                   n_jobs=-1)
    # now we can work with our selected features
    #.fit(X_train.iloc[:, feat_cols], y_train)
    print("m = " + "{:f}".format(m))
    pipe = Pipeline([('sfs1', sfs1),
                              ('KNN', knn)])
    start = time.time()
    pipe.fit(X_train, y_train)
    stop = time.time()
    print(f"Training time: {stop - start}s")
    print(sfs1.k_feature_idx_)
    start = time.time()
    cc = pipe.predict(X_valid)
    acc = pipe.score(X_valid, y_valid)
    stop = time.time()
    print(f"Valid time: {stop - start}s")
    print(acc)
    accs.append(acc)
    recall = recall_score(y_valid, cc)
    recalls.append(recall)
    precision = precision_score(y_valid, cc)
    precisions.append(precision)
    col = pipe.named_steps['sfs1'].k_feature_names_
    feat_count.append(m)
    return col, acc

for m in range(1,58):
    col, acc = sfs_knn(m)
    if acc > best_acc:
        best_acc = acc
        feat_cols = col

print("Our best accuracy = " + "{:f}".format(best_acc))
print("Features selected by forward sequential selection was: "
      f"{feat_cols}")
#%% 2) Plot the accuracy, precision, and recall of 'PCA+k-NN' and 'forward selection with k-NN' versus m (i.e. number of features in the reduced subset).
fig2, ax2 = plt.subplots(figsize=(6,6))
ax2.plot(feat_count, recalls, label='Recall')
ax2.plot(feat_count, accs, label='Precision')
ax2.plot(feat_count, precisions, label='accuracy')
ax2.set_xlabel('Feature Count')
ax2.set_ylabel('Value')
ax2.legend(loc='center left')
fig2.show()
r_2=recalls
p_2= precisions
a_2 = accs
f_2 = feat_count

#%% plot the data for m=2.
clear_lists()
X_train = X_train.values
y_train = y_train.values
X_valid = X_valid.values
acc = sfs_knn(2)

sfs1.fit(X_train, y_train)
x_train_r = sfs1.transform(X_train)
X_valid_r = sfs1.transform(X_valid)
KNN(x_train_r,y_train,X_valid_r,y_valid)
print(y_valid.unique)
fig_2, ax3 = plt.subplots(figsize=(6,6))
fig_2 = plot_decision_regions(train_X, y_train, clf=knn, legend=2)
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_title('Forward Selection KNN')
plt.show()

#%% merged
fig_m, axm = plt.subplots(figsize=(6,6))
#feat_count= np.asarray(feat_count)

axm.plot(f_1, r_1, label='Recall_PCA')
axm.plot(f_1, a_1, label='Precision_PCA')
axm.plot(f_1, p_1, label='accuracy_PCA')
axm.plot(f_2, r_2, label='Recall_SFS')
axm.plot(f_2, a_2, label='Precision_SFS')
axm.plot(f_2, p_2, label='accuracy_SFS')

axm.set_xlabel('Feature Count')
axm.set_ylabel('Value')
axm.legend(loc='center left')
fig.show()