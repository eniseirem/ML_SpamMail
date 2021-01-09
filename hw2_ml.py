import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


missing_values = ["n/a", "na", "--", " ?","?"]

data = pd.read_csv('data/dataset.txt', na_values=missing_values, header=None)

#print(data.tail)

from sklearn.neighbors import KNeighborsClassifier
cm=[]
recall = 0
precision = 0
precisions = []
recalls = []
accs = []
feat_count = []

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
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_valid,result)
    print("Confusion matrix")
    print(cm)
    from sklearn.metrics import recall_score
    recall = recall_score(y_valid, result)
    recalls.append(recall)
    from sklearn.metrics import precision_score
    precision = precision_score(y_valid, result)
    precisions.append(precision)
    from sklearn.metrics import accuracy_score
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
ax.plot(recalls, feat_count, label='Recall')
ax.plot(accs, feat_count, label='Precision')
ax.plot(precisions, feat_count, label='accuracy')
ax.set_xlabel('Value')
ax.set_ylabel('feature count')
ax.legend(loc='center left')
fig.show()
#%% plot the data for m=2.

train_X, valid_X = pca(2)
KNN(train_X, y_train, valid_X, y_valid)
print(precision)
print(recall)
from mlxtend.plotting import plot_decision_regions
plot_decision_regions(train_X, y_train, clf=knn, legend=2)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('PCA-Knn')
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
precisions = []
recalls = []
accs = []
feat_count = []
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
    from sklearn.metrics import recall_score
    recall = recall_score(y_valid, cc)
    from sklearn.metrics import precision_score
    precision = precision_score(y_valid, cc)
    col = pipe.named_steps['sfs1'].k_feature_names_
    return col, acc

for m in range(1,57):
    col, acc = sfs_knn(m)
    if acc > best_acc:
        best_acc = acc
        feat_cols = col

print("Our best accuracy = " + "{:f}".format(best_acc))
print("Features selected by forward sequential selection was: "
      f"{feat_cols}")
#%% 2) Plot the accuracy, precision, and recall of 'PCA+k-NN' and 'forward selection with k-NN' versus m (i.e. number of features in the reduced subset).
fig, ax = plt.subplots(figsize=(6,6))
ax.plot(recalls, feat_count, label='Recall')
ax.plot(accs, feat_count, label='Precision')
ax.plot(precisions, feat_count, label='accuracy')
ax.set_xlabel('Value')
ax.set_ylabel('feature count')
ax.legend(loc='center left')
fig.show()


#%% plot the data for m=2.
X_train = X_train.values
y_train = y_train.values
X_valid = X_valid.values
acc = sfs_knn(2)
# print(precision)
# print(recall)
sfs1.fit(X_train, y_train)
x_train_r = sfs1.transform(X_train)
X_valid_r = sfs1.transform(X_valid)
KNN(x_train_r,y_train,X_valid_r,y_valid)
print(y_valid.unique)
plot_decision_regions(x_train_r, y_train, clf=knn, legend=2)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Forward Selection KNN')
plt.show()
