import pandas as pd
import numpy as np

from sklearn import preprocessing


missing_values = ["n/a", "na", "--", " ?","?"]

data = pd.read_csv('data/dataset.txt', na_values=missing_values, header=None)

#print(data.tail)

from sklearn.neighbors import KNeighborsClassifier

#use 5 nearest neighbor (k = 5) and Euclidean distance to implement k-NN classifier.
def KNN(X_train, y_train, X_valid, y_valid):
    knn = KNeighborsClassifier(n_neighbors=5,metric='euclidean')
    knn.fit(X_train,y_train)
    result = knn.predict(X_valid)
   # print("KNN Results")
   # print(result)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_valid,result)
    print("Confusion matrix")
    print(cm)
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_valid, result)
    print("Accuracy")
    print(accuracy)
    #TODO: RETURN precision, and recall
    return (accuracy)


# Use random half of the dataset for training and other half for validation by preserving the distribution of the classes in the original dataset.
#1) Feed the original dataset without any dimensionality reduction as input to k-NN.

from sklearn.model_selection import train_test_split
y =data[data.columns[-1,]]
#print(y.values)

X = data.drop(data[data.columns[-1]], axis=1, inplace=False)
#print(X)

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
                        PCA(n_components=m, random_state=42))
    pca.fit(X_train, y_train)
    pca.fit(X_valid, y_valid)
    train_X= pca.transform(X_train)
    valid_X= pca.transform(X_valid)
    return train_X,valid_X

range_ay = np.arange(0.80, 1.0, 0.01)
best_acc = 0
print(
    "Feature extraction: Use PCA to reduce dimensionality to m, followed by k-NN. Try for different values of m corresponding to proportion of variance of 0.80, 0.81, 0.82, ...., 0.99")
for m in list(range_ay):
    train_X,valid_X = pca(m)
    print("Reduced to m =" + '{:f}'.format(m))
    acc = KNN(train_X,y_train,valid_X, y_valid)
    if acc > best_acc:
        best_acc = acc
        best_m = m

print("Our best accuracy = " + "{:f}".format(best_acc))
print("m was = " + "{:f}".format(best_m))

#Plot the data for m=2.
#TODO: PLOT DATA
train_X, valid_X = pca(2)
KNN(train_X, y_train, valid_X, y_valid)

# Feature Selection: Use forward selection to reduce dimensionality to m using k-NN as predictor. Train the model for each m between 1 and 57.

from sklearn.feature_selection import SequentialFeatureSelector as SFS
#from mlxtend.feature_selection import SequentialFeatureSelector as SFS

knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
feature_names = X.columns
best_acc = 0

for m in range(1,57):
    sfs = SFS(knn, n_features_to_select=m)
    sfs.fit(X, y)
    # these are the selected features
    feat_cols = feature_names[sfs.get_support().tolist()]
    # now we can work with our selected features
    #.fit(X_train.iloc[:, feat_cols], y_train)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.5, random_state=42)
    print(X_train.shape)
    print(X_train.iloc[:,feat_cols])
    print(y_train.shape)
    #acc = KNN(X_train[feat_cols], y_train, X_valid[feat_cols], y_valid)
    if acc > best_acc:
        best_acc = acc
        feat_cols = feat_cols

print("Our best accuracy = " + "{:f}".format(best_acc))
print("Features selected by forward sequential selection was: "
      f"{feat_cols}")




