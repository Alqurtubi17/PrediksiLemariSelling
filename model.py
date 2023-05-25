import pandas as pd 
import numpy as np 
data = pd.read_csv('C:/Users/Hp/Downloads/tokopediaa.csv')
data.head()
data.info()
#menentukan variael independen 
x = data.drop(['produk', 'gambar', 'produk_laku', 'bahan'], axis = 1)
x.head()
#menentukan variabel dependen 
y = data['produk_laku']
y.head()
#import package 
from sklearn.model_selection import train_test_split
#membagi data 
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.20)

from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier (n_neighbors=2)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
y_pred
knn.predict_proba(x_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred)) 
import pickle
pickle.dump(knn, open('model_lemari.pkl', 'wb')) 



