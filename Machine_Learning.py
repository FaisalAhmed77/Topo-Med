# Loading required Libraries


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
% matplotlib
inline

# Loading Feature Vectors

df = pd.read_excel("df_train_Betti_1.xlsx")
df = df.drop(df.columns[[0]], axis=1)

# Slicing Data
x = df.iloc[:, :100].values
y = df.iloc[:, 100].values

# #Spliting the data in 80:20 training to testing
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Applying Machine learning models from Scikit-learn

# XGBoost Classifier

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

modelx = XGBClassifier(use_label_encoder=True, eval_metric='mlogloss',n_estimators = 2500, \
                max_depth = 19,  learning_rate = 0.02,subsample = .5, colsample_bytree = 0.4, \
                       colsample_bylevel = 0.4, gamma = 1 , random_state = 0)

modelx.fit(x_train, y_train)

from sklearn import metrics

metrics.plot_roc_curve(modelx, x_test, y_test)
plt.show()

y_predx = modelx.predict(x_test)

print("XGboost model accuracy(in %):", accuracy_score(y_test, y_predx) * 100)

cm = confusion_matrix(y_test, y_predx)

print("Confusion Matrix : \n", cm)
plot_confusion_matrix(modelx, x_test, y_test, cmap='Blues')
plt.grid(False)
from sklearn.metrics import classification_report

print(classification_report(y_test, y_predx, digits=4))

# Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix

model_1 = RandomForestClassifier(n_estimators=300, criterion='entropy',
                                 min_samples_split=10, random_state=0)

# fitting the model on the train data
model_1.fit(x_train, y_train)

from sklearn import metrics

metrics.plot_roc_curve(model_1, x_test, y_test)
plt.show()

predictions = model_1.predict(x_test)
print("Random forest model accuracy(in %):", accuracy_score(y_test, predictions) * 100)

cm = confusion_matrix(y_test, predictions)

print("Confusion Matrix : \n", cm)
plot_confusion_matrix(model_1, x_test, y_test, cmap='Blues')
plt.grid(False)

from sklearn.metrics import classification_report

print(classification_report(y_test, predictions, digits=4))


#  MLP Code

# define network architecture
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense
import keras

MLP = Sequential()
MLP.add(InputLayer(input_shape=(200, ))) # input layer
MLP.add(Dense(256, activation='relu')) # hidden layer 1
MLP.add(Dense(256, activation='relu')) # hidden layer 2
MLP.add(Dense(4, activation='softmax')) # output layer

# summary
MLP.summary()
loss = keras.losses.categorical_crossentropy
MLP.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
MLP.fit(x_train, y_train, batch_size= 32, epochs=100, shuffle=True, verbose=0)
score = MLP.evaluate(x_test, y_test, verbose=0)
print('loss=', score[0])
print('accuracy=', score[1])

