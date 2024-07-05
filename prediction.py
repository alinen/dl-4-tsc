import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.preprocessing
import sklearn

savename = "results/fcn/UCRArchive_20180704/C1_Axes_300/best_model.keras"
print(savename)
trained = tf.keras.models.load_model(savename)

filename = "archives/UCRArchive_2018/C1_Axes_300/C1_Axes_300_TEST.tsv"
df_test = pd.read_csv(filename, sep='\t', header=None)

y_test = df_test.values[:, 0]
x_test = df_test.drop(columns=[0])
x_test.columns = range(x_test.shape[1])

test_data = x_test.values
test_labels = y_test

# znorm
std_ = test_data.std(axis=1, keepdims=True)
std_[std_ == 0] = 1.0
test_data = (test_data - test_data.mean(axis=1, keepdims=True)) / std_

# transform the labels from integers to one hot vectors
enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
enc.fit(test_labels.reshape(-1,1))
test_labels = enc.transform(test_labels.reshape(-1,1)).toarray()

print("data: ", test_data.shape)
print("labels: ", test_labels.shape)
print("data size: ", len(test_data))

#probability_model = tf.keras.Sequential([trained, tf.keras.layers.Softmax()])
#predictions = probability_model.predict(test_data)

y_pred = trained.predict(test_data)
y_pred = np.argmax(y_pred, axis=1)

print("prediction: ", y_pred)
print("actual: ", y_test)

sum = 0
for i in range(len(y_test)):
    if y_test[i] != y_pred[i]:
        sum = sum + 1
print("num errors: ", sum)
print("error rate: ", sum/len(y_test))
