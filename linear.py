import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

train_data = pd.read_csv("/Users/sharingan/Desktop/Tensorflow/train.csv")
test_data = pd.read_csv("/Users/sharingan/Desktop/Tensorflow/test.csv")


X_train = train_data.drop('TARGET(PRICE_IN_LACS)',axis =1 )
Y_train = train_data['TARGET(PRICE_IN_LACS)']

x_test = test_data.values

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(X_train.shape[1],))
])

model.compile(optimizer ='sgd', loss = 'mse', metrics = ['mae'])

tens_model = model.fit(X_train,Y_train , epochs = 5 , batch_size = 3, verbose = 1)

loss, mae = model.evaluate(x_test)
print(f"Test Loss (MSE): {loss:.4f}, Test MAE: {mae:.4f}")

Y_pred = model.predict(x_test)
print("Predictions:", Y_pred[:5])  

plt.scatter(x_test, Y_pred)
plt.xlabel("Actual Target")
plt.ylabel("Predicted Target")
plt.title("Actual vs Predicted")
plt.plot([x_test.min(), x_test.max()], [x_test.min(), x_test.max()], 'r--')
plt.show()