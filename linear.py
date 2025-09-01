import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler


train_data = pd.read_csv("/Users/sharingan/Desktop/Skills/tensorflow/train.csv")
test_data = pd.read_csv("/Users/sharingan/Desktop/Skills/tensorflow/test.csv")


def preprocess(df, is_train=True):
    df = df.copy()
    
    
    if 'ADDRESS' in df.columns:
        df.drop(['ADDRESS'], axis=1, inplace=True)
    
   
    df['BHK_OR_RK'] = df['BHK_OR_RK'].map({'BHK': 0, 'RK': 1})
    
   
    df['POSTED_BY'] = df['POSTED_BY'].map({'Owner': 0, 'Dealer': 1})
    
    
    for col in df.columns:
        if df[col].isna().sum() > 0:
            if df[col].dtype == 'float64' or df[col].dtype == 'int64':
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(0)
    
    
    if is_train:
        X = df.drop('TARGET(PRICE_IN_LACS)', axis=1)
        y = df['TARGET(PRICE_IN_LACS)']
        return X, y
    else:
        return df


X_train, Y_train = preprocess(train_data, is_train=True)
X_test = preprocess(test_data, is_train=False)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = tf.keras.Sequential([
    tf.keras.Input(shape=(X_train_scaled.shape[1],)),  
    tf.keras.layers.Dense(1, kernel_initializer='zeros')  
])


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), 
              loss='mse', 
              metrics=['mae'])


history = model.fit(X_train_scaled, Y_train, epochs=50, batch_size=32, verbose=1)


Y_pred = model.predict(X_test_scaled)
print("Predictions (first 5):", Y_pred[:5])


plt.scatter(X_test_scaled[:, 0], Y_pred)
plt.xlabel("First Feature (scaled)")
plt.ylabel("Predicted Target")
plt.title("Predicted Target vs First Feature")
plt.show()
