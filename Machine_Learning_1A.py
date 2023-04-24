import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import keras.backend as K
from keras.regularizers import l2, l1


df=pd.read_csv('dataset-HAR-PUC-Rio.csv', sep=';', low_memory=False)


df["class"]=df["class"].replace({"sitting-down":1, "standing-up":2,"standing":3,"walking":4, "sitting":5})
df['gender']=df['gender'].astype('category')
df['gender_new']=df['gender'].cat.codes
OneHotEncoder=OneHotEncoder()
enc_data=pd.DataFrame(OneHotEncoder.fit_transform(df[['gender_new']]).toarray())
new_df=df.join(enc_data)

new_df['z4']=pd.to_numeric(df['z4'], errors='coerce')
columns_to_mod=['how_tall_in_meters','body_mass_index']
for cl in columns_to_mod:
    new_df[cl]=new_df[cl].str.replace(',','.')
for cl in columns_to_mod:
    new_df[cl]=new_df[cl].astype(float)



x=new_df.drop(columns=['class','gender'])
y=new_df['class']

x_numerical=x.drop(columns=['user'])
x_numerical.columns = x_numerical.columns.astype(str)

#Normalization or Min-Max Scaling 

mms=MinMaxScaler()
x_numerical_scaled=mms.fit_transform(x_numerical)
x_numerical_scaled_df=pd.DataFrame(x_numerical_scaled,columns=x_numerical.columns)


#Standardization or Z-Score 

stds=StandardScaler()
x_numerical_scaled_s=stds.fit_transform(x_numerical)
x_numerical_scaled_s_df=pd.DataFrame(x_numerical_scaled_s,columns=x_numerical.columns)


kfold=StratifiedKFold(n_splits=5, random_state=1, shuffle=True )

#early stopping
early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=50, verbose=1,restore_best_weights=True)
   
learning_rates = [0.001, 0.001, 0.05, 0.1]
momentum_values = [0.2, 0.6, 0.6, 0.6]


opt_lr, opt_momentum= None, None
opt_accuracy = 0

hidden_size=1
output_size=5
input_size=19

# Initialize arrays to store loss, accuracy, cross entropy and mse for each fold

scores=[]
history=[]
acc=[]
mse=[]
ce=[]
train_losses=[]
val_losses=[]

for h in learning_rates:
    for m in momentum_values:
            # Iterate over each fold, build and fit the neural network model and evaluate the model  
            for i, (train_index, test_index) in enumerate(kfold.split(x_numerical_scaled_df, y)):
            # Split the data into training and test sets
            x_train, x_test = x_numerical_scaled_df[train_index], x_numerical_scaled_df[test_index]
            y_train, y_test = y[train_index], y[test_index]

    
    
            model=Sequential()
            model.add(Dense(5, input_shape=(19,), activation='relu'))
            model.add(Dense(5, activation='softmax'))
            optimizer = Adam(lr=h, beta_1=m)
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_crossentropy','mean_squared_error','accuracy', ])
            history=model.fit(x_train, y_train, epochs=150, batch_size=30, verbose=2)
            train_losses.append(history.history['loss'])
            val_losses.append(history.history['val_loss'])
            loss, score= model.evaluate(x_test, y_test, verbose=0)
            categorical_crossentropy, mean_squared_error,accuracy = score
            ce.append(categorical_crossentropy)
            mse.append(mean_squared_error)
            acc.append(accuracy)
            
            accuracy = np.mean(acc)
            if accuracy > opt_accuracy:
                    opt_accuracy = accuracy
                    opt_lr = h
                    opt_momentum = m
                    
                    
    
    
mean_ce = np.mean(ce, axis=0)
mean_mse=np.mean(mse,axis=0)
mean_accuracy=np.mean(acc,axis=0)
mean_train_losses = np.mean(train_losses, axis=0)
mean_val_losses = np.mean(val_losses, axis=0)
print("cross-entropy: %.5f" % mean_ce)
print("Mean squared error: %.5f" % mean_mse)
print("Accuracy: %.5f%%" % (mean_accuracy * 100))

plt.figure(figsize=(7,5))

plt.plot(range(1, len(mean_train_losses)+2), mean_train_losses, label='Training')
plt.plot(range(1, len(mean_val_losses)+2), mean_val_losses, label='Validation')
plt.show()


print(f"Optimal Learning Rate: {opt_lr}")
print(f"Optimal Momentum: {opt_momentum}")
