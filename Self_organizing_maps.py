import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('Credit_Card_Applications.csv')
#splitting dataset
X=data.iloc[:,:-1].values
y=data.iloc[:,-1].values
#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
X=sc.fit_transform(X)

#training the som
from minisom import MiniSom
som=MiniSom(10,10,15)
som.random_weights_init(X)
som.train_random(X,100)

from pylab import bone,pcolor,colorbar,plot,show
bone()
pcolor(som.distance_map().T)
colorbar()
marker=['o','s']
color=['r','g']
for i,x in enumerate(X):
    w=som.winner(x)
    plot(w[0]+0.5,w[1]+0.5,marker[y[i]],markeredgecolor=color[y[i]],markerfacecolor='None',markersize=10,markeredgewidth=2    )
show()
mappings= som.win_map(X)
frauds=np.concatenate((mappings[(8,1)],mappings[(6,8)]),axis=0)

from keras.models import Sequential
from keras.layers import Dense


customer=data.iloc[:,1:].values
is_fraud=np.zeros(len(data))
for i in range(len(data)):
    if data.iloc[i,0] in frauds:
        is_fraud[i]=1
    
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
ss.fit_transform(customer)
model=Sequential()
model.add(Dense(units=2,activation='relu',kernel_initializer='uniform',input_dim=15))
model.add(Dense(units=1,activation='sigmoid',kernel_initializer='uniform'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(customer,is_fraud,batch_size=1,epochs=2) 

y_pred=model.predict(customer)








