import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df5 = pd.read_csv('measurement_2022-03-26_18-48-51_1.0_0.9995_0.1_0.001_0.8_50_0_14.csv')
df6 = pd.read_csv('measurement_2022-03-26_16-26-48_1.0_0.9995_0.1_0.001_0.6_50_0_14.csv')
df7 = pd.read_csv('measurement_2022-03-26_16-25-26_1.0_0.9995_0.1_0.001_0.7_50_0_14.csv')
df8 = pd.read_csv('measurement_2022-03-26_16-24-25_1.0_0.9995_0.1_0.001_0.8_50_0_14.csv')
df9 = pd.read_csv('measurement_2022-03-26_16-23-24_1.0_0.9995_0.1_0.001_0.9_50_0_14.csv')

df5.drop(columns=df5.columns[0], axis=0, inplace=True)
df6.drop(columns=df6.columns[0], axis=0, inplace=True)
df7.drop(columns=df7.columns[0], axis=0, inplace=True)
df8.drop(columns=df8.columns[0], axis=0, inplace=True)
df9.drop(columns=df9.columns[0], axis=0, inplace=True)
#print(df5,df6,df7,df8,df9)

df5.columns = ['coins', 'steps per episode/survival rate', 'total reward']
df6.columns = ['coins', 'steps per episode/survival rate', 'total reward']
df7.columns = ['coins', 'steps per episode/survival rate', 'total reward']
df8.columns = ['coins', 'steps per episode/survival rate', 'total reward']
df9.columns = ['coins', 'steps per episode/survival rate', 'total reward']

n=500
df5['rollingmean:coins'] = df5['coins'].rolling(n).mean()
df5['rollingmean:survival'] = df5['steps per episode/survival rate'].rolling(n).mean()
df5['rollingmean:reward'] = df5['coins'].rolling(n).mean()

df6['rollingmean:coins'] = df6['coins'].rolling(n).mean()
df6['rollingmean:survival'] = df6['steps per episode/survival rate'].rolling(n).mean()
df6['rollingmean:reward'] = df6['coins'].rolling(n).mean()

df7['rollingmean:coins'] = df7['coins'].rolling(n).mean()
df7['rollingmean:survival'] = df7['steps per episode/survival rate'].rolling(n).mean()
df7['rollingmean:reward'] = df7['coins'].rolling(n).mean()

df8['rollingmean:coins'] = df8['coins'].rolling(n).mean()
df8['rollingmean:survival'] = df8['steps per episode/survival rate'].rolling(n).mean()
df8['rollingmean:reward'] = df8['coins'].rolling(n).mean()

df9['rollingmean:coins'] = df9['coins'].rolling(n).mean()
df9['rollingmean:survival'] = df9['steps per episode/survival rate'].rolling(n).mean()
df9['rollingmean:reward'] = df9['coins'].rolling(n).mean()

x = np.array([i for i in range(9999)])

fig, ax = plt.subplots(1, 3,figsize=(20,5))


ax[0].plot(x,np.array(df5["rollingmean:coins"]))
ax[0].plot(x,np.array(df6["rollingmean:coins"]))
ax[0].plot(x,np.array(df7["rollingmean:coins"]))
ax[0].plot(x,np.array(df8["rollingmean:coins"]))
ax[0].plot(x,np.array(df9["rollingmean:coins"]))

ax[1].plot(x,np.array(df5["rollingmean:survival"]))
ax[1].plot(x,np.array(df5["rollingmean:survival"]))
ax[1].plot(x,np.array(df7["rollingmean:survival"]))
ax[1].plot(x,np.array(df8["rollingmean:survival"]))
ax[1].plot(x,np.array(df9["rollingmean:survival"]))

ax[2].plot(x,np.array(df5["rollingmean:reward"]))
ax[2].plot(x,np.array(df6["rollingmean:reward"]))
ax[2].plot(x,np.array(df7["rollingmean:reward"]))
ax[2].plot(x,np.array(df8["rollingmean:reward"]))
ax[2].plot(x,np.array(df9["rollingmean:reward"]))