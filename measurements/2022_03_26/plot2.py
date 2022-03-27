import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

files = [
  ['measurement_2022-03-26_16-20-39_1.0_0.9995_0.1_0.0001_0.6_50_0_14.csv', 'a=1e-4, g=0.6'],
  ['measurement_2022-03-26_16-39-07_1.0_0.9995_0.1_0.0001_0.5_50_0_14.csv', 'a=1e-4, g=0.5'],
  ['measurement_2022-03-26_16-58-46_1.0_0.9995_0.1_0.0001_0.7_50_0_14.csv', 'a=1e-4, g=0.7'],
  ['measurement_2022-03-26_17-38-48_1.0_0.9995_0.1_0.0001_0.8_50_0_14.csv', 'a=1e-4, g=0.8'],
  ['measurement_2022-03-26_18-43-51_1.0_0.9995_0.1_0.0001_0.9_50_0_14.csv', 'a=1e-4, g=0.9']
  #['measurement_2022-03-26_19-08-05_1.0_0.9995_0.1_0.0005_0.9_50_0_14.csv', 'a=5e-4, g=0.9'],
  #['measurement_2022-03-26_19-42-28_1.0_0.9995_0.1_0.0005_0.8_50_0_14.csv', 'a=5e-4, g=0.8'],
  #['measurement_2022-03-26_20-02-56_1.0_0.9995_0.1_0.0005_0.7_50_0_14.csv', 'a=5e-4, g=0.7'],
  #['measurement_2022-03-27_09-21-37_1.0_0.9995_0.1_0.0005_0.6_50_0_14.csv', 'a=5e-4, g=0.6'],
  #['measurement_2022-03-27_10-18-34_1.0_0.9995_0.1_0.0005_0.5_50_0_14.csv', 'a=5e-4, g=0.5'],
  #['measurement_2022-03-26_16-23-24_1.0_0.9995_0.1_0.001_0.9_50_0_14.csv', 'a=1e-3, g=0.9'],
  #['measurement_2022-03-26_16-24-25_1.0_0.9995_0.1_0.001_0.8_50_0_14.csv', 'a=1e-3, g=0.8'],
  #['measurement_2022-03-26_16-25-26_1.0_0.9995_0.1_0.001_0.7_50_0_14.csv', 'a=1e-3, g=0.7'],
  #['measurement_2022-03-26_16-26-48_1.0_0.9995_0.1_0.001_0.6_50_0_14.csv', 'a=1e-3, g=0.6'],
]

num_files = len(files)
dataframes = []
n = 1000
num_episodes = 10000
grayscale = False

for i in range(num_files):
  dataframes.append(pd.read_csv(files[i][0]))

for d in dataframes:
  d.drop(columns=d.columns[0], axis=0, inplace=True)
  d.columns = ['coins', 'steps per episode/survival rate', 'total reward']
  d['rollingmean:coins'] = d['coins'].rolling(n).mean()
  d['rollingmean:survival'] = d['steps per episode/survival rate'].rolling(n).mean()
  d['rollingmean:reward'] = d['total reward'].rolling(n).mean()

x = np.arange(num_episodes-1)

fig, ax = plt.subplots(1, 3,figsize=(50,10))

for i in range(num_files):
  if grayscale:
    c = (i/num_files)*0.9
    ax[0].plot(x,np.array(dataframes[i]["rollingmean:coins"]), label=files[i][1], color=(c,c,c))
    ax[1].plot(x,np.array(dataframes[i]["rollingmean:survival"]), label=files[i][1], color=(c,c,c))
    ax[2].plot(x,np.array(dataframes[i]["rollingmean:reward"]), label=files[i][1], color=(c,c,c))
  else:
    ax[0].plot(x,np.array(dataframes[i]["rollingmean:coins"]), label=files[i][1])
    ax[1].plot(x,np.array(dataframes[i]["rollingmean:survival"]), label=files[i][1])
    ax[2].plot(x,np.array(dataframes[i]["rollingmean:reward"]), label=files[i][1])

ax[0].legend()
ax[1].legend()
ax[2].legend()