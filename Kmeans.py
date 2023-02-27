

import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


import pandas as pd
data = pd.read_csv (r'.\Dataset.csv')
data = data.fillna(0) 
print (data)

def KMeans_N(KK):
   
    model = KMeans(n_clusters=KK)
    model.fit(data)
    
    centers = model.cluster_centers_
   
    result = model.predict(data)
   
    CH_index = metrics.calinski_harabasz_score(data, result)
    return CH_index

CH_index_result = []
plt.xlabel("N_Cluster")
plt.ylabel("calinski_harabasz_score") 
plt.title('Chose The Right Cluster Number')

for ii in range(2,10):
    temp_list = []
    temp = KMeans_N(ii)
    temp_list.append(ii)
    temp_list.append(temp)
    CH_index_result.append(temp_list)
    plt.scatter(ii, temp)

plt.show()

k = 5


model = KMeans(n_clusters=k)
model.fit(data)

centers = model.cluster_centers_

result = model.predict(data)
np.savetxt('./Kmeans/Kmeans_result.txt',result)


    