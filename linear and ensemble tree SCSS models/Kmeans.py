
#import libraries
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd

#load data
data = pd.read_csv (r'.\Dataset #4.csv')
data = data.fillna(0) 
print (data)

# K means clustering
def KMeans_N(KK):
   
    model = KMeans(n_clusters=KK)
    model.fit(data)
    
    centers = model.cluster_centers_
   
    result = model.predict(data)
   
    CH_index = metrics.calinski_harabasz_score(data, result)
    return CH_index

#visualise optimum number of cluster:
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

k = 8

#run model
model = KMeans(n_clusters=k)
model.fit(data)

centers = model.cluster_centers_

result = model.predict(data)

#save the output
np.savetxt('./Kmeans/Kmeans_result.txt',result)


    