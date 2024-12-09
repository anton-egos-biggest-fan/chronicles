import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

csv_file = 'favorites_and_the_others.csv'
df = pd.read_csv(csv_file)

lat_lon_data = df[['lat', 'lon']].values

def to_radians(degree):
    return degree * np.pi / 180

lat_lon_data = to_radians(lat_lon_data)

# ceci ou quelque chose comme ça
# dbscan = DBSCAN().fit(lat_lon_data)
df['cluster'] = dbscan.fit_predict(lat_lon_data)

df.to_csv("secret_list.csv", index=False)

plt.figure(figsize=(10, 6))
for cluster in df['cluster'].unique():
    cluster_data = df[df['cluster'] == cluster]
    plt.scatter(cluster_data['lon'], cluster_data['lat'], label=f'Cluster {cluster}')

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.show()
