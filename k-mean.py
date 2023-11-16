# Explanation see: https://www.w3schools.com/python/python_ml_k-means.asp


import matplotlib.pyplot as plt

#x = [4, 5, 10, 4, 3, 11, 14 , 6, 10, 12]
#y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
x = [1, 3, 5, 7, 9, 11, 13 , 15, 17, 1, 3, 5, 7, 9, 11, 13 , 15, 17]
y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3 ]

#x = [0,4,0,-5]
#y = [-6,4,0,2]
n= 19
#n = 11
#n = 5
plt.scatter(x, y)
plt.show()

from sklearn.cluster import KMeans

data = list(zip(x, y))
inertias = []

for i in range(1,n):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)

plt.plot(range(1,n), inertias, marker='o')
plt.title('Elbow method (Intiutive method)')
plt.xlabel('Number of clusters')
plt.ylabel('Cost (Inerta)')
plt.show()

kmeans = KMeans(n_clusters=2)
kmeans.fit(data)
centroids  = kmeans.cluster_centers_

plt.scatter(x, y, c=kmeans.labels_)


xs = [x[0] for x in centroids]
ys = [x[1] for x in centroids]
plt.scatter(xs, ys)
plt.scatter(x, y, c=kmeans.labels_)

plt.show()

print(centroids)