#!/usr/bin/env python
# coding: utf-8

# # CLUSTER ANALYSIS 

# # Market segmentation example

# # Import the relevant libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Set the styles to Seaborn
sns.set()
# Import the KMeans module so we can perform k-means clustering with sklearn
from sklearn.cluster import KMeans


# # Load the data

# In[3]:


data = pd.read_csv (r'C:\Users\rawad\OneDrive\Desktop\aws Restart course\Udemy Data Science Course\exercise\3.12.+Example.csv')


# In[4]:


data 


# In[6]:


data.shape

GENERAL NOTE : In the context of cluster analysis, customer satisfaction and customer loyalty can be used as variables to group customers into different clusters based on their similarities or differences.

For example, you could use customer satisfaction and customer loyalty as two variables in a cluster analysis to identify groups of customers who have similar levels of satisfaction and loyalty. This could help you target your marketing efforts and retention strategies more effectively, by tailoring your approach to the needs and preferences of each cluster.

In this case, you would need to standardize the variables (i.e., customer satisfaction and customer loyalty) to ensure that they are on the same scale and have equal weighting in the analysis. You would then run the cluster analysis using an appropriate algorithm (e.g., k-means clustering) to group customers based on their similarity in these variables.

Once you have identified the clusters, you can then analyze the characteristics and behaviors of each group to gain insights into their needs, preferences, and behaviors. This can help you develop targeted marketing campaigns, retention strategies, and other initiatives to improve customer satisfaction and loyalty.
# ## Plot the data

# # Create a preliminary plot to see if you can spot something

# In[7]:


# We are creating a scatter plot of the two variables
plt.scatter(data['Satisfaction'],data['Loyalty'])
# Name your axes 
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')

Based on the scatter plot of customer satisfaction and loyalty data,if we divide into four distinct quadrants we notice that The first quadrant (Square 1) contained data points with low satisfaction values and low loyalty values, indicating customers who may be dissatisfied and unlikely to exhibit loyalty behaviors. The second quadrant (Square 2) contained data points with low-moderate satisfaction values and low loyalty values, indicating customers who may be moderately satisfied but still unlikely to exhibit loyalty behaviors. The third quadrant (Square 3) contained data points with high satisfaction values and moderate loyalty values, indicating customers who are likely to be satisfied but may not yet exhibit strong loyalty behaviors. Finally, the fourth quadrant (Square 4) contained data points with high satisfaction values and high loyalty values, indicating customers who are both satisfied and loyal.OVERALL from first look we dont know much about the data , so we can move forward in our analysis using the so called KMEANS THEORY NOTE ABOUT Kmean : K-means clustering is a popular unsupervised machine learning algorithm that can group similar data points together based on their characteristics.

In this case, K-means clustering can be used to group customers with similar satisfaction and loyalty scores together into distinct clusters. This can help businesses identify patterns and relationships in the data that may not be immediately apparent from a scatter plot.

By performing K-means clustering on this data, businesses can identify distinct customer segments based on their satisfaction and loyalty scores. These customer segments can then be targeted with customized marketing strategies or other interventions aimed at improving satisfaction and loyalty.
# ## Select the features

# In[9]:


# we create new data called x 
# Select both features by creating a copy of the data variable
x = data.copy()


# ## Clustering using kmean 
# Create an object (which we would call kmeans)
# The number in the brackets is K, or the number of clusters we are aiming for
kmeans = KMeans(2)
# Fit the data
kmeans.fit(x)
this code gave us a warning of changing the sklearn librerary ,  to address the warnings you can add additional parameters to the KMeans function. Here's an updated version of the code:
# In[13]:


import os
import numpy as np
from sklearn.cluster import KMeans

# Set environment variable to avoid memory leak warning on Windows
import os
os.environ['OMP_NUM_THREADS'] = '1'


# Create an object (which we would call kmeans)
# The number in the brackets is K, or the number of clusters we are aiming for
kmeans = KMeans(n_clusters=2, n_init=1)

# Fit the data
kmeans.fit(x)

The n_init parameter is set to 1 to avoid the warning about the default value changing in future versions of sklearn. Additionally, the OMP_NUM_THREADS environment variable is set to 1 to address the memory leak warning on Windows machines using MKL.
# # Clustering results

# In[14]:


# Create a copy of the input data
clusters = x.copy()
# Take note of the predicted clusters 
clusters['cluster_pred']=kmeans.fit_predict(x)


# In[15]:


# Plot the data using the longitude and the latitude
# c (color) is an argument which could be coded with a variable 
# The variable in this case has values 0,1, indicating to plt.scatter, that there are two colors (0,1)
# All points in cluster 0 will be the same colour, all points in cluster 1 - another one, etc.
# cmap is the color map. Rainbow is a nice one, but you can check others here: https://matplotlib.org/users/colormaps.html
plt.scatter(clusters['Satisfaction'],clusters['Loyalty'],c=clusters['cluster_pred'],cmap='rainbow')
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')

Based on the analysis, we can observe two clusters on the scatter plot, one representing customers with high satisfaction values above 6 and another representing customers with low satisfaction values below 6. However, the cluster analysis of customer loyalty appears to have been overlooked, which could raise concerns about the accuracy of the analysis.

To address this issue, we can perform standardization on the data, which involves scaling the values of each variable to a common range. This can help to ensure that the data is consistent and can be compared accurately.

Additionally, it is important to note that the presence of outliers or other factors may influence the relationship between customer satisfaction and loyalty, so it is important to consider these factors in any further analysis. It may also be useful to explore other clustering techniques to ensure that the analysis is robust and accurate.
# # Standardize the variables

# In[16]:


# Import a library which can do that easily
from sklearn import preprocessing
# Scale the inputs
# preprocessing.scale scales each variable (column in x) with respect to itself
# The new result is an array
x_scaled = preprocessing.scale(x)
x_scaled


# # Take advantage of the Elbow method
The elbow method is a technique used to determine the optimal number of clusters in a dataset. The idea behind this method is to calculate the sum of squared distances between data points and their assigned cluster centers for different values of K (number of clusters), and then plot these values against K. The "elbow" on the resulting plot represents the point of diminishing returns, beyond which adding more clusters does not significantly improve the quality of the clustering. The number of clusters at this "elbow" point is often chosen as the optimal number of clusters. However, it is important to note that this method is not foolproof and should be used in conjunction with other techniques to validate the optimal number of clusters.
# In[27]:





# In[23]:


get_ipython().system('pip install --upgrade numpy scikit-learn')


# In[32]:


import os
os.environ['OMP_NUM_THREADS'] = '1'

# Createa an empty list
wcss =[]

# Create all possible cluster solutions with a loop
# We have chosen to get solutions from 1 to 9 clusters; you can ammend that if you wish
for i in range(1,10):
    # Clsuter solution with i clusters
    kmeans = KMeans(i)
    # Fit the STANDARDIZED data
    kmeans.fit(x_scaled)
    # Append the WCSS for the iteration
    wcss.append(kmeans.inertia_)
    
# Check the result
wcss


# In[34]:


wcss 


# In[35]:


# Plot the number of clusters vs WCSS
plt.plot(range(1,10),wcss)
# Name your axes
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')

In this scenario, we conducted a loyalty and satisfaction level analysis using KMeans clustering. We started by standardizing the data and then using the elbow method to determine the optimal number of clusters for the analysis. The elbow method involved running KMeans clustering with a range of cluster solutions from 1 to 9 and calculating the within-cluster sum of squares (WCSS) for each solution. We then plotted the number of clusters against the corresponding WCSS values and looked for an "elbow point" in the plot, where the decrease in WCSS slows down significantly.

Based on the elbow plot, we observed that the optimal number of clusters for the loyalty and satisfaction level analysis is 3. This means that the data can be divided into 3 distinct clusters, each with their own loyalty and satisfaction levels. We can now use this information to further analyze the data and gain insights into customer behavior and preferences.

It's worth noting that the elbow method is a heuristic and may not always provide a clear elbow point. In such cases, other methods, such as the silhouette method, can be used to determine the optimal number of clusters. Additionally, clustering results should always be interpreted and validated with caution, as they are dependent on the quality of the data and the chosen clustering algorithm.
# ## Explore clustering solutions and select the number of clusters

# To explore clustering solutions and select the number of clusters, we can use the elbow method. The elbow method involves plotting the number of clusters against the within-cluster sum of squares (WCSS) and identifying the "elbow" point where the rate of decrease in WCSS slows down. This point represents the optimal number of clusters.
# 
# We have already calculated the WCSS for different numbers of clusters using KMeans clustering. We can now plot the number of clusters against the WCSS to visualize the elbow point.

# In[45]:


# Fiddle with K (the number of clusters)
kmeans_new = KMeans(2)
# Fit the data
kmeans_new.fit(x_scaled)
# Create a new data frame with the predicted clusters
clusters_new = x.copy()
clusters_new['cluster_pred'] = kmeans_new.fit_predict(x_scaled)


# In[41]:


# Check if everything seems right
clusters_new


# In[46]:


# Plot
plt.scatter(clusters_new['Satisfaction'],clusters_new['Loyalty'],c=clusters_new['cluster_pred'],cmap='rainbow')
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')

Based on the loyalty and satisfaction level analysis, we standardized the data and plotted the sum of squared distances (WCSS) for different numbers of clusters (K) to determine the optimal number of clusters. The elbow plot showed that there were three points of slowing down at K=3, K=4, and K=5.

We decided to proceed with K-means clustering to divide the data into 2 clusters. We labeled one of the clusters as "Alienated", which included individuals with a loyalty and satisfaction level below the mean value of 6.4, and the other two clusters as "The Everything Else Cluster".

# In[ ]:


## lets try with 3 clusters 


# In[47]:


# Fiddle with K (the number of clusters)
kmeans_new = KMeans(3)
# Fit the data
kmeans_new.fit(x_scaled)
# Create a new data frame with the predicted clusters
clusters_new = x.copy()
clusters_new['cluster_pred'] = kmeans_new.fit_predict(x_scaled)


# In[48]:


# Plot
plt.scatter(clusters_new['Satisfaction'],clusters_new['Loyalty'],c=clusters_new['cluster_pred'],cmap='rainbow')
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')

ae we can see that the alientated cluster is still holding while the everything cluster is divided into 2 clusters group  we call the new clusters is (# supporters) and (# everything else cluster) 
# In[49]:


# lets check out 4 cluster solution 


# In[50]:


# Fiddle with K (the number of clusters)
kmeans_new = KMeans(4)
# Fit the data
kmeans_new.fit(x_scaled)
# Create a new data frame with the predicted clusters
clusters_new = x.copy()
clusters_new['cluster_pred'] = kmeans_new.fit_predict(x_scaled)


# In[51]:


# Plot
plt.scatter(clusters_new['Satisfaction'],clusters_new['Loyalty'],c=clusters_new['cluster_pred'],cmap='rainbow')
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')

After exploring different clustering solutions and analyzing the elbow plot, we decided to proceed with K=4 as it provided the best clustering solution for our loyalty and satisfaction level analysis. We labeled the two categories divided by the mean of the analysis at 6.4 as "Alienated" and "Everything else". The elbow plot showed three points of slowing down at K=3, K=4, and K=5. We decided to investigate further by running K-means clustering for each of these values of K.

The results showed that K=4 provided the best clustering solution. We labeled the four resulting clusters as "Alienated", "Supporters", "Fans", and "Roumers". The first two clusters represented customers who had low loyalty levels, while the latter two clusters represented customers with higher loyalty levels. The "Fans" cluster represented customers who were highly satisfied and loyal, while the "Roumers" cluster represented customers who were predominantly satisfied but not loyal.

Overall, this clustering solution provided a clear and meaningful segmentation of our customers based on their loyalty and satisfaction levels. However, it's important to note that the choice of the optimal number of clusters can vary depending on the specific dataset and problem at hand. Therefore, it's always a good practice to explore multiple clustering solutions and evaluate their effectiveness in order to choose the best one for the given problem.




# In[ ]:




