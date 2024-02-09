import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the dataset
Dataset = pd.read_excel(r"C:\Users\Hp\Downloads\train.xlsx")
X = Dataset.iloc[:, :-1].values



# Define Streamlit app
st.title('KMeans Clustering Visualization')

# Display the Elbow Method plot
st.subheader('Elbow Method')
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, random_state=42)  # Set n_init explicitly
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

fig1, ax1 = plt.subplots()
ax1.plot(range(1, 11), wcss)
ax1.set_title('The Elbow Method')
ax1.set_xlabel('Number of clusters')
ax1.set_ylabel('WCSS')
st.pyplot(fig1)



# Perform KMeans clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=5, init='k-means++', n_init=10, random_state=42)  # Set n_init explicitly
y_kmeans = kmeans.fit_predict(X_scaled)

# Display the Clustering Results
st.subheader('Clustering Results (KMeans with 5 clusters)')
fig2, ax2 = plt.subplots()
scatter = ax2.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_kmeans, cmap='viridis')
ax2.set_title('Clustering Results (KMeans with 5 clusters)')
fig2.colorbar(scatter, ax=ax2)
st.pyplot(fig2)


# Define function to predict cluster for new data
def predict_cluster(new_data):
    new_data_scaled = scaler.transform(new_data.reshape(1, -1))
    cluster = kmeans.predict(new_data_scaled.reshape(1, -1))[0]
    return cluster

# Example of predicting cluster for new data
new_data=np.array([-74,	-76,-74	,-68,-76,	-71,	-72,	-55,	-61,	-78,	-78,	-75,	-70,	-62,	-76,	-78,	-70,	-62])

#new_data = np.array([-65, -60, -65, -55, -50, -60, -80, -55, -75, -75, -65, -65, -60, -58, -70, -75, -65, -75])
cluster = predict_cluster(new_data)
print(cluster)
st.subheader('Predicting Cluster for New Data')
st.write("The new data point belongs to cluster", cluster)