import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from kmedoids import KMedoids
import folium
import matplotlib.cm as cm
import matplotlib.colors as colors

'''

This script will hold the helper functions I developed earlier, so I can load them into the final clustering script.



'''


#1. KMeans Clustering

def clustering_kmeans(input, k): 
    coords = np.array(input)  # list of [lat, lon] from the households df
    model = KMeans(n_clusters=k) # dynamic k 
    return model.fit_predict(input)


#2. KMedoids Pre-computed (retired)

def clustering_kmedoids(distance_matrix, k): # input here is preloaded road distance matrix
    model = KMedoids(
        n_clusters=k,
        metric='precomputed'
    )
    labels = model.fit(distance_matrix).labels_
    return labels

#3. KMedoids Manhattan
def clustering_kmedoids_manhattan(input, k):
    coords = np.array(input)
    model = KMedoids(n_clusters=k, metric='manhattan', method='pam', random_state=0)
    return model.fit_predict(coords)

#4. Kmeoids Haversine
def clustering_kmedoids_haversine(input, k):
    """
    Perform KMedoids clustering using Haversine distance (great-circle distance)
    
    Parameters:
    -----------
    input : list of [lon, lat] pairs in DEGREES
    k : int, number of clusters
    
    Returns:
    --------
    labels : array of cluster assignments
    """
    coords = np.array(input)  # [lon, lat] pairs in degrees
    
    # Haversine requires coordinates in radians and in [lat, lon] order
    coords_rad = np.radians(coords[:, [1, 0]])  # Swap to [lat, lon] and convert to radians
    
    model = KMedoids(n_clusters=k, metric='haversine', method='pam', random_state=0)
    return model.fit_predict(coords_rad)

#. Number of Households Check

def check_num_households(cluster_dict):
    # save household numbers to list
    num_households = []
    for cluster_id in cluster_dict.keys():
        # Iterate through keys and just take length
        num_households.append(len(cluster_dict[cluster_id]))
        
    return num_households


#4. Number of Voters Check

def check_num_voters(voters_dict):
    # initialize empty list to store counts
    num_voters = []
    for cluster in voters_dict.keys():
        # iterate through each clusters key and sum
        total = sum(voters_dict[cluster])
        # append back
        num_voters.append(total)
    return num_voters



#5. KMeans Iterative Cluster Function

def iterative_kmeans(households, k_min, k_max, max_households, max_voters):
    # Functions take as input user-defined upper limit thresholds of households and voters, household-level DF, and a large enough range of k
    
    # Establish range of possible k's to test
    for k in range(k_min, k_max +1 ):
    
        # Define clusters with Kmeans helper function
        
        coords = households[['lon', 'lat']].values.tolist()  # transform coordinates from Households DF to list
        
        labels = clustering_kmeans(coords, k) # Pass coordinates list as argument to pre-defined function and perform cluster
        
        # Project the clusters back onto the household set
        households['cluster'] = labels
        
        # Construct dictionary of clusters with number of households
        household_dict = households.groupby("cluster")["id"].apply(list).to_dict()

        # Construct dictionary of clusters with number of voters
        voters_dict = households.groupby("cluster")["NUM_VOTERS"].apply(list).to_dict()

        
    
        
    
    ##########
    # Checking the upper limit constraints
    ##########
        
        # set default
        # If at nay point, this default is switched to False, it will increase k by 1 and send the clustering to a new round.
        conditions_met = True
        
        
        # 1. Number of Households per cluster
    
        # Call function defined above on the household dictionary
        num_households = check_num_households(household_dict)
        
        # Define for loop to send cluster to new k, if any cluster exceeds the limit
        for household in num_households:
            if household > max_households:
                conditions_met = False
                break
            
        if not conditions_met:
            continue
                
        # 2. Number of Voters per cluster
        
        # Call function defined above to check the upper limit threshold of voters per cluster.
        num_voters = check_num_voters(voters_dict)
        
        # Send to next round of k clusters, if exceeded
        for voters in num_voters:
            if voters > max_voters:
                conditions_met = False
                break
        
        if not conditions_met:
            continue
        
        
        # At this stage, the perfect k has been found
        
        # Assign final k clusters onto households DF
        households['cluster'] = labels
        households.to_csv('households.csv', index=False)

        # Create resulting dicitonaries for both limits
        result_household_dict = households.groupby("cluster")["id"].apply(list).to_dict()
        result_voters_dict = households.groupby("cluster")["NUM_VOTERS"].apply(list).to_dict()
        
        # Return the results
        return (households, k, labels, result_household_dict, result_voters_dict)
    
    # Return None if no optimal k was found
    return None




#6. Kmedoids Iterative Clustering Function


# Very similar to KMeans function
'''
def iterative_kmedoids(households, k_min, k_max, max_households, max_voters, distance_matrix):
    
    households = households.copy()
    households["id"] = households["id"].astype(int)
    if "cluster" in households.columns:
        households = households.drop(columns=["cluster"])
    # --- Distance matrix prep ---
    df = distance_matrix.copy()

    # Index is float64 like 239.0, 574.0 → cast to int to match households["id"]
    df.index = df.index.astype(float).astype(int)

    # We don't actually need column labels for k-medoids, only the numeric matrix
    D = df.values.astype(float)
    D = np.minimum(D, D.T)  # enforce symmetry

    # Keep the ID order that corresponds to the rows of D
    IDs = df.index.to_numpy()  

    # Perform some Matrix modifications
    # Save the IDs from the matrix, so they can be matched later
    #IDs = distance_matrix.index.to_numpy()
    # Extract just the distance values
    #distance_matrix = distance_matrix.values
    # Convert to floats for easier handling
    #distance_matrix= distance_matrix.astype(float)
    # To make this a real, symmetric matrix, we take the minimum of the matrix and its transpose values
    # This derives from the caveat that road-distances from A to B are not always the same as from B to A
    # I only found this out during the process, so this is anothe assumption we have to make.
    #distance_matrix = np.minimum(distance_matrix, distance_matrix.T)    
    
    
    
    for k in range(k_min, k_max +1 ):
    
        # Pass the distance metrix into the predefined clustering function
        labels = clustering_kmedoids(D, k)
        
    
        # Put clusters and household IDs back together
        cluster_assignments = pd.DataFrame({
            "id": IDs,
            "cluster": labels
            })
        
        # Merge labels onto households
        households_clusters = households_clusters.merge(cluster_assignments, on="id")
        
        # Construct dictionary of clusters with number of households
        household_dict = households_clusters.groupby("cluster")["id"].apply(list).to_dict()

        # Construct dictionary of clusters with number of voters
        voters_dict = households.groupby("cluster")["NUM_VOTERS"].apply(list).to_dict()
        
        ##########
        # Checking the upper limit constraints
        ##########
        
        # set default
        conditions_met = True
        
        # 1. Number of Households per cluster
    
        num_households = check_num_households(household_dict)
        
        for household in num_households:
            if household > max_households:
                conditions_met = False
                break
            
        if not conditions_met:
            continue
                
        # 2. Number of Voters per cluster
        
        num_voters = check_num_voters(voters_dict)
        
        for voters in num_voters:
            if voters > max_voters:
                conditions_met = False
                break
        
        # Again, send to next k if violated
        if not conditions_met:
            continue
        
        households['cluster'] = labels
        households.to_csv('households.csv', index=False)
    
        result_household_dict = households.groupby("cluster")["id"].apply(list).to_dict()
        result_voters_dict = households.groupby("cluster")["NUM_VOTERS"].apply(list).to_dict()
        
        return (k, labels, result_household_dict, result_voters_dict)
    
    return None
    
'''

def iterative_kmedoids_manhattan(households, k_min, k_max, max_households, max_voters):
    # Functions take as input user-defined upper limit thresholds of households and voters, household-level DF, and a large enough range of k
    
    # Make a copy to avoid modifying original
    households = households.copy()
    
    # Establish range of possible k's to test
    for k in range(k_min, k_max + 1):
    
        # Define clusters with KMedoids Manhattan helper function
        
        coords = households[['lon', 'lat']].values.tolist()  # transform coordinates from Households DF to list
        
        labels = clustering_kmedoids_manhattan(coords, k)  # Pass coordinates list as argument to pre-defined function and perform cluster
        
        # Project the clusters back onto the household set
        households['cluster'] = labels
        
        # Construct dictionary of clusters with number of households
        household_dict = households.groupby("cluster")["id"].apply(list).to_dict()

        # Construct dictionary of clusters with number of voters
        voters_dict = households.groupby("cluster")["NUM_VOTERS"].apply(list).to_dict()
        
        ##########
        # Checking the upper limit constraints
        ##########
        
        # set default
        # If at any point, this default is switched to False, it will increase k by 1 and send the clustering to a new round.
        conditions_met = True
        
        # 1. Number of Households per cluster
    
        # Call function defined above on the household dictionary
        num_households = check_num_households(household_dict)
        
        # Define for loop to send cluster to new k, if any cluster exceeds the limit
        for household in num_households:
            if household > max_households:
                conditions_met = False
                break
            
        if not conditions_met:
            continue
                
        # 2. Number of Voters per cluster
        
        # Call function defined above to check the upper limit threshold of voters per cluster.
        num_voters = check_num_voters(voters_dict)
        
        # Send to next round of k clusters, if exceeded
        for voters in num_voters:
            if voters > max_voters:
                conditions_met = False
                break
        
        if not conditions_met:
            continue
        
        # At this stage, the perfect k has been found
        
        # Assign final k clusters onto households DF
        households['cluster'] = labels
        households.to_csv('households_manhattan.csv', index=False)

        # Create resulting dictionaries for both limits
        result_household_dict = households.groupby("cluster")["id"].apply(list).to_dict()
        result_voters_dict = households.groupby("cluster")["NUM_VOTERS"].apply(list).to_dict()
        
        # Return the results
        return (households, k, labels, result_household_dict, result_voters_dict)
    
    # Return None if no optimal k was found
    return None


def iterative_kmedoids_haversine(households, k_min, k_max, max_households, max_voters):
    """
    Iteratively find the minimum k that satisfies constraints using KMedoids with Haversine distance
    
    Parameters:
    -----------
    households : DataFrame
        Must contain columns: 'id', 'lat', 'lon', 'NUM_VOTERS'
    k_min : int
        Minimum number of clusters to try
    k_max : int
        Maximum number of clusters to try
    max_households : int
        Maximum households allowed per cluster
    max_voters : int
        Maximum voters allowed per cluster
    
    Returns:
    --------
    tuple : (households, k, labels, result_household_dict, result_voters_dict)
        or None if no valid clustering found
    """
    
    # Make a copy to avoid modifying original
    households = households.copy()
    
    # Establish range of possible k's to test
    for k in range(k_min, k_max + 1):
    
        # Define clusters with KMedoids Haversine helper function
        coords = households[['lon', 'lat']].values.tolist()  # transform coordinates from Households DF to list
        
        labels = clustering_kmedoids_haversine(coords, k)  # Pass coordinates list and perform clustering
        
        # Project the clusters back onto the household set
        households['cluster'] = labels
        
        # Construct dictionary of clusters with number of households
        household_dict = households.groupby("cluster")["id"].apply(list).to_dict()

        # Construct dictionary of clusters with number of voters
        voters_dict = households.groupby("cluster")["NUM_VOTERS"].apply(list).to_dict()
        
        ##########
        # Checking the upper limit constraints
        ##########
        
        # set default
        # If at any point, this default is switched to False, it will increase k by 1 and send the clustering to a new round.
        conditions_met = True
        
        # 1. Number of Households per cluster
        num_households = check_num_households(household_dict)
        
        # Define for loop to send cluster to new k, if any cluster exceeds the limit
        for household in num_households:
            if household > max_households:
                conditions_met = False
                break
            
        if not conditions_met:
            continue
                
        # 2. Number of Voters per cluster
        num_voters = check_num_voters(voters_dict)
        
        # Send to next round of k clusters, if exceeded
        for voters in num_voters:
            if voters > max_voters:
                conditions_met = False
                break
        
        if not conditions_met:
            continue
        
        # At this stage, the perfect k has been found
        
        # Assign final k clusters onto households DF
        households['cluster'] = labels
        households.to_csv('households_haversine.csv', index=False)

        # Create resulting dictionaries for both limits
        result_household_dict = households.groupby("cluster")["id"].apply(list).to_dict()
        result_voters_dict = households.groupby("cluster")["NUM_VOTERS"].apply(list).to_dict()
        
        # Return the results
        return (households, k, labels, result_household_dict, result_voters_dict)
    
    # Return None if no optimal k was found
    return None


# Function to get a results dictionary

def get_full_results(result_household_dict, result_voters_dict):
    
    # Initilize at empty
    cluster_results = []
    
    # Loop through each clust in both resulting dictionaries and count the houses and voters again
    for cluster in result_household_dict.keys():
        
        # Count houses and voters again
        num_households = len(result_household_dict[cluster])
        num_voters = sum(result_voters_dict[cluster])
        # Append back to results list
        cluster_results.append({
            "cluster": cluster,
            "households": num_households,
            "voters": num_voters,
            "ids": result_household_dict[cluster]
        })

    return cluster_results


# Results Part


# With the results above, lets define a function to display just the top10 clusters, so its easier to visualize
# I define a top cluster as the ones that hold the most houses and voters and still adhere to the constraints

def get_top_10_clusters(cluster_results):
    # Passing above results as arguments

    # Sort by #households descending (you can change this)
    cluster_results.sort(key=lambda x: x["households"], reverse=True)

    # Return the Top 10
    return cluster_results[:10]



# Visualize Results



# Lets now visualize the top10 clusters

def plot_clusters_interactive(top10, zoom_start=11): # top10 is the result from above, start folioum zoom at 11, this looks good

    # transform, check for cluster actually being ints
    # top10["cluster"] = top10["cluster"].astype(int)


    unique_clusters = sorted(top10["cluster"].unique())
    num_clusters = len(unique_clusters)

    # Create colomap
    colormap = cm.get_cmap('tab10', num_clusters)
    
    cluster_color = {
        cl: colors.rgb2hex(colormap(i))
        for i, cl in enumerate(unique_clusters)
    }

    # Center map on median coordinates
    lat_center = top10["lat"].median()
    lon_center = top10["lon"].median()

    m = folium.Map(location=[lat_center, lon_center], zoom_start=zoom_start)

    for _, row in top10.iterrows():
        cl = row["cluster"]
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=3,
            popup=f"ID: {row['id']}<br>Cluster: {cl}",
            color=cluster_color[cl],
            fill=True,
            fill_color=cluster_color[cl],
            fill_opacity=0.7
        ).add_to(m)

    return m

