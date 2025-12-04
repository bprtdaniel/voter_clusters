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

    coords = np.array(input)
    
    # Haversine requires coordinates in radians and in [lat, lon] order, switch order as well
    coords_rad = np.radians(coords[:, [1, 0]])  
    
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
    households = households.copy()
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

# Unfortunately, I retired this part of the project as I was not able to get it to work in the timeframe.
# I think I got very close and I would love some guidance or analysis, if the reviewers can spot a major issue
# I did get cluster returns but they were spread apart 
# In a way, it did work, as I dont have a max_distance travelled upper limit, so it brute forces clusters across the current limitations.





# Very similar to KMeans function
def iterative_kmedoids(households, k_min, k_max, max_households, max_voters, distance_matrix):
    
    households = households.copy()
    households["id"] = households["id"].astype(int)
    
    if "cluster" in households.columns:
        households = households.drop(columns=["cluster"])
    
    # Filter households to only those in the distance matrix
    matrix_ids = set(distance_matrix.index.astype(int))
    households = households[households['id'].isin(matrix_ids)]
    print(f"Filtered to {len(households)} households that exist in distance matrix")
    
    df = distance_matrix.copy()
    IDs = df.index.to_numpy().astype(int)
    D = df.values.astype(float)
    D = np.minimum(D, D.T)  # Symmetrize the matrix
    
    for k in range(k_min, k_max + 1):
        
        labels = clustering_kmedoids(D, k)
        
        cluster_assignments = pd.DataFrame({
            "id": IDs,
            "cluster": labels
        })
        
        households_clusters = households.merge(cluster_assignments, on="id", how="left")
        
        household_dict = households_clusters.groupby("cluster")["id"].apply(list).to_dict()
        voters_dict = households_clusters.groupby("cluster")["NUM_VOTERS"].apply(list).to_dict()
        
        num_households = check_num_households(household_dict)
        if max(num_households) > max_households:
            print(f"k={k}: Failed - too many households in a cluster ({max(num_households)})")
            continue
                
        num_voters = check_num_voters(voters_dict)
        if max(num_voters) > max_voters:
            print(f"k={k}: Failed - too many voters in a cluster ({max(num_voters)})")
            continue
        
        print(f"✓ SUCCESS with k={k}")
        households_clusters.to_csv('households_clustered.csv', index=False)
        
        result_household_dict = households_clusters.groupby("cluster")["id"].apply(list).to_dict()
        result_voters_dict = households_clusters.groupby("cluster")["NUM_VOTERS"].apply(list).to_dict()

        return (households_clusters, k, labels, result_household_dict, result_voters_dict)
    
    print(f"No valid k found in range [{k_min}, {k_max}]")
    return None




# KMedoids with Manhattan Distance

def iterative_kmedoids_manhattan(households, k_min, k_max, max_households, max_voters):

    
    # Make a copy to avoid modifying original
    households = households.copy()
    

    for k in range(k_min, k_max + 1):
    
        # Define clusters with KMedoids Manhattan helper function
        
        coords = households[['lon', 'lat']].values.tolist()
        
        labels = clustering_kmedoids_manhattan(coords, k)
        
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
        
        # At this stage, the optimal k has been found
        
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


# KMedoids with curveture Haversine Distance

def iterative_kmedoids_haversine(households, k_min, k_max, max_households, max_voters):

    # Make a copy to avoid modifying original
    households = households.copy()
    
    for k in range(k_min, k_max + 1):
    
        # Define clusters with KMedoids Haversine helper function
        coords = households[['lon', 'lat']].values.tolist()  
        
        labels = clustering_kmedoids_haversine(coords, k)  
        
        # Project the clusters back onto the household sets
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
        
        # At this stage, the optimal k has been found
        
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

    # build map variable
    m = folium.Map(location=[lat_center, lon_center], zoom_start=zoom_start)

    for _, row in top10.iterrows(): # loop throuch each row
        cl = row["cluster"] # grab cluster
        folium.CircleMarker(
            location=[row["lat"], row["lon"]], # set marker on lat and lon
            radius=3,  # assign size to marker
            popup=f"Cluster: {cl}", # I just want to show cluster number here, ID is no use to user and address was too long
            color=cluster_color[cl], # now, assign one of the cluster colors above
            fill=True, # some adjustments
            fill_color=cluster_color[cl],
            fill_opacity=0.7
        ).add_to(m)

    return m

