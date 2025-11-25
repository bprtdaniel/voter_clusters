
# import all libraries and from helper script
import streamlit as st
from helper import (iterative_kmeans,   # iteratively updating KMeans
                    iterative_kmedoids_haversine, # Kmedoids for manhattan
                    iterative_kmedoids_manhattan, # Kmedoids for Haversine
                    get_full_results, # function to grab all results from the cluster process
                    get_top_10_clusters, # function to reduce the results to the top10 clusters
                    plot_clusters_interactive) # function to plot
import pandas as pd
from streamlit_folium import st_folium
from streamlit.components.v1 import html

#############
# Config Section
#############


# page config in wide layout
st.set_page_config(
    page_title="Upload Data",
    layout="wide"
)

# Set title
st.title("Upload your Files here:")

st.subheader("Let's upload a Household File first:")

##############
# Upload Section
##############



# finished household level voter file 
household_file = st.file_uploader(
    label= "Household Upload",
    type=["csv"],
    key="household_upload"
)

# upload file here
# read as csv
# transform the IDs to INTs, this has casused major issues, as after the upload ID type is switched to Objects or Floats
# This causes the final merge in the clustering functions to return NULL as the IDs are of different types

# Also: Another major fix was to write another copy of the uploaded file and save it seperately
# This is done, so the original file is not overwritten and does not break the cluster results later
if household_file is not None:
    households_df = pd.read_csv(household_file)
    households_df['id'] = households_df['id'].astype(int)
    st.session_state["households_original"] = households_df  
    st.session_state["households_uploaded"] = True # amend session state
    st.success("Household file uploaded!") # success message

st.subheader("Let's also upload our Distances for later")

# same for matrix
matrix_file = st.file_uploader(
    label= "Matrix Upload",
    type=["csv"],
    key="matrix_upload"
)

if matrix_file is not None:
    dist_df = pd.read_csv(matrix_file, index_col=0)
    dist_df.index = dist_df.index.astype(float).astype(int) # type conversions for matrix
    dist_df.columns = pd.to_numeric(dist_df.columns).astype(int)
    st.session_state["distance_matrix"] = dist_df
    st.success("Distance matrix uploaded!")


st.subheader("Now, let's define some parameters")

# Message to first upload household level data, as thats the most essential part
if "households_original" not in st.session_state: 
    st.error("Please upload your Household File first.")
    st.stop()
    
    
st.write("""
Choose between **K-Means** and **K-Medoids**, then set your clustering parameters.
""")

st.subheader("1. Select Clustering Algorithm")

# assign the 3 methods, no more road-distance kmedoids now
# Radio button is best here

# I now use Manhattan (for city block logic) and also Haversine (curverture logic) but they should not be too different from each other here.
method = st.radio(
    "Choose algorithm:",
    ["K-Means", "K-Medoids Manhattan", "K-Medoids Haversine"], 
    horizontal=True
)



# write to state
st.session_state["clustering_method"] = method


st.subheader("2. Set Parameters")
st.write("Remember, not every house will have someone currently at home, feel free to set these a little higher.")
st.write("Try something like 50 houses and 100 people.")
col1, col2 = st.columns(2)

with col1:
    max_households = st.number_input(
        "Maximum Number of Households per List",
        min_value=1,
        max_value=5000,
        step=1
    )

with col2:
    max_voters = st.number_input(
        "Maximum Number of Voters per List",
        min_value=1,
        max_value=20000,
        step=1
    )

# Save parameters in session so they are saved for the next part
st.session_state["max_households"] = max_households
st.session_state["max_voters"] = max_voters


####
# Perform the Clustering
####


# insert cluster button
if st.button("Run Clustering"):
    
    # Ensure everything uploaded, otherwise it should not run
    if "households_original" not in st.session_state:
        st.error("Please upload your Household File first.")
        st.stop()

    # Assignt the variables for the next part, from the state
    households_df = st.session_state["households_original"].copy() 
    method = st.session_state["clustering_method"]
    max_households = st.session_state["max_households"]
    max_voters = st.session_state["max_voters"]

    # little wait message, better than a loading circle thingy
    # this needs to go before the if clauses for the method
    st.info(f"Running {method} clustering. This will take a moment.")

    # Run selected algorithm
    if method == "K-Means":
        result = iterative_kmeans( # simple kmeans
            households=households_df,
            k_min=2, # switched to hardcoding min and max k as they impact the range, a little change from the original function
            k_max=100,
            max_households=max_households,
            max_voters=max_voters,
        )
    elif method == "K-Medoids Manhattan":
        result = iterative_kmedoids_manhattan( #manhattan
            households=households_df,
            k_min=2,
            k_max=100,
            max_households=max_households,
            max_voters=max_voters,
        )
    else:  # K-Medoids Haversine
        result = iterative_kmedoids_haversine( # and haversine
            households=households_df,
            k_min=2,
            k_max=100,
            max_households=max_households,
            max_voters=max_voters,
        )

    # No valid solution if the result from the functions is 'None'.
    if result is None:
        st.error("No valid clustering found within the constraints.")
        st.stop()

    # Save the results clearly
    # I just transferred this from all the previous steps in notebooks, its not all necessary here
    households_df, best_k, labels, result_household_dict, result_voters_dict = result

    # write to state
    st.session_state["best_k"] = best_k
    st.session_state["labels"] = labels
    st.session_state["result_household_dict"] = result_household_dict
    st.session_state["result_voters_dict"] = result_voters_dict
    st.session_state["households_clustered"] = households_df  # IMPORTANT: This is now the clustered df, which came from the copy
    st.session_state["cluster_result"] = result

    st.success("List selection completed successfully!") # success message
    st.write(f"Number of Lists created: {len(households_df['cluster'].unique())}") # Show number of lists


# I want to be able to visualize 10 lists on the map
# I usually generate around 35 to 42 clusters acros the methods, which would be too cluttered to show

# After showing 10, I want to download up to 10, which should be customizable by the user
st.write("Let's go ahead and look at 10 of the lists that just got created, choose whichever and as many as you like!")


# Map button
if st.button("Show My Lists on a Map"):
    # write the map data to session state here so it does not disappear
    st.session_state["show_map"] = True

# This checks if buton waw alreayd clicked, fixed an issue where all content was overriden, get from state
if st.session_state.get("show_map", False):

    # again, break if no custering for some reason
    if "households_clustered" not in st.session_state:
        st.error("Run the clustering tool first!")
        st.session_state["show_map"] = False
        st.stop()

    # Get clustered results from session state
    households_df = st.session_state["households_clustered"]
    result_household_dict = st.session_state["result_household_dict"]
    result_voters_dict = st.session_state["result_voters_dict"]

    # Get full results and filter to top 10 using functions defined in helper
    full_results_dict = get_full_results(result_household_dict, result_voters_dict)
    top10 = get_top_10_clusters(full_results_dict)
    top_cluster_labels = [c["cluster"] for c in top10]
    
    # Filter results to top 10 
    top10_df = households_df[households_df["cluster"].isin(top_cluster_labels)]

    # finally, plot with helper
    m = plot_clusters_interactive(top10_df)
    
    # show with html, as pyarrow was throwing errors all over the place
    html(m._repr_html_(), width=900, height=600)
    
    
    
    # Start the download section below the map
    st.markdown("---") # break
    st.subheader("🗳️Select Clusters to Download🗳️")
    
    # Show cluster info
    st.write("**Top 10 Clusters:**")
    # Show how many houses and voters for each List, so volunteers can combine map info with stats
    for cluster in top10:
        st.write(f"**List {cluster['cluster']}:** {cluster['households']} Houses and {cluster['voters']} Voters")
    
    
    
    # Insert multiselect as a fix for some dynamic download issues
    selected_clusters = st.multiselect(
        "Which List(s) would you like to download and volunteer for?:",
        options=top_cluster_labels, # give all top10
        default=top_cluster_labels, # start at top10 and X away 
        key="cluster_selector"
    )
    
    if selected_clusters:
        # Filter dataframe to selected clusters
        download_df = households_df[households_df["cluster"].isin(selected_clusters)].copy()
        
        # Show preview
        st.write(f"**Selected {len(selected_clusters)} Lists with {len(download_df)} total Houses**")
        st.write(f"**This looks like a great selection, thank you for volunteering!**")
        st.write("**Click the Download Button below and happt GOTV!!**")
        
        # insert download button
        
        csv = download_df.to_csv(index=False)
        st.download_button(
            label=f"Download Selected Lists",
            data=csv,
            file_name=f"Get_Out_The_Vote_Lists.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:   # if no list were selected and button is clicked
        st.info("Select at least one List to download.")
        
    st.markdown("---") # break for footer
    st.markdown(
        """
        <div style='text-align: center; color: gray; padding: 20px;'>
            <p>Voter Clustering App | Georgetown University | Daniel Boppert | 2024</p>
        </div>
        """,
        unsafe_allow_html=True
)