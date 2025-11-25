import streamlit as st
from helper import clustering_kmeans, clustering_kmedoids, check_num_households, check_num_voters, iterative_kmeans, iterative_kmedoids_haversine, iterative_kmedoids_manhattan, get_full_results, get_top_10_clusters, plot_clusters_interactive
import pandas as pd
from streamlit_folium import st_folium
from streamlit.components.v1 import html


st.set_page_config(
    page_title="Upload Data",
    layout="wide"
)

st.title("Upload your Files here:")

st.subheader("Let's upload a Household File first:")

household_file = st.file_uploader(
    label= "Household Upload",
    type=["csv"],
    key="household_upload"
)


if household_file is not None:
    households_df = pd.read_csv(household_file)
    households_df['id'] = households_df['id'].astype(int)
    st.session_state["households_original"] = households_df  # Store original
    st.session_state["households_uploaded"] = True
    st.success("Household file uploaded!")

st.subheader("Let's also upload our Distances for later")


matrix_file = st.file_uploader(
    label= "Matrix Upload",
    type=["csv"],
    key="matrix_upload"
)

if matrix_file is not None:
    dist_df = pd.read_csv(matrix_file, index_col=0)
    dist_df.index = dist_df.index.astype(float).astype(int)
    dist_df.columns = pd.to_numeric(dist_df.columns).astype(int)
    st.session_state["distance_matrix"] = dist_df
    st.success("Distance matrix uploaded!")


st.subheader("Now, let's define some parameters")

if "households_original" not in st.session_state: 
    st.error("Please upload your Household File first.")
    st.stop()
    
    
st.write("""
Choose between **K-Means** and **K-Medoids**, then set your clustering parameters.
""")

st.subheader("1. Select Clustering Algorithm")

method = st.radio(
    "Choose algorithm:",
    ["K-Means", "K-Medoids Manhattan", "K-Medoids Haversine"],
    horizontal=True
)

st.session_state["clustering_method"] = method


st.subheader("2. Set Parameters")
st.write("Remember, not every house will have someone currently at home, feel free to set these a little higher.")
st.write("Try something like 50 houses and 100 people.")
col1, col2 = st.columns(2)

with col1:
    max_households = st.number_input(
        "Max households per cluster",
        min_value=1,
        max_value=5000,
        step=1
    )

with col2:
    max_voters = st.number_input(
        "Max voters per cluster",
        min_value=1,
        max_value=20000,
        step=1
    )

# Save parameters in session
st.session_state["max_households"] = max_households
st.session_state["max_voters"] = max_voters


####
# Perform the Clustering
####
if st.button("Run Clustering"):
    # Ensure everything uploaded
    if "households_original" not in st.session_state:
        st.error("Please upload your Household File first.")
        st.stop()

    households_df = st.session_state["households_original"].copy()  # Use original
    method = st.session_state["clustering_method"]
    max_households = st.session_state["max_households"]
    max_voters = st.session_state["max_voters"]

    st.info(f"Running {method} clustering… This may take a moment.")

    # Run selected algorithm
    if method == "K-Means":
        result = iterative_kmeans(
            households=households_df,
            k_min=2,
            k_max=100,
            max_households=max_households,
            max_voters=max_voters,
        )
    elif method == "K-Medoids Manhattan":
        result = iterative_kmedoids_manhattan(
            households=households_df,
            k_min=2,
            k_max=100,
            max_households=max_households,
            max_voters=max_voters,
        )
    else:  # K-Medoids Haversine
        result = iterative_kmedoids_haversine(
            households=households_df,
            k_min=2,
            k_max=100,
            max_households=max_households,
            max_voters=max_voters,
        )

    # No valid solution
    if result is None:
        st.error("No valid clustering found within the constraints.")
        st.stop()

    # Unpack correctly: 5 values
    households_df, best_k, labels, result_household_dict, result_voters_dict = result

    # Save to session state - use new key for clustered data
    st.session_state["best_k"] = best_k
    st.session_state["labels"] = labels
    st.session_state["result_household_dict"] = result_household_dict
    st.session_state["result_voters_dict"] = result_voters_dict
    st.session_state["households_clustered"] = households_df  # Save clustered version
    st.session_state["cluster_result"] = result

    st.success("Clustering completed successfully!")
    st.write(f"Best k found: {best_k}")
    st.write(f"Number of clusters created: {len(households_df['cluster'].unique())}")


st.write("Let's go ahead and look at 10 of the lists that just got created, choose whichever and as many as you like!")


if st.button("Show My Lists on a Map"):

    if "households_clustered" not in st.session_state:
        st.error("❌ Please run clustering first!")
        st.stop()

    # Get clustered results from session state
    households_df = st.session_state["households_clustered"]
    result_household_dict = st.session_state["result_household_dict"]
    result_voters_dict = st.session_state["result_voters_dict"]

    # Get full results and filter to top 10
    full_results_dict = get_full_results(result_household_dict, result_voters_dict)
    top10 = get_top_10_clusters(full_results_dict)
    top_cluster_labels = [c["cluster"] for c in top10]
    
    # Filter dataframe to only top 10 clusters
    top10_df = households_df[households_df["cluster"].isin(top_cluster_labels)]

    # Plot the map
    m = plot_clusters_interactive(top10_df)
    html(m._repr_html_(), width=900, height=600)
    

    
    
    
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>Voter Clustering App | Georgetown University | Daniel Boppert | 2024</p>
    </div>
    """,
    unsafe_allow_html=True
)