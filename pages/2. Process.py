import streamlit as st
from PIL import Image
import pandas as pd

st.set_page_config(
    page_title="Process Page",
    layout="wide"
)

st.markdown(
    """
    <div style='text-align: right; color: gray; padding: 10px;'>
        <p><strong>Developed by: Daniel Boppert</strong></p>
    </div>
    """,
    unsafe_allow_html=True
)

st.header("Data Process and Algorithm Design")

st.write("I got the voterfile from the Ohio Secretary of State's website, as Ohio provides free and open downloads to individual voter registration data.")

# Create hardcoded sample data from actual Ohio voter file
voterfile = {
    'SOS_VOTERID': ['OH123', 'OH124', 'OH125', 'OH126', 'OH127'],
    'COUNTY_NUMBER': ['77', '77', '77', '77', '77'],
    'COUNTY_ID': ['1063081', '766622', '656218', '971326', '730432'],
    'LAST_NAME': ['B', 'D', 'F', 'H', 'J'],
    'FIRST_NAME': ['A', 'C', 'E', 'G', 'I'],
    'MIDDLE_NAME': ['A1', 'C1', 'E1', 'G1', 'I1'],
    'SUFFIX': ['II', '', '', '', ''],
    'DATE_OF_BIRTH': ['1965-12-08', '1959-08-02', '1947-10-26', '1991-05-18', '1972-05-05'],
    'REGISTRATION_DATE': ['2016-03-29', '2000-07-13', '2008-10-06', '2010-08-03', '2008-02-29'],
    'VOTER_STATUS': ['CONFIRMATION', 'CONFIRMATION', 'CONFIRMATION', 'CONFIRMATION', 'CONFIRMATION'],
    'PARTY_AFFILIATION': ['', '', '', '', ''],
    'RESIDENTIAL_ADDRESS1': ['3044 IRA RD', '51 COTTER AVE', '750 AUSTIN AVE', '4299 APPIAN WAY', '786 GOOD PARK BLVD'],
    'RESIDENTIAL_SECONDARY_ADDR': ['', 'APT 3', '', '', ''],
    'RESIDENTIAL_CITY': ['AKRON', 'AKRON', 'AKRON', 'AKRON', 'AKRON'],
    'RESIDENTIAL_STATE': ['OH', 'OH', 'OH', 'OH', 'OH'],
    'RESIDENTIAL_ZIP': ['44333', '44305', '44306', '44333', '44320'],
    'CONGRESSIONAL_DISTRICT': ['13', '13', '13', '13', '13'],
    'STATE_REPRESENTATIVE_DISTRICT': ['31', '32', '33', '31', '33'],
    'STATE_SENATE_DISTRICT': ['27', '28', '28', '27', '28'],
    'PRECINCT_NAME': ['BATH TWP C', 'AKRON 10-C', 'AKRON 7-G', 'BATH TWP E', 'AKRON 4-F'],
    'PRECINCT_CODE': ['77ARE', '77AIH', '77AFQ', '77ARI', '77ACY']
}

voterfile = pd.DataFrame(voterfile)

st.subheader("Sample Ohio Voter File Data")
st.dataframe(voterfile, use_container_width=True)

st.caption(f"Showing {len(voterfile)} sample records from Summit County.")



st.subheader("Household Level Dataset")

st.write("By counting the instances of unique addresses, I collapsed the dataset onto hosehold level and assigned th result as the number of registered voters.")


household_sample = {
    'hosuehold_ID': [181995, 42004, 4605, 142831, 49538],
    'Address': [
        '2890 WIGEON WAY APT 312, COVENTRY TOWNSHIP, OH 44319',
        '3071 9TH ST , CUYAHOGA FALLS, OH 44221',
        '943 REED AVE , AKRON, OH 44306',
        '1916 ADELAIDE BLVD , AKRON, OH 44305',
        '2786 SPRINGFIELD LAKE DR , AKRON, OH 44312'
    ],
    'Household_Voters': [1, 2, 4, 1, 2]
}

household_sample = pd.DataFrame(household_sample)

st.dataframe(household_sample, use_container_width=True)


st.subheader("Batch Upload the Household Data into a Geolocation Service and extract Latitude and Longitude.")

full_data = {
    'Voter_ID': [181995, 42004, 4605, 142831, 49538],
    'Longitude': [-81.53137473, -81.49352503, -81.49311647, -81.45925807, -81.42478419],
    'Latitude': [41.01143738, 41.15443106, 41.04298328, 41.06509667, 41.02730094],
    'Household_Voters': [1, 2, 4, 1, 2],
    'Street_Address': [
        '2890 WIGEON WAY APT 312',
        '3071 9TH ST',
        '943 REED AVE',
        '1916 ADELAIDE BLVD',
        '2786 SPRINGFIELD LAKE DR'
    ],
    'City': ['COVENTRY TOWNSHIP', 'CUYAHOGA FALLS', 'AKRON', 'AKRON', 'AKRON'],
    'State': ['OH', 'OH', 'OH', 'OH', 'OH'],
    'Zip': ['44319', '44221', '44306', '44305', '44312']
}

full_data = pd.DataFrame(full_data)

st.dataframe(full_data, use_container_width=True)


st.info("For K-Means, this is where the process ends. For K-Medoids, we still need the road-distances.")

st.subheader("Open Source Road-Distance Matrix with Project OSRM")

st.write("""
        
        - To get the road distance matrix, without going bankrupt using the Google Matrix API, I use Project OSRM's Docker set-up.
        - Once a local instance is running, I can create a script that batch-calculates the road-distances via the API.
        - As a result, we get a non-symmetric Matrix that holds pairwise distances for all addresses.
        - For this to work, we need to pass a symmetric Matrix, so we need to assume (Road Distance A -> B) = (Road Distance B -> A).
        - I set the Matrix to use the minimum of those distances and make sure the Matrix is equal to its Transpose and the diagonal is 0s only.
        """)

st.info("The most crucial step is to keep the original household ID unique and unchanged in order across all steps and joins.")

matrix = {
    'origin_id': [181995, 42004, 4605, 142831, 49538],
    '181995': [0, 38247.3, 32458.1, 9107.6, 11994.5],
    '42004': [38247.3, 0, 9598.4, 49678.8, 38266.5],
    '4605': [32458.1, 9598.4, 0, 39498.8, 25808.8],
    '142831': [9107.6, 49678.8, 39498.8, 0, 16420.7],
    '49538': [11994.5, 38266.5, 25808.8, 16420.7, 0]
}


matrix = pd.DataFrame(matrix)

st.dataframe(matrix, use_container_width=True)



st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>Voter Clustering App | Georgetown University | Daniel Boppert | 2025</p>
    </div>
    """,
    unsafe_allow_html=True
)



st.subheader("Algorithms")


st.write("""
        
        Both implementations followed the same logic:
        
        The algorithm is wrapped in a loop that performs KMeans/KMedoids at increasing numbers of k, until the upper limit constraints are met.
        
        
        1. Initialize the algorithm at k = 2
        2. Perform the clustering
        3. Check if number of households or number of voters exceed the maximum
        4. If exceeded, increase k and send the algorithm to another round
        5. Clustering will stop once an optimal k is found and returns lists of results.
        
        """)