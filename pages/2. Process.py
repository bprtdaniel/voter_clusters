import streamlit as st
from PIL import Image
import pandas as pd

st.set_page_config(
    page_title="Process Page",
    layout="wide"
)

st.write("I got the voterfile from the Ohio Secretary of State's website, as Ohio provides free and open downloads to individual voter registration data.")

# Create hardcoded sample data from actual Ohio voter file
data = {
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

df = pd.DataFrame(data)

st.subheader("Sample Ohio Voter File Data")
st.dataframe(df, use_container_width=True)

st.caption(f"Showing {len(df)} sample records from Summit County:")