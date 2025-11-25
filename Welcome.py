import streamlit as st
from PIL import Image




st.set_page_config(
page_title="Cluster Tool!",
layout="wide"
)

st.title("Welcome to the Household Clustering Tool!")
st.subheader("🗳️This is your One-Stop-Shop to get your Voter List and Get Out The Vote!🗳️")

st.write("""This Tool allows you to:
        
        - Upload the data of the area you want to volunteer in
        - Select how many people you want to talk to
        - Customize the number of doors you want to knock
        - (Choose the clustering algorithm)""")



st.write("""After you've used the tool you can:
        
        - Choose which Voter List to use
        - See a map of where you are headed""")


st.info("Use the panel on the left to navigate the Tool. Happy GOTV!")






