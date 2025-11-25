import streamlit as st
from PIL import Image


st.set_page_config(
page_title="What is a Voter List!",
layout="wide"
)


st.title("What is a Voter List?")

st.subheader("I'm glad you asked!")


st.write("""For each election, we create so called Voter Targets.
        
        Generally, these are eligible voters we want to speak to about the upcoming election.
        
        We rely on the great help of volunteers like you, to talk to as many voters as possible.
        
        In order to do so, you need what we call a "Turf List", but we'll just call it a voter list here.
        
        Basically, this is a smaller list, taken from our overall voter targets, so we can pass it out to you, so you know who to talk to and where.
        
        Usually, we'd have to draw little clusters on a map of the entire state and capture each voter's address individually.
        
        Additionally, we have to define how many houses each volunteer visits and how many voters they would talk to. 
        
        This is a very tedious and manual process and we also don't know about the preferences of each volunteer.
        
        We want to make this experience as easy and inclusive as possible for you, so we have created this little tool.
        
        It lets you create your own list, since you best know where and how many people you want to talk to.
        
        This lets you better adjust your volunteering experience and for us its easier and faster.""")


st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>Voter Clustering App | Georgetown University | 2024</p>
    </div>
    """,
    unsafe_allow_html=True
)
