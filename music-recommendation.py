#!/usr/bin/env python
# coding: utf-8

# In[1]:





# In[2]:


import streamlit as st
from music_recommendation_system import hybrid_recommendations


# In[ ]:


st.title("Music Recommendation System")
st.write("Enter your music preferences to get personalized recommendations.")


# In[3]:


user_input = st.text_input("Enter your preferences:")


# In[5]:


if st.button("Get Recommendations"):
    if user_input:
        # Call the recommendation function
        recommendations = hybrid_recommendations(user_input,num_recommendations=5)
        st.write("Your Recommendations:")
        st.dataframe(recommendations)
    else:
        st.write("Please enter your preferences.")


# In[ ]:




