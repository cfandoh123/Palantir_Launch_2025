# test_streamlit.py
import streamlit as st

st.title("Simple Test App")
st.write("If you can see this, Streamlit is working!")

# Display environment information
import os
import socket
st.write(f"Hostname: {socket.gethostname()}")
st.write(f"IP Addresses: {socket.gethostbyname_ex(socket.gethostname())}")
st.write("Environment Variables:")
for key, value in os.environ.items():
    if 'path' in key.lower() or 'host' in key.lower() or 'proxy' in key.lower() or 'url' in key.lower():
        st.write(f"{key}: {value}")