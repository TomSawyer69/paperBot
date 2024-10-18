import streamlit as st
import os

# Use absolute paths
current_dir = os.path.dirname(__file__)  # Gets the directory where interface.py is located
file_dir = os.path.join(current_dir, "..", "resources")  # Path to the resources directory

# Load the contents of the text files
def load_file(file_name):
    file_path = os.path.join(file_dir, file_name)
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        return ''.join(lines)  # Return full content as a string
    except FileNotFoundError:
        return f"File '{file_name}' not found."

# File names
query_file = 'query.txt'
reply_file = 'reply.txt'

# Display the content in the UI
st.title("Query and Reply Viewer")

st.subheader("Query Text:")
query_content = load_file(query_file)
st.text_area("Query", query_content, height=200)

st.subheader("Reply Text:")
reply_content = load_file(reply_file)
st.text_area("Reply", reply_content, height=200)
