import os
import streamlit as st
import zipfile
import shutil

# Define the folder where files will be uploaded
UPLOAD_FOLDER = 'resources\\uploads'
QUERY_FILE = os.path.join(UPLOAD_FOLDER, 'query.txt')

# Function to upload and extract zip files
def upload_zip_file():
    """Upload a zip file and extract its contents to the upload folder."""
    # Clear the upload directory
    if os.path.exists(UPLOAD_FOLDER):
        shutil.rmtree(UPLOAD_FOLDER)
    os.makedirs(UPLOAD_FOLDER)
    st.success("Previous upload session cleared, proceed to upload.")

    uploaded_file = st.file_uploader("Upload a zip file with PDFs", type="zip")
    if uploaded_file:
        zip_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(zip_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(UPLOAD_FOLDER)
        st.success("Uploaded and extracted successfully!")


# Page layout and title
st.set_page_config(page_title="File Upload and paperBot", page_icon="📂", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
        .title {
            text-align: center;
            color: #4CAF50;
            font-size: 2.5em;
            margin-bottom: 20px;
        }
        .subtitle {
            text-align: center;
            color: #555;
            font-size: 1.2em;
            margin-bottom: 40px;
        }
    </style>
""", unsafe_allow_html=True)

# Navigation for multi-page app
page = st.sidebar.selectbox("Select Page", ("Upload Files", "Send Query"))

if page == "Upload Files":
    # Title and description for file upload page
    st.markdown('<div class="title">File Upload</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Upload your zip file containing PDFs below.</div>', unsafe_allow_html=True)

    # Call the upload function
    upload_zip_file()

elif page == "Send Query":
    # Title and description for paperBot page
    st.markdown('<div class="title">paperBot</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Ask your question below:</div>', unsafe_allow_html=True)

    # User query input
    user_query = st.text_input("Your Question:", placeholder="Type your question here...")

    # Submit button for user query
    if st.button("Submit"):
        if user_query:
            with open(QUERY_FILE, "a") as f:  # Use "a" to append
                f.write(user_query + "\n")
            st.success("Query Added")
        else:
            st.warning("Please enter a question before submitting.")
