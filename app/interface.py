import streamlit as st
import zipfile
import os
import shutil

# File upload handler
def upload_zip_file(upload_dir='uploads/'):
    # Clear the upload directory
    # os.makedirs(upload_dir, exist_ok=True)
    if os.path.exists(upload_dir):
        shutil.rmtree(upload_dir)
    os.makedirs(upload_dir)

    uploaded_file = st.file_uploader("Upload a zip file with PDFs", type="zip")
    if uploaded_file:
        zip_path = os.path.join(upload_dir, uploaded_file.name)
        with open(zip_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(upload_dir)
        st.success("Uploaded and extracted successfully!")
        return upload_dir

# Streamlit app title
st.title("PDF Zip File Uploader")

# Call the upload function
upload_zip_file()
#check commit