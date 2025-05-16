import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    st.title("Product Feature Comparison Assistant")
    st.write("Upload product specification sheets and compare features across products.")

    # File upload section
    st.header("Upload Specification Sheets")
    uploaded_files = st.file_uploader(
        "Choose specification files (PDF or TXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.write(f"Uploaded {len(uploaded_files)} files")
        
        # Feature input section
        st.header("Features to Compare")
        features_input = st.text_input(
            "Enter features to compare (comma-separated)",
            placeholder="e.g., RAM, Storage, Price"
        )
        
        if features_input:
            features = [f.strip() for f in features_input.split(",")]
            
            if st.button("Compare Features"):
                st.write("Feature comparison will be implemented here...")
                # TODO: Implement feature comparison logic

if __name__ == "__main__":
    main() 