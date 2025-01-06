import streamlit as st

st.title("OpenCV Streamlit Demo")
st.header("header")
image=st.file_uploader("Upload an image file")

if image:
    st.image(image)
st.text("This is text")
value=st.selectbox("Select Box", ["None", "Filter1", "Filter2"])
st.write(value)
checkbox = st.checkbox("Apply Filter")
st.write(checkbox)