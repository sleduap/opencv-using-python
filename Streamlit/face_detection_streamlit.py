import numpy as np
import streamlit as st
import cv2
from PIL import Image
from io import BytesIO
import base64




# Create application title and file uploader widget.
st.title("OpenCV Deep Learning based Face Detection")
img_file_buffer = st.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png'])

# Initialize session state variables
if 'file_uploaded_name' not in st.session_state:
    st.session_state.file_uploaded_name = None
if 'detections' not in st.session_state:
    st.session_state.detections = None

# Function for detecting faces in an image.
def detectFaceOpenCVDnn(net, frame):
    # Create a blob from the image and apply some pre-processing.
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    # Set the blob as input to the model.
    net.setInput(blob)
    # Get Detections.
    detections = net.forward()
    return detections

# Function for annotating the image with bounding boxes for each detected face.
def process_detections(frame, detections, conf_threshold=0.5):
    bboxes = []
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]
    # Loop over all detections and draw bounding boxes around each face.
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_w)
            y1 = int(detections[0, 0, i, 4] * frame_h)
            x2 = int(detections[0, 0, i, 5] * frame_w)
            y2 = int(detections[0, 0, i, 6] * frame_h)
            bboxes.append([x1, y1, x2, y2])
            bb_line_thickness = max(1, int(round(frame_h / 200)))
            # Draw bounding boxes around detected faces.
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), bb_line_thickness, cv2.LINE_8)
    return frame, bboxes

# Function to load the DNN model.
@st.cache_resource()
def load_model():
    modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    configFile = "deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    return net

# Function to generate a download link for output file.
def get_image_download_link(img, filename, text):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href

net = load_model()

if img_file_buffer is not None:
    # Read the file and convert it to OpenCV Image.
    raw_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
    # Loads image in a BGR channel order.
    image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
    file_name = img_file_buffer.name

    # Create placeholders to display input and output images.
    placeholders = st.columns(2)
    # Display Input image in the first placeholder.
    placeholders[0].image(image, channels='BGR')
    placeholders[0].text("Input Image")

    # Create a Slider and get the threshold from the slider.
    conf_threshold = st.slider("SET Confidence Threshold", min_value=0.0, max_value=1.0, step=.01, value=0.5)

    # Check if the loaded image is "new", if so call the face detection model function.
    if file_name != st.session_state.file_uploaded_name:
        # Set the file_uploaded_name equal to the name of the file that was just uploaded.
        st.session_state.file_uploaded_name = file_name
        # Save the detections in the session-state for future use with the current loaded image.
        st.session_state.detections = detectFaceOpenCVDnn(net, image)
        st.write("New image uploaded, calling the face detection model.")
    else:
        st.write("Same image used, processing with the previous detections.")

    # Process the detections based on the current confidence threshold.
    out_image, _ = process_detections(image, st.session_state.detections, conf_threshold=conf_threshold)

    # Display Detected faces.
    placeholders[1].image(out_image, channels='BGR')
    placeholders[1].text("Output Image")

    # Convert OpenCV image to PIL.
    out_image = Image.fromarray(out_image[:, :, ::-1])
    # Create a link for downloading the output file.
    st.markdown(get_image_download_link(out_image, "face_output.jpg", 'Download Output Image'), unsafe_allow_html=True)