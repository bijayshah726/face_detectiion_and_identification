import cv2
import face_recognition
import streamlit as st
import tempfile
import os
import shutil
from zipfile import ZipFile
import numpy as np

DEMO_VIDEO = 'ri1.mp4'
detected_faces = []  # List to store detected face images

import base64

def get_binary_file_downloader_html(bin_file, label='Download', button_text='Download'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">{button_text}</a>'
    return href


def main():
    st.title('Face Detection and Identification App')
    st.sidebar.title('Face Detection and Identification App')
    st.sidebar.subheader('Parameters')

    # Create a space for displaying the original video
    video_container = st.empty()

    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value=0.0, max_value=1.0, value=0.5)
    model_selection = st.sidebar.selectbox('Model Selection', options=["hog", "cnn"])

    video_file_buffer = st.sidebar.file_uploader("Upload a video", type=["mp4"]) #other file formats if needed can be included
    tffile = tempfile.NamedTemporaryFile(delete=False)

    if not video_file_buffer:
        st.sidebar.warning("Please upload a video in mp4 format.")
        return  # Stop further execution if no video is selected

    
    tffile.write(video_file_buffer.read())
    vid = cv2.VideoCapture(tffile.name)

    st.markdown('## Detected Face')
    stframe = st.empty()

    st.sidebar.text('Input Video')
    st.sidebar.video(tffile.name)

    # Create a directory to store extracted face images
    extracted_faces_dir = tempfile.mkdtemp()

    known_face_encodings = []
    face_names = []
    # Display the original video
    #video_container.video(vid.read())
    ii=0
    while vid.isOpened():
        ret, image = vid.read()

        if not ret:
            break
        
        video_container.image(image, caption= "Original Video", channels="BGR", use_column_width=True)
        if model_selection == 'hog':
            face_locations = face_recognition.face_locations(image, model='hog')
        elif model_selection == 'cnn':
            face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0,model='cnn')

        #face_locations = face_recognition.face_locations(image, model=model_selection)
        if len(face_locations) > 0:
            temp=0
            for idx, (top, right, bottom, left) in enumerate(face_locations):
                face_image = image[top:bottom, left:right]
                face_encodings = face_recognition.face_encodings(face_image)
                if len(face_encodings) > 0:
                    face_encoding = face_encodings[0]

                    # Save the detected face image
                    face_image_path = os.path.join(extracted_faces_dir, f"face_{ii+idx + 1}.jpg")
                    #cv2.imwrite(face_image_path, cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(face_image_path, cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))

                    known_face_encodings.append(face_encoding)
                    face_names.append(f"Face {ii+idx + 1}")

                    stframe.image(face_image, caption=f"Face {ii+idx + 1}", use_column_width=True)
                    temp+=1
                # Store the detected face image in the list
                #detected_faces.append(face_image)
        ii+= temp

    vid.release()
    video_container.empty()
    stframe.empty()

    # Clear the content of detected faces
    #video_container.clear()
    #stframe.clear()
    # Directory containing the face images
    faces_directory = extracted_faces_dir


    # List to store face image filenames
    face_filenames = [filename for filename in os.listdir(faces_directory) if filename.startswith('face_')]

    # Create a dictionary to store face encodings
    face_encodings = {}
    # Threshold for considering faces as similar
    similarity_threshold = 0.2

    for face_filename in face_filenames:
        # Load each face image
        face_image = face_recognition.load_image_file(os.path.join(faces_directory, face_filename))
        
        # Compute face encodings
        face_encoding = face_recognition.face_encodings(face_image)
        if len(face_encoding) > 0:
            face_encodings[face_filename] = face_encoding[0]

    # Create a temporary directory to store grouped faces
    with tempfile.TemporaryDirectory() as grouped_faces_directory:
        # Threshold for considering faces as similar
        similarity_threshold = 0.6

        # Create a dictionary to store face groups
        face_groups = {}
        for face_filename, _ in face_encodings.items():
            face_groups[face_filename] = face_filename
        #print("Before merge", face_groups)
        # Calculate distances between all face pairs
        ITERATED=[]
        for face_filename1, face_encoding1 in face_encodings.items():
            print(face_filename1)
            for face_filename2, face_encoding2 in face_encodings.items():
                if face_filename1 != face_filename2 and face_filename2 not in ITERATED :
                    distance = face_recognition.face_distance([face_encoding1], face_encoding2)
                    if distance <= similarity_threshold:
                        
                        # Add face to a group
                        group_id1 = face_groups[face_filename1]
                        group_id2 = face_groups[face_filename2]
                        if group_id1 != group_id2:
                            # Merge groups
                            group_ids = [group_id1, group_id2]
                            merged_group_id = min(group_ids)
                            #face_groups[face_filename1]= merged_group_id
                            #face_groups[face_filename2]= merged_group_id
                            for face_filename, group_id in face_groups.items():
                                if group_id in group_ids:
                                    face_groups[face_filename] = merged_group_id
            ITERATED.append(face_filename1)
        #print("IDS", face_groups)
    

        # Move grouped faces to their respective directories
        for face_filename, group_id in face_groups.items():
            group_directory = os.path.join(grouped_faces_directory, f'Group_{group_id[:-4]}')
            os.makedirs(group_directory, exist_ok=True)

            source_path = os.path.join(faces_directory, face_filename)
            target_path = os.path.join(group_directory, face_filename)
            shutil.move(source_path, target_path)

        with ZipFile('grouped_faces.zip', 'w') as zipf:
            for root, dirs, _ in os.walk(grouped_faces_directory):
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    for file_name in os.listdir(dir_path):
                        file_path = os.path.join(dir_path, file_name)
                        zipf.write(file_path, os.path.relpath(file_path, grouped_faces_directory))


    # Display the download link for the zip file
    #st.sidebar.markdown('[Download Extracted Faces](extracted_faces.zip)')
    st.sidebar.markdown(get_binary_file_downloader_html('grouped_faces.zip', 'Download Grouped Extracted Faces', button_text="Download Grouped Extracted Faces"), unsafe_allow_html=True)

    # Create a placeholder for displaying grouped images
    grouped_images_placeholder = st.empty()

    for root, dirs, _ in os.walk(grouped_faces_directory):
        print(dirs, grouped_faces_directory)
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            for file_name in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file_name)
                print(file_path)
                image_path = os.path.join(grouped_faces_directory, file_path)
                print(image_path)
                grouped_images_placeholder.image(image_path, caption=f"Group {dir_path}", use_column_width=True)




    # Clean up
    #shutil.rmtree(grouped_faces_directory)

if __name__ == '__main__':
    main()
