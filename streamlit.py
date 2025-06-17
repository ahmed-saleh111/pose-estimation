import cv2
import streamlit as st
import tempfile
from datetime import datetime
import os
from pose import process_pose_frame

st.title(":blue[School Video Analysis]")

st.markdown(
    ":orange-badge[⚠️ Upload video in mp4 format only]"
)

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())

    st.divider()

    if st.button("Start processing"):
        vf = cv2.VideoCapture(tfile.name)

        if not vf.isOpened():
            st.markdown(":orange-badge[⚠️ Failed to open video]")
            tfile.close()
            os.unlink(tfile.name)
        else:
            frame_width = int(vf.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(vf.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = vf.get(cv2.CAP_PROP_FPS)

            # Output file with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"output_{timestamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

            stframe = st.empty()
            frame_idx = 0

            while True:
                ret, frame = vf.read()
                if not ret:
                    break
                frame_idx += 1

                # Process frame with all models
                processed_frame = frame.copy()
                processed_frame = process_pose_frame(processed_frame)  # Pose estimation

                # Save and display frame
                out.write(processed_frame)
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame_rgb, channels="RGB")

            vf.release()
            out.release()
            tfile.close()
            os.unlink(tfile.name)

            # Display the processed video
            st.video(output_path)