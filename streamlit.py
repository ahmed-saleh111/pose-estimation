import cv2
import streamlit as st
from ultralytics import YOLO
from pose import process_pose_frame
import tempfile
from datetime import datetime

st.title(":blue[School]")

st.markdown(
    ":orange-badge[⚠️ Upload video in mp4 format only]"
)

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi"])  # all video paths

if uploaded_file is not None:
    # st.badge("Success", icon=":material/check:", color="green")
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    st.divider()

    if st.button("Start processing"):
        vf = cv2.VideoCapture(tfile.name)

        if not vf.isOpened():
            st.markdown(
                ":orange-badge[⚠️ False]"
            )
        

        frame_width = int(vf.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(vf.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vf.get(cv2.CAP_PROP_FPS)

        # Output file with timestamp (unique per execution)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"output_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        stframe = st.empty()

        while True:
            ret, frame = vf.read()
            if not ret:
                break
            processed_frame = process_pose_frame(frame)
            # Immediately save each frame
            out.write(processed_frame)

            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB")
        vf.release()


# st.video(uploaded_file, autoplay=True)
#streamlit run streamlit.py