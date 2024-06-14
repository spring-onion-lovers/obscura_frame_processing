import cv2 as cv
import numpy as np
from face_recog_methods import compute_face_embedding, mouse_callback, selected_faces
import mediapipe as mp
from face_embedding_storage import FaceEmbeddingStorage

fes = FaceEmbeddingStorage(debug_mode=True)

# #################################################
# CHANGE THE VARIABLES BELOW
# #################################################
_CAMERA_INDEX = 1  # 0 for built-in webcam, 1 for external webcam (or it could be different, try with each)

# use mediapipe to detect faces
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.face_detection

cap = cv.VideoCapture(_CAMERA_INDEX)

with mp_face_detection.FaceDetection() as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # calculate frame timestamp in ms
        frame_timestamp_ms = cap.get(cv.CAP_PROP_POS_MSEC)  # current frame timestamp in milliseconds
        results = face_detection.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

        bboxes = []
        embeddings = []

        if results.detections:
            # loop through each detection
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box

                # print(bbox)

                if bbox.xmin < 0 or bbox.ymin < 0 or bbox.width < 0 or bbox.height < 0:
                    continue

                xmin = int(bbox.xmin * frame.shape[1])
                ymin = int(bbox.ymin * frame.shape[0])
                width = int(bbox.width * frame.shape[1])
                height = int(bbox.height * frame.shape[0])

                # append the bbox to the list
                bbox = (xmin, ymin, width, height)
                bboxes.append(bbox)

                # Compute face embedding
                embedding = compute_face_embedding(frame, (xmin, ymin, width, height))

                # append to the list of embeddings
                if embedding is not None:
                    fes.add_embedding(embedding)
                    # embeddings.append(embedding)

        print(f"total embeddings: {fes.get_total_embeddings()}")
        print(f"total selected faces: {len(selected_faces)}")

        # Draw bounding boxes on the frame
        for idx, bbox in enumerate(bboxes):
            xmin, ymin, width, height = bbox

            # Compute face embedding for the current bbox
            embedding = compute_face_embedding(frame, (xmin, ymin, width, height))

            # Check if the embedding matches any of the selected faces
            is_selected = any(
                FaceEmbeddingStorage.calculate_embedding_similarity(e1=embedding, e2=e, threshold=0.6) for e in
                selected_faces)

            # Choose the color based on whether the face is selected
            color = (0, 0, 255) if is_selected else (0, 255, 0)  # Red for selected faces, green for others

            # Draw the bounding box
            cv.rectangle(frame, (xmin, ymin), (xmin + width, ymin + height), color, 2)

        cv.imshow('fd', frame)
        cv.setMouseCallback('fd', mouse_callback,
                            {
                                'frame': frame,
                                'bboxes': bboxes,
                                'embeddings': [embedding for embedding in fes.get_embeddings()],
                                'storage': fes
                            })

        # Quit the app
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
# out.release()
cv.destroyAllWindows()
