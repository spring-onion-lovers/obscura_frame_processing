import cv2 as cv
import numpy as np
from face_recog_methods import compute_face_embedding, mouse_callback, selected_faces
import mediapipe as mp

# use mediapipe to detect faces
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.face_detection

cap = cv.VideoCapture(1)
input_fps = cap.get(cv.CAP_PROP_FPS)
out = None
fourcc = cv.VideoWriter_fourcc(*'mp4v')

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
                    embeddings.append(embedding)

        # Draw bounding boxes on the frame
        for idx, bbox in enumerate(bboxes):

            xmin, ymin, width, height = bbox

            if any(np.linalg.norm(embedding - embeddings[idx]) < 0.5 for embedding in selected_faces):
                color = (0, 0, 255)  # Red for selected faces
            else:
                color = (0, 255, 0)  # Green for other faces
            cv.rectangle(frame, bbox, color, 2)

        cv.imshow('Face Detection', frame)
        cv.setMouseCallback('Face Detection', mouse_callback,
                            {'frame': frame, 'bboxes': bboxes, 'embeddings': embeddings})

        # Quit the app
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
# out.release()
cv.destroyAllWindows()
