import uuid

import cv2
import face_recognition
import mediapipe as mp
import numpy as np
from sklearn.cluster import DBSCAN
from uuid import uuid4
import pickle
from collections import defaultdict

# Initialize Mediapipe face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


# #################################################
# CHANGE THE VARIABLES BELOW
# #################################################
_VIDEO_PATH = './images/group.mp4' # Absolute Path to the video file

def extract_face_locations_mediapipe(frame):
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the frame
    with mp_face_detection.FaceDetection(
            min_detection_confidence=0.2) as face_detection:
        results = face_detection.process(rgb_frame)

        face_locations = []
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                box = (int(bboxC.xmin * iw), int(bboxC.ymin * ih),
                       int(bboxC.width * iw), int(bboxC.height * ih))
                face_locations.append(box)

        return face_locations

def extract_face_embeddings(video_path):
    print(f"Extracting face embeddings from video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    all_face_encodings = []
    frame_faces = defaultdict(list)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        print(f"Processing frame {frame_count}")

        face_locations = extract_face_locations_mediapipe(frame)

        for tup in face_locations:
            (left, top, width, height) = tup
            print(left,top,width,height)
            encoding = encode_face(frame, face_locations)  # Replace with your face encoding function
            if encoding is not None:
                all_face_encodings.append(encoding)
                frame_faces[cap.get(cv2.CAP_PROP_POS_FRAMES)].append((top, left, top + height, left + width))

            print(len(all_face_encodings))

    cap.release()
    print(f"Finished extracting face embeddings. Total frames processed: {frame_count}")
    return all_face_encodings, frame_faces


def encode_face(face_image, face_locations):
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    face_encodings = face_recognition.face_encodings(face_image, face_locations)
    return face_encodings[0] if face_encodings else None

def cluster_faces(face_encodings):
    print("Clustering face embeddings...")
    print(face_encodings)
    dbscan = DBSCAN(eps=0.5, min_samples=2).fit(face_encodings)
    unique_labels = set(dbscan.labels_)
    print(f"Number of unique faces identified: {len(unique_labels)}")
    return dbscan.labels_, unique_labels
    # return [], []

uuid_to_blur = ''

def save_embeddings_with_uuid(face_encodings, labels):
    face_encodings = np.array(face_encodings)

    print("Saving face embeddings with UUIDs...")


    print('save_embeddings_with_uuid labels')
    print(labels)

    print('face_encodings')
    print(face_encodings)

    face_data = {}
    for label in set(labels):
        if label == -1:
            continue  # Ignore noise points
        uuid_str = str(uuid4())
        print(f"UUID for label {label}: {uuid_str}")
        print(f"Condition: {np.where(labels == label)}")
        print(f"Mean encoding for label {label}: {np.mean(face_encodings[np.where(labels == label)], axis=0)})")
        face_data[uuid_str] = np.mean(face_encodings[np.where(labels == label)], axis=0)

    # TODO: Remove the whole uuid_to_blur rng
    # Pick a random UUID to blur from face_data
    global uuid_to_blur

    if len(face_data.keys()) > 0:
        uuid_to_blur = list(face_data.keys())[0]

    print("face_data")
    print(face_data)

    with open('face_embeddings.pkl', 'wb') as f:

        pickle.dump(face_data, f)
    print("Face embeddings saved successfully.")
    return face_data

def blur_faces_in_video(input_video_path, output_video_path, blur_face_ids):
    print(f"Blurring faces in video: {input_video_path}")
    with open('face_embeddings.pkl', 'rb') as f:
        face_data = pickle.load(f)

    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        print(f"Processing frame {frame_count}")

        face_locations = extract_face_locations_mediapipe(frame)
        for tup in face_locations:
            (left, top, width, height) = tup
            face = frame[top:top + height, left:left + width]
            encoding = encode_face(frame, face_locations)
            if encoding is not None:
                for uuid, stored_encoding in face_data.items():
                    if np.linalg.norm(encoding - stored_encoding) < 0.7 and uuid in blur_face_ids:
                        print('face roi blurred')

                        if face is None or face.size == 0:
                            continue

                        blurred_face = cv2.GaussianBlur(face, (99, 99), 30)
                        frame[top:top + height, left:left + width] = blurred_face

        out.write(frame)

    cap.release()
    out.release()
    print(f"Face blurring completed. Total frames processed: {frame_count}")
    cv2.destroyAllWindows()

all_face_encodings, frame_faces = extract_face_embeddings(_VIDEO_PATH)
labels, unique_labels = cluster_faces(all_face_encodings)
face_data = save_embeddings_with_uuid(all_face_encodings, labels)

print('labels, unique_labels')
print(labels, unique_labels)
print('uuid_to_blur')
print(uuid_to_blur)

blur_face_ids = [uuid_to_blur]  # Example UUID to blur
blur_faces_in_video(_VIDEO_PATH, f'output_video_{uuid.uuid4()}.mp4', blur_face_ids)
