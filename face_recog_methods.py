import cv2
import face_recognition
import numpy as np

from face_embedding_storage import FaceEmbeddingStorage

selected_faces = []


# Mouse callback function to select faces
def mouse_callback(event, x, y, flags, param):
    print('cliclked')

    if event == cv2.EVENT_LBUTTONDOWN:
        global selected_faces

        frame, bboxes, embeddings, storage = param['frame'], param['bboxes'], param['embeddings'], param['storage']

        for idx, bbox in enumerate(bboxes):
            x1, y1, w, h = bbox

            if x1 <= x <= x1 + w and y1 <= y <= y1 + h:

                # loop through embeddings
                for e in embeddings:
                    selected_face_embedding = e
                    found = False

                    for existing_embedding in selected_faces:
                        if FaceEmbeddingStorage.calculate_embedding_similarity(existing_embedding,
                                                                               selected_face_embedding, threshold=0.6):
                            selected_faces.remove(existing_embedding)
                            found = True
                            break
                    if not found:
                        selected_faces.append(selected_face_embedding)
                    break


# Function to compute face embeddings
def compute_face_embedding(frame, bbox):
    x, y, w, h = bbox
    face_image = frame[y:y + h, x:x + w]
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    face_locations = [(0, w, h, 0)]
    face_encodings = face_recognition.face_encodings(face_image, face_locations)
    return face_encodings[0] if face_encodings else None
