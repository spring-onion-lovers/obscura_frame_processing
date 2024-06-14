import uuid
import numpy as np

'''
    FaceEmbeddingStorage is a class for storing face embeddings
    and other helper methods related to facial embeddings
'''


class FaceEmbeddingStorage:
    def __init__(self, threshold=0.6, debug_mode=False):
        self.embeddings = {}
        self.threshold = threshold
        self.selected_faces = []

    def add_embedding(self, embedding):
        found = self.__check_if_embedding_exists__(embedding)

        if found is False:
            # Assign a new id and set it in the embedding
            embedding_id = uuid.uuid4()
            self.embeddings[embedding_id] = embedding

    @staticmethod
    def calculate_embedding_similarity(e1, e2, threshold):
        """
        Return a Boolean value if the two embeddings are similar within the threshold
        :param e1:
        :param e2:
        :param threshold:
        :return:
        """
        return np.linalg.norm(e1 - e2) < threshold

    def __check_if_embedding_exists__(self, embedding):
        found = False

        for existing_embedding in self.embeddings.values():
            if self.calculate_embedding_similarity(e1=existing_embedding, e2=embedding, threshold=self.threshold):  # threshold for face match
                found = True

        return found

    def get_embeddings(self):
        return self.embeddings.values()

    def get_embeddings_id(self):
        return self.embeddings.keys()

    def get_total_embeddings(self):
        return len(self.embeddings.values())
