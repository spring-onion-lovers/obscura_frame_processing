# Frame Processing

## Important
You MUST install (outside of the `requirements.txt` file):

[//]: # (- `tesseract` [installed on your local machine]&#40;https://tesseract-ocr.github.io/tessdoc/Installation.html&#41; – `pytesseract` is just a library wrapper around the `tesseract` binary.`)
- `CMake` [installed on your local machine](https://cmake.org/download/) – `dlib` and `face-recognition` requires `CMake` to build the library.

## Usage
1. Remember to install the required packages from `requirements.txt` before running the script (by running `pip install -r requirements.txt`).
2. According to the file you want to run, open up the file in a code editor and change the variables with underscores at the front (e.g. `_VIDEO_PATH` or `_THRESHOLD`) to your liking.
3. Run one of the scripts below

### Process Webcam
Running this script will open up your webcam and process the frames in real-time. The script will detect faces in the frame and draw a rectangle around them. The green rectangles are the faces that are recognized, while the red rectangles are the faces that are not "saved". To save a face, click on one of the rectangles and it should change to "red".
```bash
python process_webcam.py
```

Expected Outcome:
> The face that was saved should be recognized and a green rectangle should be drawn around it. Even if there are multiple faces in the frame, the saved face should be recognized.

Result:
> Not accurate. There's a chance that other faces are recognized as the saved face.

### Process Video
At this point of time, running this script will process a video and attempt to blur out the first face it recognizes in the video.
```bash
python process_video.py
```

#### How it (theoretically) works
1. Process the video frame by frame and detect all faces in each frame.
2. Turn each face into an embedding and add it to an array.
3. After running the video frame-by-frame, we use DBSCAN to cluster the embeddings. (Basically, we're giving each face's embedding into DBSCAN and it will group similar embeddings together). This means that if there are multiple faces of the same person, they will be grouped together and be assigned a label.
4. Using the labels, we can now know how many unique people are in the video.
5. For each label, we will assign a UUID and also save the embedding to a storage.
6. This UUID will be used for the user to blur out the face in the video at a later step.
> Note that the UUID in the script is random. It will take the first embedding/face/person seen.
7. Now, the script will process each video again, frame by frame, and detect all the faces in each frame.
8. For each of the faces detected in each frame, convert it into an embedding.
9. Compare the embedding with the embeddings in the storage and see if the embedding matches the to-be-blurred UUID by measuring the Euclidean distance.
10. If the embedding matches, blur out the face in the frame.

Expected Outcome:
> Only 1 face is blurred out in the video. The face that was saved in the first step. But since it is random, only 1 face will be blurred out.

Results:
> Not accurate. I know :( Such sophisticated logic and steps but still inaccurate.