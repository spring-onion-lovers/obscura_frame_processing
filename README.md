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

The ultimate test would be to see if the script can recognize the face that was saved out of a frame of multiple faces.

Result
> Not accurate. Theres a chance that other faces are recognized as the saved face.
> 
```bash
python process_webcam.py
```

### Process Video
At this point of time, running this script will process a video and attempt to blur out the first face it recognizes in the video.

#### How it (theoretically) works
1. Initialize a makeshift vector store.
2. Process the video frame by frame and detect all faces in each frame.
3. Turn each face into an embedding.
4. Check if the faces are in the vector store by running a similarity check between the embedding and all the existing embeddings that is in storage.
5. For faces that do not pass the similarity check, add them to the vector store.
6. Now that the storage contains the embeddings of all the **unique** faces in the video, and each of the embedding (or face) has a unique UUID, we can now blur out the faces using the UUID.

> Note that the UUID in the script is random. It will take the first embedding found in the storage.
7. Now, the script will process each video again, frame by frame, and detect all the faces in each frame.
8. For each of the faces detected in each frame, make an embedding.
9. Check if the embedding of the detected faces matches against the embedding by the UUID (requires looking up the storage).
10. If the embedding matches, blur out the face in the frame.

Results:
> Not accurate. I know :( Such sophisticated logic and steps but still inaccurate.


How it works is that the script will process the video frame by frame, and for each frame will tag detected faces. Each unique face is given a UUID.
```bash
python process_video.py
```