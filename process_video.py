import pytesseract
import cv2 as cv
import sys
from frame_methods import convert_to_grayscale, ocr_frame, otsu_thresh, draw_on_image

[filepath] = sys.argv[1:]

# Check if the user has provided a video file
if len(sys.argv) < 2:
    print("Usage: python process_video.py <path_to_video>")
    sys.exit(1)

cap = cv.VideoCapture(filepath)
input_fps = cap.get(cv.CAP_PROP_FPS)
out = None
fourcc = cv.VideoWriter_fourcc(*'mp4v')
psm = 12
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/Cellar/tesseract/5.4.0/bin/tesseract'

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Convert the frame to grayscale
    gray = convert_to_grayscale(frame)

    # Apply otsu thresholding
    th3 = otsu_thresh(gray)

    # Show th3
    # cv.imshow('frame', th3)

    # Perform OCR and get the dataframe
    df = ocr_frame(th3, psm)

    # Filter out rows with low confidence and empty text
    df_filtered = df[(df['conf'] > 0)]

    # Calculate and print the progress
    current_frame = cap.get(cv.CAP_PROP_POS_FRAMES)
    total_frames = cap.get(cv.CAP_PROP_FRAME_COUNT)
    progress = (current_frame / total_frames) * 100
    print(f"Progress: {progress:.2f}%")


    # Draw bounding boxes and text on the image
    for index, row in df_filtered.iterrows():
        draw_on_image(frame, row)


    cv.imshow('image', th3)
    cv.imshow('frame', frame)

    if out is None:
        out = cv.VideoWriter(f"output_{str('--psm ' + str(psm)).replace('-','').replace(' ', '_')}.mp4", fourcc, input_fps, (frame.shape[1], frame.shape[0]))

    out.write(frame)

    # Quit the app
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
cv.destroyAllWindows()
