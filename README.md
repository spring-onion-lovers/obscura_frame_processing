# Frame Processing

## Important
You MUST install

[//]: # (- `tesseract` [installed on your local machine]&#40;https://tesseract-ocr.github.io/tessdoc/Installation.html&#41; – `pytesseract` is just a library wrapper around the `tesseract` binary.`)
- `CMake` [installed on your local machine](https://cmake.org/download/) – `dlib` and `face-recognition` requires `CMake` to build the library.

## Usage
```bash
# Install the required packages from requirement.txt
pip install -r requirements.txt

# Process Webcam
python process_webcam.py

# Process Single Image
> Coming soon!