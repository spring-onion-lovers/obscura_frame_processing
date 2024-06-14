# Frame Processor

## Important
You MUST install
- `tesseract` [installed on your local machine](https://tesseract-ocr.github.io/tessdoc/Installation.html) – `pytesseract` is just a library wrapper around the `tesseract` binary.`

## Usage
```bash
# Install the required packages from requirement.txt
pip install -r requirements.txt

# Process Single Video
python process_video.py /path/to/video.mp4

# Process Single Image
> Coming soon!
```

## Packages needed (from requirement.txt)
```bash
pillow
matplotlib
pandas
opencv-python
pytesseract
```