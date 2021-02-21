# Goggles

Goggles is a Python tool for selfie/webcam super resolution that first detects where the faces are, then runs the ESRGAN face-specific SR model on those cropped busts.

## Installation

This uses the model from [Face-Super-Resolution](https://github.com/ewrfcas/Face-Super-Resolution), so you must follow all of their installation instructions

```bash
git clone https://github.com/ewrfcas/Face-Super-Resolution
```

## Usage

```bash
python3 image_face_sr.py [image_path]
python3 webcam_face_sr.py
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
