# SAHI YOLO RTDETR-X Video Inference

This project is an experimental implementation for running object detection on videos using [SAHI](https://github.com/obss/sahi) and [YOLO RTDETR-X](https://docs.ultralytics.com/pt/models/rtdetr/#usage-examples) from [Ultralytics](https://ultralytics.com/).

## Features

- Runs inference on video files
- Uses slicing for improved detection on large images
- Filters out unwanted classes (e.g., airplanes, drones)
- Annotates detected objects and displays FPS/count

## Model Download

You must manually download the model file `rtdetr-x.pt` and place it in the project folder. This file is not included in the repository. You can download it from the official Ultralytics website: https://docs.ultralytics.com/pt/models/rtdetr/#usage-examples or from the appropriate source for RTDETR-X models.

## Usage
```sh
python main.py --source demo.mp4
```


## Requirements

- Python 3.8+
- OpenCV
- NumPy
- SAHI
- Ultralytics YOLO

Install dependencies with:

```sh
pip install -r requirements.txt
```

## License

Experimental use only.

## Credits Video 

GixxerZTM 