# Tennis Ball Detection System

This program detects tennis balls in a video stream and tracks impacts on a white background, mapping them to a virtual screen.

## Prerequisites

- Python 3.x
- OpenCV (`cv2`)
- NumPy

## Installation

1. Install required packages:
   ```
   pip install opencv-python numpy
   ```

2. Download or clone this repository.

## Usage

1. Place your tennis ball video file in the same directory as the script (default: "Tennis_ball_wall_12.mp4").

2. Run the script:
   ```
   python Tennis_ball_det_white_BG.py
   ```

3. Two windows will appear:
   - **Live Feed**: Shows the input video with detected tennis ball and white zone
   - **Virtual Screen**: Shows a white canvas where impacts are marked as black dots

4. Press 'q' to exit the program.

## Configuration

You can adjust the following parameters in the code:

- **Video source**: Change `cap = cv2.VideoCapture("Tennis_ball_wall_12.mp4")` to use a different video file or camera source.
  - For webcam, use `cap = cv2.VideoCapture(0)`

- **Color detection settings**: Modify the HSV color ranges in the `detect_ball` and `detect_white_zone` functions to better match your specific tennis ball color and background.

- **Hit velocity threshold**: Adjust `hit_velocity_threshold = 95` to make the impact detection more or less sensitive.

## How It Works

1. The program detects a tennis ball based on its yellow-green color.
2. It identifies the white background area in the frame.
3. When the ball's velocity changes significantly (indicating an impact) and the impact occurs within the white zone, the system maps this point to a virtual screen.
4. Each impact is visualized as a black dot on the virtual screen.

## Troubleshooting

- If the ball isn't being detected, try adjusting the HSV color ranges in the `detect_ball` function.
- If impacts aren't being registered correctly, adjust the `hit_velocity_threshold` value.
- For better detection, ensure good lighting conditions and a clear contrast between the tennis ball and background. 