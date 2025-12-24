# Visual Positioning

This project implements a high-precision 3D positioning system based on computer vision. By combining an Intel RealSense RGB-D depth camera with AprilTag markers, it captures the precise position (X, Y, Z) and orientation (Roll, Pitch, Yaw) of targets in real-time within complex environments.

This technology aims to solve **precise spatial positioning** challenges in robot navigation, automated grasping, and Augmented Reality (AR) scenarios.

![Visual Positioning Demo](assets/visual_positioning.png)
*Figure: Real-time 3D positioning using RealSense camera, showing target coordinates (X, Y, Z) and orientation.*

## Background & Functionality

In automation and intelligent interaction, **positioning** is a core capability. Traditional monocular visual positioning suffers from scale ambiguity, while pure depth cameras often struggle with textureless surfaces.

This project achieves highly reliable **visual positioning** through the following techniques:
1.  **RGB-D Fusion**: Combines feature extraction from color images (AprilTag) with distance measurement from depth images to eliminate scale uncertainty.
2.  **Multi-Face Cube Tracking**: Utilizes the geometric constraints of a cube's multiple faces to output stable center coordinates even under partial occlusion or large rotation angles.
3.  **High-Precision Pose Estimation**: Provides millimeter-level accuracy using the PnP (Perspective-n-Point) algorithm with depth correction.
4.  **Smoothing & Continuity**: Built-in rotation continuity algorithms ensure smooth transitions of coordinates and poses during tag switching, preventing sudden jumps.

## Core Features

- **Real-Time Positioning**: High-speed tag detection based on `pupil_apriltags`.
- **Depth Enhancement**: Uses RealSense depth streams to correct distance estimates, significantly improving Z-axis accuracy.
- **Multi-Face Fusion**: Automatically fuses observations from different faces of the cube to calculate a unified physical center.
- **Data Communication**: Supports real-time data broadcasting via **Shared Memory** or **TCP Socket**.
- **Visual Debugging**: Real-time display of coordinate axes, tag IDs, and pose parameters.
- **Modular Design**: Separation of configuration and code for easy integration into other systems.

## Installation

1. Clone the repository or download the source code.
2. Install the required Python libraries using the provided requirements file:

```bash
pip install -r requirements.txt
```

Key dependencies include:
- `pyrealsense2`: Intel RealSense driver
- `pupil-apriltags`: High-performance AprilTag detection library
- `opencv-python`: Image processing and visualization
- `numpy` & `scipy`: Math operations and spatial transformations

## Usage

### Command Line Execution

Start the positioning system directly using the example script:

```bash
python examples/run.py
```

Upon startup, the program will prompt you to select a communication mode:
- `s`: **Shared Memory Mode** - Suitable for high-performance local communication.
- `t`: **TCP Socket Mode** (Default) - Suitable for inter-process or cross-device communication.
- `n`: **No Communication Mode** - Only displays the positioning visualization locally.
- `m`: **Manual Configuration Mode**

### Configuration

System parameters are managed via `config/config.yaml` and can be adjusted without modifying the code:

- **Camera Parameters**: Resolution, FPS, Intrinsic matrices, distortion coefficients.
- **Target Settings**: Physical size of the Cube, Tag size.
- **Communication Settings**: TCP port, shared memory filename.
- **Algorithm Parameters**: Detector sharpening, decimation, etc.

## Output & Communication

### 1. Positioning Data Files
The system automatically logs data during runtime:
- `pose_matrix3.txt`: Records historical pose matrices.
- `server.log`: Runtime logs.

### 2. Real-Time Data Stream (IPC)
The system broadcasts positioning data in real-time via IPC in JSON format:

```json
{
  "t": 1703234567.123456,          // Timestamp (seconds)
  "pose_matrix": [                 // 4x4 Transformation Matrix
    [0.987, -0.123, 0.012, 104.5], // R11, R12, R13, X (mm)
    [0.123, 0.987, -0.023, 49.2],  // R21, R22, R23, Y (mm)
    [-0.012, 0.023, 0.999, 441.3], // R31, R32, R33, Z (mm)
    [0.0, 0.0, 0.0, 1.0]
  ]
}
```
> **Note**: The translation vector (X, Y, Z) in the matrix is in **millimeters (mm)**.

#### TCP Mode
- Default Port: `8081`
- Clients receive a continuous data stream upon connection.

#### Shared Memory Mode
- Default Map File: `apriltag_pose_data.tmp`
- Memory Layout: `[Data Length (4B)][Timestamp (8B)][JSON Data (N Bytes)]`

## Project Structure

```
├── config/
│   └── config.yaml          # Configuration file
├── src/
│   ├── app.py               # Main application logic
│   ├── pose.py              # Core pose estimation & coordinate transformation
│   ├── camera.py            # Camera driver wrapper
│   ├── ipc.py               # Inter-process communication module
│   └── ...
├── examples/
│   └── run.py               # Startup script
├── requirements.txt         # Dependency list
└── README.md                # Project documentation
```

## License

MIT License
