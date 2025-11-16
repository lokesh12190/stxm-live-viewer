# stxm-live-viewer
🔬 STXM Live Viewer - High-performance real-time visualization application for  Scanning Transmission X-ray Microscopy (STXM) beamline experiments at DESY.  Handles large detector frames (up to 4000×4000) at high frame rates (100+ FPS)  with comprehensive analysis and export capabilities.

# STXM Live Visualization Application

A high-performance real-time visualization tool for Scanning Transmission X-ray Microscopy (STXM) data acquisition, developed for DESY beamline experiments.

![Version](https://img.shields.io/badge/version-9.0-blue)
![Python](https://img.shields.io/badge/python-3.9+-green)
![License](https://img.shields.io/badge/license-MIT-orange)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Performance Optimization](#performance-optimization)
- [Export Formats](#export-formats)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Architecture](#architecture)
- [Contributing](#contributing)

---

## 🔬 Overview

This application provides real-time visualization of STXM data streams from beamline experiments. It receives detector frames and metadata via ZeroMQ (0MQ) and displays:

1. **Detector Frames**: Live view of 2D detector data (up to 4000×4000 pixels)
2. **STXM Image**: Reconstructed scan image where each pixel intensity represents the sum of all detector pixels at that scan position

### Key Capabilities

- ✅ **High-speed acquisition**: Handles 100+ frames per second
- ✅ **Large detector support**: Optimized for frames up to 4000×4000 pixels
- ✅ **Low-resource friendly**: Runs on office-grade laptops
- ✅ **Real-time analysis**: ROI statistics, crosshair pixel inspection
- ✅ **Comprehensive export**: Multiple file formats (NPZ, CSV, TIFF, PNG)

---

## ✨ Features

### Display & Visualization
- **Dual view**: Detector frame + STXM image side-by-side
- **Colormap selection**: Viridis (colored) and Grayscale (B&W)
- **Auto-scaling**: Dynamic intensity range adjustment
- **Frame averaging**: Noise reduction through temporal averaging
- **Downsampling**: 1-10× for performance optimization
- **Display throttling**: Update every N frames for responsiveness

### Analysis Tools
- **ROI (Region of Interest)**: 
  - Draggable/resizable rectangle
  - Real-time statistics (mean, sum, max, min, std)
  - Save/load ROI positions
  - Export ROI measurement history
- **Crosshair cursor**: Hover to see pixel coordinates and values
- **FPS monitoring**: Instantaneous and average frame rate

### Data Management
- **Pause/Resume**: Control acquisition without disconnecting
- **Snapshots**: Capture current state (frame + STXM)
- **Clear views**: Reset displays and data
- **Export options**:
  - STXM data (NPZ, CSV, TXT)
  - Current detector frame (NPY, TIFF, PNG)
  - All frames from scan (NPZ compressed)
  - All snapshots (NPZ)
  - ROI statistics (CSV, JSON)

---

## 📦 Requirements

### System Requirements
- **OS**: Windows, Linux, or macOS
- **Python**: 3.9 or higher
- **RAM**: Minimum 4 GB (8 GB recommended for 4000×4000 frames)
- **CPU**: Multi-core processor recommended

### Python Packages

```txt
numpy>=1.24.3
pyzmq>=27.1.0
numba>=0.60.0
pyqtgraph>=0.13.0
PyQt5>=5.15.0
pytest>=8.0.0 (for testing)
pytest-qt>=4.0.0 (for testing)
Pillow>=9.0.0 (optional, for TIFF/PNG export)
```

---

## 🚀 Installation

### Step 1: Clone or Download the Project

```bash
cd your-workspace
# If using git:
git clone <repository-url>
cd coding-assignment
```

### Step 2: Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv desy
desy\Scripts\activate
```

**Linux/macOS:**
```bash
python3 -m venv desy
source desy/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**For full functionality (including TIFF/PNG export):**
```bash
pip install -r requirements.txt
pip install Pillow
```

---

## 🎯 Quick Start

### Option 1: Manual Start (2 Terminals)

**Terminal 1 - Start Data Emulator:**
```bash
# Activate environment
desy\Scripts\activate  # Windows
source desy/bin/activate  # Linux/macOS

# Run emulator
python emulate_data_stream.py
```

**Terminal 2 - Start Viewer:**
```bash
# Activate environment
desy\Scripts\activate  # Windows
source desy/bin/activate  # Linux/macOS

# Run viewer
python stxm_viewer.py
```

### Option 2: Windows Batch File (Recommended for Windows)

Double-click `run_both.bat` - this will automatically:
1. Open two command windows
2. Activate the virtual environment in both
3. Start the emulator in the first window
4. Start the viewer in the second window

---

## 📖 Detailed Usage

### Data Emulator Configuration

Edit `emulate_data_stream.py` to customize:

```python
# At the bottom of the file:
if __name__ == "__main__":
    publisher = PublisherEmulator(
        detector_shape=(4000, 4000),  # Frame size (200-4000)
        sleep_between_scans_s=5       # Delay between scans
    )
    publisher.run()
```

**For testing different scenarios:**

```python
# Small frames, fast acquisition (testing):
detector_shape=(200, 200)

# Medium frames (typical use):
detector_shape=(1000, 1000)

# Large frames (stress test):
detector_shape=(4000, 4000)
```

### Viewer Interface

#### 1. Display Settings Panel
- **Downsample (1-10)**: Reduces displayed resolution
  - `1` = Full resolution
  - `4` = Quarter resolution (recommended for 2000×2000+)
  - `8` = Eighth resolution (for 4000×4000 on slow laptops)

- **Update Every (1-20 frames)**: Display refresh rate
  - `1` = Every frame (default)
  - `5` = Every 5th frame (recommended for high FPS)
  - `10` = Every 10th frame (for maximum responsiveness)

- **Colormap**: Visual appearance
  - `viridis` = Blue-green-yellow (default, good contrast)
  - `grayscale` = Black to white (publications, printing)

- **Auto-scale**: ☑ Enabled = Automatic intensity scaling per frame
  - Disable for consistent scaling across frames

- **Average frames**: ☑ Enabled = Temporal averaging for noise reduction
  - Uses last 10 frames

#### 2. Analysis Tools Panel
- **Enable ROI**: Create draggable rectangle on detector frame
  - Drag to move, drag corners to resize
  - Real-time statistics displayed below
  
- **Save ROI**: Export current ROI position to JSON file
- **Load ROI**: Import saved ROI position
- **Crosshair**: Show pixel coordinates on mouse hover
- **Export ROI Data**: Save statistics history (CSV/JSON)

#### 3. Acquisition & Export Panel
- **⏸ Pause/▶ Resume**: Stop/start data reception (preserves connection)
- **🗑 Clear Views**: Reset all displays and stored data
- **📷 Snapshot**: Capture current frame + STXM for comparison
- **💾 Export STXM**: Save STXM image with metadata
- **💾 Export Frame**: Save current detector frame
- **💾 Export All**: Save all frames from current scan
- **💾 Export Snapshots**: Save all captured snapshots

---

## ⚡ Performance Optimization

### For Different Hardware Scenarios

#### 🐌 Slow Laptop + Large Frames (4000×4000)
```
Downsample: 8
Update Every: 10 frames
☐ Auto-scale (off)
☐ Average frames (off)
☐ Enable ROI (off)
```
**Expected**: 5-10 FPS display, all data processed correctly

#### 💻 Standard Laptop + Medium Frames (1000×1000)
```
Downsample: 2-4
Update Every: 5 frames
☑ Auto-scale
☐ Average frames
```
**Expected**: 20-30 FPS display

#### 🚀 High-Performance PC + Any Frame Size
```
Downsample: 1
Update Every: 1 frame
☑ Auto-scale
☑ Average frames (optional)
☑ Enable ROI (optional)
```
**Expected**: 50+ FPS for 200×200, 10-20 FPS for 4000×4000

### Performance Tips

1. **Window Not Responding?**
   - Increase "Update Every" to 5-10
   - Increase "Downsample" to 4-8
   - Disable "Auto-scale"
   - Disable "ROI" when not needed

2. **Slow FPS but smooth UI?** = Normal! Display update rate ≠ data processing rate
   - All data is still processed and saved correctly
   - Display throttling prevents UI freezing

3. **Memory Usage Growing?**
   - All frames are stored in memory during scan
   - Click "Clear Views" between scans
   - For very long scans, consider restarting application

---

## 💾 Export Formats

### STXM Data

**NPZ (Recommended)**
```python
data = np.load('stxm_data.npz')
stxm_image = data['stxm_data']
x_coords = np.linspace(data['x_start'], data['x_stop'], data['x_num'])
y_coords = np.linspace(data['y_start'], data['y_stop'], data['y_num'])
```

**CSV** (Excel/Spreadsheet compatible)
```python
import numpy as np
stxm_data = np.loadtxt('stxm_data.csv', delimiter=',')
```

### Detector Frames

**NPY** (Lossless, Python native)
```python
frame = np.load('detector_frame.npy')
```

**TIFF** (16-bit, compatible with ImageJ/Fiji)
```python
from PIL import Image
img = Image.open('detector_frame.tiff')
frame = np.array(img)
```

**PNG** (8-bit, universal compatibility)
```python
from PIL import Image
img = Image.open('detector_frame.png')
```

### All Frames

```python
data = np.load('all_frames.npz', allow_pickle=True)
frames = data['frames']  # Shape: (num_frames, height, width)
metadata = data['metadata'].item()
print(f"Scan: {metadata['x_num']}×{metadata['y_num']}")
```

### Snapshots

```python
data = np.load('snapshots.npz', allow_pickle=True)
frame_0 = data['snapshot_0_frame']
stxm_0 = data['snapshot_0_stxm']
timestamps = data['timestamps']
```

### ROI Statistics

**CSV**
```csv
timestamp,mean,sum,max,min,std,shape
1700000000.0,0.52,2048.5,0.98,0.12,0.15,"(100, 100)"
```

**JSON**
```json
[
  {
    "timestamp": 1700000000.0,
    "mean": 0.52,
    "sum": 2048.5,
    "max": 0.98,
    "min": 0.12,
    "std": 0.15,
    "shape": [100, 100]
  }
]
```

---

## 🧪 Testing

### Run All Tests

```bash
pytest test_viewer.py -v
```

### Run Specific Test

```bash
pytest test_viewer.py::test_initial_state -v
```

### Run with Coverage

```bash
pip install pytest-cov
pytest test_viewer.py --cov=stxm_viewer --cov-report=html
```

View coverage report: `htmlcov/index.html`

### Test Results

✅ **42 tests** covering:
- Initial state & UI elements
- Metadata processing
- Frame reception & STXM calculation
- Downsampling & colormaps
- ROI functionality
- Pause/Resume
- Snapshots & Clear views
- All export formats
- Edge cases & error handling
- Full workflow integration

---

## 🔧 Troubleshooting

### Common Issues

#### 1. "Connection Error" or "Waiting for data..."
**Cause**: Emulator not running or wrong ports

**Solution**:
```bash
# Check if emulator is running
# You should see: "ScanMD(...)" output

# Restart emulator
python emulate_data_stream.py
```

#### 2. "Module not found" errors
**Cause**: Missing dependencies

**Solution**:
```bash
pip install -r requirements.txt
```

#### 3. Window freezes / "Not Responding"
**Cause**: CPU overload from large frames or high FPS

**Solution**:
- Increase "Update Every" to 5-10
- Increase "Downsample" to 4-8
- Disable "Auto-scale"

#### 4. Very low FPS with small frames
**Cause**: Slow exposure time in emulator

**Solution**: Edit `emulate_data_stream.py`:
```python
MIN_DWELL_TIME = 0.001  # Faster frames
```

#### 5. Export fails with "Permission Error" (Windows)
**Cause**: File locked by another process

**Solution**:
- Close any programs viewing the file
- Try a different filename
- Restart application

#### 6. TIFF/PNG export not available
**Cause**: Pillow not installed

**Solution**:
```bash
pip install Pillow
```

#### 7. Tests fail with file permission errors
**Cause**: Windows file locking

**Solution**: Tests are fixed in latest version. If issues persist:
```bash
# Close all Python processes
# Restart terminal
pytest test_viewer.py -v
```

---

## 🏗️ Architecture

### System Overview

```
┌─────────────────────┐         ┌──────────────────────┐
│  Data Emulator      │  ZMQ    │  Visualization App   │
│  (emulate_data_     │ ──────> │  (stxm_viewer.py)    │
│   stream.py)        │         │                      │
│                     │         │  ┌────────────────┐  │
│  - Generate frames  │ Port    │  │ ZMQDataReceiver│  │
│  - Send metadata    │ 50001   │  │  (Background   │  │
│  - Simulate timing  │         │  │   Thread)      │  │
│                     │ Port    │  └────────┬───────┘  │
│                     │ 50002   │           │          │
└─────────────────────┘         │           ▼          │
                                │  ┌────────────────┐  │
                                │  │  Main Window   │  │
                                │  │  (Qt GUI)      │  │
                                │  │                │  │
                                │  │  - Frame view  │  │
                                │  │  - STXM view   │  │
                                │  │  - Controls    │  │
                                │  │  - Export      │  │
                                │  └────────────────┘  │
                                └──────────────────────┘
```

### Key Components

1. **ZMQDataReceiver (QThread)**
   - Runs in background thread
   - Non-blocking ZMQ polling
   - Emits signals to main thread

2. **STXMVisualizationApp (QMainWindow)**
   - Main GUI window
   - Handles all user interactions
   - Processes and displays data

3. **Data Flow**
   ```
   Emulator → ZMQ → Receiver Thread → Signal → Main Thread → Display
   ```

### Data Processing Pipeline

```
Incoming Frame
    ↓
Store in all_frames[]
    ↓
Calculate STXM intensity (sum)
    ↓
Update STXM data array
    ↓
Should update display? (throttling check)
    ├─ Yes → Process display frame
    │         ├─ Downsample
    │         ├─ Average (if enabled)
    │         └─ Update image widget
    └─ No → Skip display update
```

---

## 📝 File Structure

```
coding-assignment/
├── desy/                      # Virtual environment
├── emulate_data_stream.py     # Data simulator (5 KB)
├── stxm_viewer.py             # Main application (35 KB)
├── test_viewer.py             # Test suite (23 KB)
├── requirements.txt           # Dependencies (1 KB)
├── README.md                  # This file (1 KB → enhanced)
└── run_both.bat               # Windows launcher (1 KB)
```

---

## 🤝 Contributing

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd coding-assignment

# Create environment
python -m venv desy
source desy/bin/activate  # or desy\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
pip install pytest pytest-qt pytest-cov

# Run tests
pytest test_viewer.py -v
```

### Code Style

- Follow PEP 8
- Add docstrings to functions
- Update tests for new features
- Ensure all tests pass before committing

---

## 📄 License

MIT License - See LICENSE file for details

---
## 📚 Additional Resources

### ZeroMQ (0MQ)
- Documentation: https://zeromq.org/
- Python bindings: https://pyzmq.readthedocs.io/

### PyQtGraph
- Documentation: https://pyqtgraph.readthedocs.io/
- Examples: https://github.com/pyqtgraph/pyqtgraph/tree/master/examples

### STXM Background
- STXM Technique: https://en.wikipedia.org/wiki/Scanning_transmission_X-ray_microscopy
- DESY: https://www.desy.de/

---

## 📊 Performance Benchmarks

### Typical Performance (on standard laptop)

| Frame Size | Emulator FPS | Display FPS | Settings |
|------------|--------------|-------------|----------|
| 200×200    | ~100         | ~50         | Default  |
| 1000×1000  | ~50          | ~25         | DS=2     |
| 2000×2000  | ~20          | ~10         | DS=4     |
| 4000×4000  | ~5           | ~3          | DS=8     |

*DS = Downsample factor*
*Update Every = 1 frame*

---

## ✅ Feature Checklist

- [x] Dual view display (Detector + STXM)
- [x] Live data streaming via ZMQ
- [x] Support for large frames (4000×4000)
- [x] High frame rate handling (100+ FPS)
- [x] Performance optimization (downsampling, throttling)
- [x] ROI analysis with statistics
- [x] Multiple colormap options
- [x] Pause/Resume functionality
- [x] Snapshot capture
- [x] Comprehensive export options (7 formats)
- [x] Save/Load ROI positions
- [x] Crosshair pixel inspection
- [x] FPS monitoring
- [x] Comprehensive test suite (42 tests)
- [x] Windows batch launcher
- [x] Complete documentation

---

## 🎓 Quick Reference Card

### Essential Keyboard Shortcuts
(Future enhancement)

### Essential Commands
```bash
# Start everything
run_both.bat  # Windows
# or manually in 2 terminals

# Run tests
pytest test_viewer.py -v

# Install with optional features
pip install -r requirements.txt Pillow
```

### Default Ports
- Metadata: `tcp://127.0.0.1:50001`
- Data: `tcp://127.0.0.1:50002`

---
