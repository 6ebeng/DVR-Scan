# DVR-Scan CUDA Setup Guide for Windows 11

This guide explains how to enable CUDA GPU acceleration in DVR-Scan after installing via MSI on Windows 11.

## Overview

DVR-Scan supports NVIDIA CUDA for GPU-accelerated motion detection using the `MOG2_CUDA` background subtractor. This can significantly speed up video processing on systems with compatible NVIDIA GPUs.

## Requirements

1. **NVIDIA GPU** with CUDA Compute Capability 3.0 or higher
2. **NVIDIA Driver** (latest version recommended)
3. **CUDA Toolkit** 11.x or 12.x
4. **OpenCV with CUDA support** (requires replacing the default OpenCV package)

## Quick Start

### Step 1: Check Your Current Setup

Run the CUDA diagnostics command:

```powershell
dvr-scan --cuda-info
```

Or run the PowerShell setup script:

```powershell
.\scripts\setup_cuda.ps1
```

### Step 2: Install NVIDIA Driver

If not already installed, download and install the latest NVIDIA driver:
- https://www.nvidia.com/drivers

Verify installation:
```powershell
nvidia-smi
```

### Step 3: Install CUDA Toolkit

Download and install CUDA Toolkit from:
- https://developer.nvidia.com/cuda-downloads

**Recommended:** CUDA 12.x for best compatibility

After installation, verify the `CUDA_PATH` environment variable is set:
```powershell
echo $env:CUDA_PATH
```

If not set, add it manually:
```powershell
# For CUDA 12.4 (adjust version as needed)
setx CUDA_PATH "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
```

### Step 4: Install OpenCV with CUDA Support

The standard `opencv-python` package from PyPI does NOT include CUDA support. You need to replace it with a CUDA-enabled build.

#### Option A: Pre-built CUDA Wheel (Recommended)

1. Download the appropriate wheel from:
   https://github.com/cudawarped/opencv-python-cuda-wheels/releases

2. Choose the wheel matching your Python version (e.g., `cp311` for Python 3.11)

3. Uninstall existing OpenCV packages:
   ```powershell
   pip uninstall opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless -y
   ```

4. Install the CUDA wheel:
   ```powershell
   pip install opencv_python_cuda-4.11.0.20250124-cp311-cp311-win_amd64.whl
   ```

#### Option B: Build from Source

For advanced users who need custom configurations:
- See: https://docs.opencv.org/master/d3/d52/tutorial_windows_install.html

### Step 5: Verify CUDA Support

Run the diagnostics again:
```powershell
dvr-scan --cuda-info
```

Or verify directly with Python:
```python
import cv2
print("OpenCV Version:", cv2.__version__)
print("CUDA Devices:", cv2.cuda.getCudaEnabledDeviceCount())
```

### Step 6: Use CUDA in DVR-Scan

Once configured, use the `MOG2_CUDA` subtractor:

**Command Line:**
```powershell
dvr-scan -i video.mp4 -b MOG2_CUDA
```

**Config File (dvr-scan.cfg):**
```ini
[dvr-scan]
bg-subtractor = MOG2_CUDA
```

**GUI Application:**
Settings → Motion → Subtractor → MOG2_CUDA

## Troubleshooting

### CUDA Not Detected

1. Verify NVIDIA driver is installed: `nvidia-smi`
2. Check CUDA_PATH environment variable is set correctly
3. Restart your terminal/PowerShell after setting environment variables

### OpenCV CUDA Not Available

1. Ensure you've uninstalled all existing OpenCV packages
2. Verify you downloaded the correct wheel for your Python version
3. Check that CUDA Toolkit version matches the wheel requirements

### DLL Load Errors

DVR-Scan automatically configures CUDA DLL paths, but if you encounter issues:

1. Add CUDA bin to PATH:
   ```powershell
   $env:PATH += ";$env:CUDA_PATH\bin"
   ```

2. Or add permanently:
   ```powershell
   setx PATH "$env:PATH;$env:CUDA_PATH\bin"
   ```

### Performance Issues

- Ensure your GPU has enough VRAM for the video resolution
- Try reducing video resolution with `-df` (downscale factor)
- Close other GPU-intensive applications

## Files Reference

| File | Description |
|------|-------------|
| `dvr_scan/cuda_setup.py` | CUDA detection and setup utilities |
| `dvr_scan/opencv_loader.py` | Automatic CUDA DLL path configuration |
| `scripts/setup_cuda.ps1` | PowerShell setup script |
| `scripts/setup_cuda.bat` | Batch file setup script |

## Related Documentation

- [DVR-Scan Documentation](https://www.dvr-scan.com/docs)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [OpenCV CUDA Module](https://docs.opencv.org/master/d2/dbc/cuda_intro.html)

## Support

For issues related to DVR-Scan CUDA support:
- GitHub Issues: https://github.com/Breakthrough/DVR-Scan/issues
- Discord: https://discord.gg/69kf6f2Exb
