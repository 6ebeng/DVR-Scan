@echo off
REM DVR-Scan CUDA Setup Script for Windows 11
REM This script helps configure CUDA support for DVR-Scan after MSI installation
REM 
REM Requirements:
REM   1. NVIDIA GPU with CUDA support
REM   2. NVIDIA Driver installed
REM   3. CUDA Toolkit installed (https://developer.nvidia.com/cuda-downloads)
REM   4. OpenCV with CUDA support (see instructions below)

echo ============================================================
echo           DVR-Scan CUDA Setup for Windows 11
echo ============================================================
echo.

REM Check for administrator privileges
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo NOTE: Running without admin privileges. Some features may be limited.
    echo.
)

REM Check if Python is available
python --version >nul 2>&1
if %errorLevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.10+ from https://python.org
    goto :end
)

REM Check NVIDIA driver
echo Checking NVIDIA Driver...
nvidia-smi >nul 2>&1
if %errorLevel% neq 0 (
    echo WARNING: NVIDIA driver not detected.
    echo          Please install NVIDIA drivers from https://www.nvidia.com/drivers
    echo.
) else (
    echo NVIDIA Driver: OK
    nvidia-smi --query-gpu=driver_version,name --format=csv,noheader 2>nul
    echo.
)

REM Check CUDA installation
echo Checking CUDA Toolkit...
if defined CUDA_PATH (
    if exist "%CUDA_PATH%\bin\nvcc.exe" (
        echo CUDA Toolkit: OK
        echo Path: %CUDA_PATH%
        "%CUDA_PATH%\bin\nvcc.exe" --version 2>nul | findstr "release"
    ) else (
        echo WARNING: CUDA_PATH is set but nvcc.exe not found.
    )
) else (
    REM Try to find CUDA automatically
    if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" (
        for /f "delims=" %%i in ('dir /b /ad /o-n "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*" 2^>nul') do (
            echo CUDA Toolkit found: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\%%i
            echo Consider setting CUDA_PATH environment variable:
            echo   setx CUDA_PATH "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\%%i"
            goto :cuda_found
        )
    )
    echo WARNING: CUDA Toolkit not found.
    echo          Please install from https://developer.nvidia.com/cuda-downloads
    echo          Recommended: CUDA 12.x
)
:cuda_found
echo.

REM Check OpenCV CUDA support
echo Checking OpenCV CUDA Support...
python -c "import cv2; print('OpenCV Version:', cv2.__version__); cuda_count = cv2.cuda.getCudaEnabledDeviceCount() if hasattr(cv2, 'cuda') else 0; print('CUDA Devices:', cuda_count); exit(0 if cuda_count > 0 else 1)" 2>nul
if %errorLevel% neq 0 (
    echo.
    echo WARNING: OpenCV does not have CUDA support enabled.
    echo.
    echo To enable CUDA support, you need to replace the standard OpenCV package
    echo with a CUDA-enabled build. There are two options:
    echo.
    echo OPTION 1: Install Pre-built CUDA Wheel [Recommended]
    echo ----------------------------------------------------
    echo 1. Download from: https://github.com/cudawarped/opencv-python-cuda-wheels/releases
    echo 2. Run the following commands:
    echo    pip uninstall opencv-python opencv-python-headless opencv-contrib-python -y
    echo    pip install [downloaded-wheel-file].whl
    echo.
    echo OPTION 2: Build OpenCV from Source
    echo -----------------------------------
    echo See: https://docs.opencv.org/master/d3/d52/tutorial_windows_install.html
    echo.
) else (
    echo OpenCV CUDA: OK
)
echo.

REM Run DVR-Scan CUDA diagnostics
echo ============================================================
echo Running DVR-Scan CUDA Diagnostics...
echo ============================================================
python -m dvr_scan.cuda_setup 2>nul
if %errorLevel% neq 0 (
    echo DVR-Scan CUDA diagnostics module not available.
    echo Run: python -c "from dvr_scan.cuda_setup import print_cuda_diagnostics; print_cuda_diagnostics()"
)

echo.
echo ============================================================
echo                    Setup Complete
echo ============================================================
echo.
echo If CUDA is properly configured, run DVR-Scan with:
echo   dvr-scan -i video.mp4 -b MOG2_CUDA
echo.
echo To verify CUDA is being used, check the output for:
echo   "Using MOG2_CUDA background subtractor"
echo.

:end
pause
