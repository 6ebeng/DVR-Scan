#
#      DVR-Scan: Video Motion Event Detection & Extraction Tool
#   --------------------------------------------------------------
#       [  Site: https://www.dvr-scan.com/                 ]
#       [  Repo: https://github.com/Breakthrough/DVR-Scan  ]
#
# Copyright (C) 2016 Brandon Castellano <http://www.bcastell.com>.
# DVR-Scan is licensed under the BSD 2-Clause License; see the included
# LICENSE file, or visit one of the above pages for details.
#
"""``dvr_scan.cuda_setup`` Module

Provides CUDA detection, setup, and diagnostic utilities for Windows 11 native installations.
This module helps users configure CUDA support after installing DVR-Scan via MSI.
"""

import logging
import os
import platform
import subprocess
import sys
import typing as ty
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("dvr_scan")

# Standard CUDA installation paths on Windows
CUDA_DEFAULT_PATHS = [
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
    r"C:\Program Files\NVIDIA Corporation\CUDA",
]

# Required CUDA DLLs for OpenCV CUDA support
CUDA_REQUIRED_DLLS = [
    "cudart64_*.dll",
    "cublas64_*.dll",
    "cufft64_*.dll",
    "curand64_*.dll",
    "cusparse64_*.dll",
    "nppc64_*.dll",
    "nppial64_*.dll",
    "nppicc64_*.dll",
    "nppidei64_*.dll",
    "nppif64_*.dll",
    "nppig64_*.dll",
    "nppim64_*.dll",
    "nppist64_*.dll",
    "nppisu64_*.dll",
    "nppitc64_*.dll",
]

# cuDNN DLLs (optional but recommended for better performance)
CUDNN_DLLS = [
    "cudnn64_*.dll",
    "cudnn_ops_infer64_*.dll",
    "cudnn_cnn_infer64_*.dll",
]


@dataclass
class CudaInfo:
    """Information about CUDA installation."""
    is_installed: bool
    cuda_path: ty.Optional[str]
    cuda_version: ty.Optional[str]
    driver_version: ty.Optional[str]
    gpu_name: ty.Optional[str]
    gpu_compute_capability: ty.Optional[str]
    cudnn_installed: bool
    cudnn_version: ty.Optional[str]
    opencv_cuda_available: bool
    missing_dlls: ty.List[str]
    errors: ty.List[str]


def get_nvidia_driver_info() -> ty.Tuple[ty.Optional[str], ty.Optional[str], ty.Optional[str]]:
    """Get NVIDIA driver version and GPU info using nvidia-smi.
    
    Returns:
        Tuple of (driver_version, gpu_name, cuda_version_from_driver)
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,name,compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            if len(parts) >= 3:
                return parts[0], parts[1], parts[2]
            elif len(parts) >= 2:
                return parts[0], parts[1], None
            elif len(parts) >= 1:
                return parts[0], None, None
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
        logger.debug(f"nvidia-smi not available or failed: {e}")
    return None, None, None


def find_cuda_path() -> ty.Optional[str]:
    """Find the CUDA installation path on Windows.
    
    Returns:
        Path to CUDA installation, or None if not found.
    """
    # First check environment variable
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path and os.path.exists(cuda_path):
        return cuda_path
    
    # Check standard installation paths
    for base_path in CUDA_DEFAULT_PATHS:
        if os.path.exists(base_path):
            # Find the latest version
            try:
                versions = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
                if versions:
                    # Sort by version number (e.g., v12.4, v12.3, v11.8)
                    versions.sort(key=lambda x: [int(p) for p in x.lstrip('v').split('.') if p.isdigit()], reverse=True)
                    return os.path.join(base_path, versions[0])
            except (OSError, ValueError):
                continue
    
    return None


def get_cuda_version(cuda_path: str) -> ty.Optional[str]:
    """Get CUDA version from installation path.
    
    Args:
        cuda_path: Path to CUDA installation.
        
    Returns:
        CUDA version string, or None if not found.
    """
    # Try to read version from version.txt
    version_file = os.path.join(cuda_path, "version.txt")
    if os.path.exists(version_file):
        try:
            with open(version_file, 'r') as f:
                content = f.read()
                # Parse "CUDA Version X.Y.Z"
                if "CUDA Version" in content:
                    return content.split("CUDA Version")[-1].strip().split()[0]
        except Exception:
            pass
    
    # Try to extract from path
    path_name = os.path.basename(cuda_path)
    if path_name.startswith('v'):
        return path_name[1:]
    
    return None


def check_cuda_dlls(cuda_path: str) -> ty.List[str]:
    """Check for required CUDA DLLs.
    
    Args:
        cuda_path: Path to CUDA installation.
        
    Returns:
        List of missing DLL patterns.
    """
    import fnmatch
    
    missing = []
    bin_path = os.path.join(cuda_path, "bin")
    
    if not os.path.exists(bin_path):
        return CUDA_REQUIRED_DLLS.copy()
    
    try:
        files = os.listdir(bin_path)
    except OSError:
        return CUDA_REQUIRED_DLLS.copy()
    
    for pattern in CUDA_REQUIRED_DLLS:
        found = any(fnmatch.fnmatch(f.lower(), pattern.lower()) for f in files)
        if not found:
            missing.append(pattern)
    
    return missing


def check_cudnn(cuda_path: str) -> ty.Tuple[bool, ty.Optional[str]]:
    """Check for cuDNN installation.
    
    Args:
        cuda_path: Path to CUDA installation.
        
    Returns:
        Tuple of (is_installed, version).
    """
    import fnmatch
    
    bin_path = os.path.join(cuda_path, "bin")
    if not os.path.exists(bin_path):
        return False, None
    
    try:
        files = os.listdir(bin_path)
    except OSError:
        return False, None
    
    for pattern in CUDNN_DLLS:
        if any(fnmatch.fnmatch(f.lower(), pattern.lower()) for f in files):
            # Try to extract version from filename
            for f in files:
                if fnmatch.fnmatch(f.lower(), "cudnn64_*.dll"):
                    parts = f.replace(".dll", "").split("_")
                    if len(parts) >= 2:
                        return True, parts[1]
            return True, None
    
    return False, None


def check_opencv_cuda() -> bool:
    """Check if OpenCV has CUDA support.
    
    Returns:
        True if OpenCV CUDA is available.
    """
    try:
        import cv2
        return hasattr(cv2, "cuda") and hasattr(cv2.cuda, "createBackgroundSubtractorMOG2")
    except ImportError:
        return False


def get_opencv_cuda_info() -> ty.Dict[str, ty.Any]:
    """Get detailed OpenCV CUDA information.
    
    Returns:
        Dictionary with OpenCV CUDA details.
    """
    info = {
        "opencv_version": None,
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_devices": [],
        "build_info": None,
    }
    
    try:
        import cv2
        info["opencv_version"] = cv2.__version__
        
        if hasattr(cv2, "cuda"):
            info["cuda_available"] = True
            try:
                count = cv2.cuda.getCudaEnabledDeviceCount()
                info["cuda_device_count"] = count
                for i in range(count):
                    try:
                        cv2.cuda.setDevice(i)
                        props = cv2.cuda.DeviceInfo(i)
                        info["cuda_devices"].append({
                            "index": i,
                            "name": props.name() if hasattr(props, 'name') else "Unknown",
                            "compute_capability": f"{props.majorVersion()}.{props.minorVersion()}" 
                                if hasattr(props, 'majorVersion') else "Unknown",
                            "total_memory_mb": props.totalMemory() // (1024 * 1024)
                                if hasattr(props, 'totalMemory') else 0,
                        })
                    except Exception:
                        pass
            except Exception:
                pass
        
        # Get build info
        if hasattr(cv2, 'getBuildInformation'):
            build_info = cv2.getBuildInformation()
            # Extract CUDA-related lines
            cuda_lines = [line for line in build_info.split('\n') 
                         if 'CUDA' in line.upper() or 'NVIDIA' in line.upper()]
            info["build_info"] = '\n'.join(cuda_lines) if cuda_lines else None
            
    except ImportError:
        pass
    
    return info


def setup_cuda_dll_paths() -> bool:
    """Setup CUDA DLL paths for Windows.
    
    This adds CUDA bin directories to the DLL search path so OpenCV can find CUDA libraries.
    
    Returns:
        True if paths were configured successfully.
    """
    if os.name != "nt":
        logger.debug("CUDA DLL path setup is only needed on Windows")
        return True
    
    cuda_path = find_cuda_path()
    if not cuda_path:
        logger.warning("CUDA installation not found. Please install CUDA Toolkit.")
        return False
    
    cuda_bin = os.path.join(cuda_path, "bin")
    if not os.path.exists(cuda_bin):
        logger.warning(f"CUDA bin directory not found: {cuda_bin}")
        return False
    
    try:
        os.add_dll_directory(cuda_bin)
        logger.debug(f"Added CUDA DLL directory: {cuda_bin}")
        
        # Also add cuDNN path if it exists in a different location
        cudnn_path = os.environ.get("CUDNN_PATH")
        if cudnn_path:
            cudnn_bin = os.path.join(cudnn_path, "bin")
            if os.path.exists(cudnn_bin):
                os.add_dll_directory(cudnn_bin)
                logger.debug(f"Added cuDNN DLL directory: {cudnn_bin}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to add CUDA DLL directory: {e}")
        return False


def get_cuda_info() -> CudaInfo:
    """Get comprehensive CUDA installation information.
    
    Returns:
        CudaInfo dataclass with all CUDA details.
    """
    errors = []
    
    # Check NVIDIA driver
    driver_version, gpu_name, compute_cap = get_nvidia_driver_info()
    
    # Find CUDA installation
    cuda_path = find_cuda_path()
    cuda_version = None
    missing_dlls = []
    
    if cuda_path:
        cuda_version = get_cuda_version(cuda_path)
        missing_dlls = check_cuda_dlls(cuda_path)
        if missing_dlls:
            errors.append(f"Missing CUDA DLLs: {', '.join(missing_dlls[:5])}")
    else:
        errors.append("CUDA Toolkit not found. Please install from: https://developer.nvidia.com/cuda-downloads")
    
    # Check cuDNN
    cudnn_installed = False
    cudnn_version = None
    if cuda_path:
        cudnn_installed, cudnn_version = check_cudnn(cuda_path)
    
    # Check OpenCV CUDA
    opencv_cuda = check_opencv_cuda()
    if not opencv_cuda:
        errors.append(
            "OpenCV CUDA support not available. "
            "Install opencv-python-cuda wheel from: "
            "https://github.com/cudawarped/opencv-python-cuda-wheels/releases"
        )
    
    return CudaInfo(
        is_installed=cuda_path is not None,
        cuda_path=cuda_path,
        cuda_version=cuda_version,
        driver_version=driver_version,
        gpu_name=gpu_name,
        gpu_compute_capability=compute_cap,
        cudnn_installed=cudnn_installed,
        cudnn_version=cudnn_version,
        opencv_cuda_available=opencv_cuda,
        missing_dlls=missing_dlls,
        errors=errors,
    )


def print_cuda_diagnostics():
    """Print comprehensive CUDA diagnostics to console."""
    print("\n" + "=" * 60)
    print("DVR-Scan CUDA Diagnostics")
    print("=" * 60)
    
    info = get_cuda_info()
    
    print(f"\n{'System Information':^60}")
    print("-" * 60)
    print(f"  OS:              {platform.platform()}")
    print(f"  Python:          {platform.python_version()}")
    
    print(f"\n{'NVIDIA Driver':^60}")
    print("-" * 60)
    if info.driver_version:
        print(f"  Driver Version:  {info.driver_version}")
        print(f"  GPU Name:        {info.gpu_name or 'Unknown'}")
        print(f"  Compute Cap:     {info.gpu_compute_capability or 'Unknown'}")
    else:
        print("  Status:          NOT DETECTED")
        print("  Action:          Install NVIDIA drivers from https://www.nvidia.com/drivers")
    
    print(f"\n{'CUDA Toolkit':^60}")
    print("-" * 60)
    if info.is_installed:
        print(f"  Status:          INSTALLED")
        print(f"  Path:            {info.cuda_path}")
        print(f"  Version:         {info.cuda_version or 'Unknown'}")
        if info.missing_dlls:
            print(f"  Missing DLLs:    {len(info.missing_dlls)} DLLs not found")
    else:
        print("  Status:          NOT INSTALLED")
        print("  Action:          Install from https://developer.nvidia.com/cuda-downloads")
    
    print(f"\n{'cuDNN':^60}")
    print("-" * 60)
    if info.cudnn_installed:
        print(f"  Status:          INSTALLED")
        print(f"  Version:         {info.cudnn_version or 'Unknown'}")
    else:
        print("  Status:          NOT INSTALLED (optional)")
        print("  Action:          Install from https://developer.nvidia.com/cudnn")
    
    print(f"\n{'OpenCV CUDA':^60}")
    print("-" * 60)
    opencv_info = get_opencv_cuda_info()
    print(f"  OpenCV Version:  {opencv_info['opencv_version'] or 'Not installed'}")
    if opencv_info["cuda_available"]:
        print(f"  CUDA Support:    AVAILABLE")
        print(f"  CUDA Devices:    {opencv_info['cuda_device_count']}")
        for dev in opencv_info["cuda_devices"]:
            print(f"    [{dev['index']}] {dev['name']} (CC {dev['compute_capability']}, {dev['total_memory_mb']} MB)")
    else:
        print("  CUDA Support:    NOT AVAILABLE")
        print("  Action:          Replace opencv-python with CUDA-enabled build")
    
    print(f"\n{'DVR-Scan CUDA Status':^60}")
    print("-" * 60)
    if info.opencv_cuda_available and info.is_installed and not info.missing_dlls:
        print("  Status:          READY TO USE")
        print("  Usage:           dvr-scan -i video.mp4 -b MOG2_CUDA")
    else:
        print("  Status:          NOT READY")
        print("\n  Issues to resolve:")
        for i, error in enumerate(info.errors, 1):
            print(f"    {i}. {error}")
    
    print("\n" + "=" * 60)
    
    return info


def install_cuda_opencv_instructions() -> str:
    """Generate instructions for installing OpenCV with CUDA support.
    
    Returns:
        Multi-line string with installation instructions.
    """
    return """
================================================================================
                    Installing OpenCV with CUDA Support
================================================================================

DVR-Scan requires OpenCV built with CUDA support to use GPU acceleration.
The standard opencv-python package from PyPI does NOT include CUDA support.

OPTION 1: Pre-built CUDA Wheel (Recommended)
--------------------------------------------------------------------------------
1. Download the appropriate wheel from:
   https://github.com/cudawarped/opencv-python-cuda-wheels/releases

2. Uninstall existing OpenCV:
   pip uninstall opencv-python opencv-python-headless opencv-contrib-python

3. Install the CUDA wheel:
   pip install opencv_python_cuda-X.X.X-cpXX-cpXX-win_amd64.whl

OPTION 2: Build OpenCV from Source
--------------------------------------------------------------------------------
This requires Visual Studio, CMake, and the CUDA Toolkit installed.
See: https://docs.opencv.org/master/d3/d52/tutorial_windows_install.html

VERIFICATION
--------------------------------------------------------------------------------
After installation, run:
   python -c "import cv2; print('CUDA devices:', cv2.cuda.getCudaEnabledDeviceCount())"

If this shows a number > 0, CUDA is working correctly.

Then run DVR-Scan with:
   dvr-scan -i video.mp4 -b MOG2_CUDA

================================================================================
"""


def auto_configure_cuda() -> bool:
    """Automatically configure CUDA for DVR-Scan.
    
    This function attempts to:
    1. Find and setup CUDA DLL paths
    2. Verify OpenCV CUDA support
    3. Provide guidance if configuration fails
    
    Returns:
        True if CUDA is ready to use.
    """
    if os.name != "nt":
        # On non-Windows systems, CUDA typically works out of the box if installed
        return check_opencv_cuda()
    
    # Setup DLL paths
    setup_cuda_dll_paths()
    
    # Check if CUDA is now available
    info = get_cuda_info()
    
    if info.opencv_cuda_available:
        logger.info("CUDA acceleration is available. Use '-b MOG2_CUDA' for GPU processing.")
        return True
    
    if info.errors:
        logger.warning("CUDA acceleration not available:")
        for error in info.errors:
            logger.warning(f"  - {error}")
    
    return False


if __name__ == "__main__":
    # Run diagnostics when executed directly
    import argparse
    
    parser = argparse.ArgumentParser(description="DVR-Scan CUDA Diagnostics")
    parser.add_argument("--instructions", action="store_true", 
                       help="Show OpenCV CUDA installation instructions")
    args = parser.parse_args()
    
    if args.instructions:
        print(install_cuda_opencv_instructions())
    else:
        print_cuda_diagnostics()
