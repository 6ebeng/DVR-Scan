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
"""``dvr_scan.opencv_loader`` Module

Ensures required DLL files can be loaded by Python when importing OpenCV, and provides
better error messaging in cases where the module isn't installed.

On Windows 11, this module also handles automatic CUDA DLL path configuration for
native GPU acceleration support after MSI installation.
"""

import importlib
import importlib.util
import logging
import os
import typing as ty

logger = logging.getLogger("dvr_scan")

# Standard CUDA installation paths on Windows
_CUDA_DEFAULT_PATHS = [
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
    r"C:\Program Files\NVIDIA Corporation\CUDA",
]


def _find_cuda_path() -> ty.Optional[str]:
    """Find CUDA installation path on Windows.
    
    Checks CUDA_PATH environment variable first, then standard installation locations.
    Returns the path to the latest CUDA version found.
    """
    # First check environment variable
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path and os.path.exists(cuda_path):
        return cuda_path
    
    # Check standard installation paths
    for base_path in _CUDA_DEFAULT_PATHS:
        if os.path.exists(base_path):
            try:
                versions = [d for d in os.listdir(base_path) 
                           if os.path.isdir(os.path.join(base_path, d))]
                if versions:
                    # Sort by version number to get latest (e.g., v12.4, v12.3, v11.8)
                    versions.sort(
                        key=lambda x: [int(p) for p in x.lstrip('v').split('.') if p.isdigit()], 
                        reverse=True
                    )
                    return os.path.join(base_path, versions[0])
            except (OSError, ValueError):
                continue
    
    return None


def _setup_cuda_dll_directories() -> bool:
    """Setup CUDA DLL directories for Windows.
    
    This function adds CUDA bin directories to the DLL search path so that OpenCV
    can find the required CUDA libraries at runtime.
    
    Returns:
        True if CUDA directories were successfully added, False otherwise.
    """
    cuda_paths_added = []
    
    # Try CUDA_PATH environment variable first
    cuda_path_env = os.environ.get("CUDA_PATH")
    if cuda_path_env and os.path.exists(cuda_path_env):
        cuda_bin = os.path.abspath(os.path.join(cuda_path_env, "bin"))
        if os.path.exists(cuda_bin):
            try:
                os.add_dll_directory(cuda_bin)
                cuda_paths_added.append(cuda_bin)
            except (OSError, AttributeError):
                pass
    
    # Also try to find CUDA automatically
    auto_cuda_path = _find_cuda_path()
    if auto_cuda_path:
        cuda_bin = os.path.abspath(os.path.join(auto_cuda_path, "bin"))
        if os.path.exists(cuda_bin) and cuda_bin not in cuda_paths_added:
            try:
                os.add_dll_directory(cuda_bin)
                cuda_paths_added.append(cuda_bin)
            except (OSError, AttributeError):
                pass
    
    # Add cuDNN path if available
    cudnn_path = os.environ.get("CUDNN_PATH")
    if cudnn_path and os.path.exists(cudnn_path):
        cudnn_bin = os.path.abspath(os.path.join(cudnn_path, "bin"))
        if os.path.exists(cudnn_bin) and cudnn_bin not in cuda_paths_added:
            try:
                os.add_dll_directory(cudnn_bin)
                cuda_paths_added.append(cudnn_bin)
            except (OSError, AttributeError):
                pass
    
    # Also check NVIDIA system paths for cuDNN
    nvidia_paths = [
        os.path.join(os.environ.get("ProgramFiles", ""), "NVIDIA", "CUDNN"),
        os.path.join(os.environ.get("ProgramFiles", ""), "NVIDIA GPU Computing Toolkit", "CUDNN"),
    ]
    for nvidia_path in nvidia_paths:
        if os.path.exists(nvidia_path):
            try:
                versions = [d for d in os.listdir(nvidia_path) 
                           if os.path.isdir(os.path.join(nvidia_path, d))]
                if versions:
                    versions.sort(reverse=True)
                    cudnn_bin = os.path.join(nvidia_path, versions[0], "bin")
                    if os.path.exists(cudnn_bin) and cudnn_bin not in cuda_paths_added:
                        os.add_dll_directory(cudnn_bin)
                        cuda_paths_added.append(cudnn_bin)
            except (OSError, ValueError, AttributeError):
                pass
    
    return len(cuda_paths_added) > 0


# On Windows, make sure we include any required DLL paths for CUDA support.
# This enables native CUDA acceleration after MSI installation on Windows 11.
if os.name == "nt":
    _cuda_configured = _setup_cuda_dll_directories()
    if _cuda_configured:
        logger.debug("CUDA DLL directories configured for Windows native support")

# OpenCV is a required package, but we don't have it as an explicit dependency since we
# need to support both opencv-python and opencv-python-headless. Include some additional
# context with the exception if this is the case.

if not importlib.util.find_spec("cv2"):
    raise ModuleNotFoundError(
        "OpenCV could not be found, try installing opencv-python:\n\npip install opencv-python",
        name="cv2",
    )
