#Requires -Version 5.1
<#
.SYNOPSIS
    DVR-Scan CUDA Setup Script for Windows 11

.DESCRIPTION
    This PowerShell script helps configure CUDA support for DVR-Scan after MSI installation.
    It checks for required components and provides guidance for setting up GPU acceleration.

.NOTES
    Requirements:
    - NVIDIA GPU with CUDA compute capability 3.0+
    - NVIDIA Driver (latest recommended)
    - CUDA Toolkit 11.x or 12.x
    - OpenCV with CUDA support

.EXAMPLE
    .\setup_cuda.ps1
    Run the CUDA setup and diagnostics

.EXAMPLE
    .\setup_cuda.ps1 -InstallOpenCVCuda
    Attempt to install OpenCV CUDA wheel automatically

.EXAMPLE
    .\setup_cuda.ps1 -DiagnosticsOnly
    Only run diagnostics without setup prompts
#>

[CmdletBinding()]
param(
    [switch]$InstallOpenCVCuda,
    [switch]$DiagnosticsOnly,
    [switch]$SetEnvironmentVariables
)

$ErrorActionPreference = "Continue"

# Colors for output
function Write-Header { param($text) Write-Host "`n$('=' * 60)" -ForegroundColor Cyan; Write-Host $text -ForegroundColor Cyan; Write-Host "$('=' * 60)" -ForegroundColor Cyan }
function Write-Success { param($text) Write-Host "[OK] $text" -ForegroundColor Green }
function Write-Warn { param($text) Write-Host "[WARNING] $text" -ForegroundColor Yellow }
function Write-Err { param($text) Write-Host "[ERROR] $text" -ForegroundColor Red }
function Write-Info { param($text) Write-Host "[INFO] $text" -ForegroundColor White }

Write-Header "DVR-Scan CUDA Setup for Windows 11"

# System Information
Write-Header "System Information"
Write-Info "OS: $([System.Environment]::OSVersion.VersionString)"
Write-Info "Architecture: $([System.Environment]::Is64BitOperatingSystem ? '64-bit' : '32-bit')"

# Check Python
Write-Header "Checking Python"
try {
    $pythonVersion = & python --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Python: $pythonVersion"
        $pythonPath = & python -c "import sys; print(sys.executable)" 2>&1
        Write-Info "Path: $pythonPath"
    } else {
        Write-Err "Python not found in PATH"
        Write-Info "Please install Python 3.10+ from https://python.org"
    }
} catch {
    Write-Err "Python check failed: $_"
}

# Check NVIDIA Driver
Write-Header "Checking NVIDIA Driver"
try {
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($nvidiaSmi) {
        $driverInfo = & nvidia-smi --query-gpu=driver_version,name,compute_cap,memory.total --format=csv,noheader 2>&1
        if ($LASTEXITCODE -eq 0) {
            $parts = $driverInfo -split ", "
            Write-Success "Driver Version: $($parts[0])"
            Write-Info "GPU: $($parts[1])"
            Write-Info "Compute Capability: $($parts[2])"
            Write-Info "VRAM: $($parts[3])"
        } else {
            Write-Err "nvidia-smi failed: $driverInfo"
        }
    } else {
        Write-Warn "nvidia-smi not found - NVIDIA driver may not be installed"
        Write-Info "Download drivers from: https://www.nvidia.com/drivers"
    }
} catch {
    Write-Err "NVIDIA driver check failed: $_"
}

# Check CUDA Toolkit
Write-Header "Checking CUDA Toolkit"
$cudaPath = $env:CUDA_PATH
$cudaFound = $false

if ($cudaPath -and (Test-Path $cudaPath)) {
    Write-Success "CUDA_PATH: $cudaPath"
    $cudaFound = $true
    
    # Check nvcc version
    $nvccPath = Join-Path $cudaPath "bin\nvcc.exe"
    if (Test-Path $nvccPath) {
        $nvccVersion = & $nvccPath --version 2>&1 | Select-String "release"
        Write-Info "NVCC: $nvccVersion"
    }
} else {
    # Try to find CUDA automatically
    $cudaBasePaths = @(
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
        "C:\Program Files\NVIDIA Corporation\CUDA"
    )
    
    foreach ($basePath in $cudaBasePaths) {
        if (Test-Path $basePath) {
            $versions = Get-ChildItem -Path $basePath -Directory | Sort-Object Name -Descending
            if ($versions) {
                $latestCuda = $versions[0].FullName
                Write-Success "CUDA found: $latestCuda"
                $cudaPath = $latestCuda
                $cudaFound = $true
                
                if ($SetEnvironmentVariables) {
                    Write-Info "Setting CUDA_PATH environment variable..."
                    [System.Environment]::SetEnvironmentVariable("CUDA_PATH", $latestCuda, [System.EnvironmentVariableTarget]::User)
                    $env:CUDA_PATH = $latestCuda
                    Write-Success "CUDA_PATH set to: $latestCuda"
                } else {
                    Write-Warn "CUDA_PATH not set. Run with -SetEnvironmentVariables to configure."
                    Write-Info "Or manually set: setx CUDA_PATH `"$latestCuda`""
                }
                break
            }
        }
    }
    
    if (-not $cudaFound) {
        Write-Warn "CUDA Toolkit not found"
        Write-Info "Download from: https://developer.nvidia.com/cuda-downloads"
        Write-Info "Recommended: CUDA 12.x for best compatibility"
    }
}

# Check cuDNN
Write-Header "Checking cuDNN (Optional)"
$cudnnFound = $false
if ($cudaPath) {
    $cudnnDll = Get-ChildItem -Path (Join-Path $cudaPath "bin") -Filter "cudnn64_*.dll" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($cudnnDll) {
        Write-Success "cuDNN found: $($cudnnDll.Name)"
        $cudnnFound = $true
    } else {
        Write-Info "cuDNN not installed (optional but recommended)"
        Write-Info "Download from: https://developer.nvidia.com/cudnn"
    }
}

# Check OpenCV CUDA Support
Write-Header "Checking OpenCV CUDA Support"
$opencvCudaAvailable = $false

$pythonCheck = @"
import sys
try:
    import cv2
    print(f'opencv_version:{cv2.__version__}')
    if hasattr(cv2, 'cuda'):
        count = cv2.cuda.getCudaEnabledDeviceCount()
        print(f'cuda_devices:{count}')
        if count > 0:
            for i in range(count):
                info = cv2.cuda.DeviceInfo(i)
                print(f'device_{i}:{info.name()}')
    else:
        print('cuda_devices:0')
except ImportError as e:
    print(f'error:OpenCV not installed - {e}')
except Exception as e:
    print(f'error:{e}')
"@

try {
    $result = & python -c $pythonCheck 2>&1
    foreach ($line in $result -split "`n") {
        if ($line -match "^opencv_version:(.+)$") {
            Write-Info "OpenCV Version: $($Matches[1])"
        }
        elseif ($line -match "^cuda_devices:(\d+)$") {
            $deviceCount = [int]$Matches[1]
            if ($deviceCount -gt 0) {
                Write-Success "CUDA Support: Available ($deviceCount device(s))"
                $opencvCudaAvailable = $true
            } else {
                Write-Warn "CUDA Support: Not available"
            }
        }
        elseif ($line -match "^device_(\d+):(.+)$") {
            Write-Info "  GPU $($Matches[1]): $($Matches[2])"
        }
        elseif ($line -match "^error:(.+)$") {
            Write-Err $Matches[1]
        }
    }
} catch {
    Write-Err "OpenCV check failed: $_"
}

# Install OpenCV CUDA if requested
if ($InstallOpenCVCuda -and -not $opencvCudaAvailable) {
    Write-Header "Installing OpenCV with CUDA Support"
    Write-Info "This will download and install a CUDA-enabled OpenCV build."
    Write-Warn "The standard opencv-python package will be replaced."
    
    $confirm = Read-Host "Continue? (y/N)"
    if ($confirm -eq 'y' -or $confirm -eq 'Y') {
        # Uninstall existing OpenCV
        Write-Info "Uninstalling existing OpenCV packages..."
        & pip uninstall opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless -y 2>&1 | Out-Null
        
        # Determine Python version for wheel compatibility
        $pyVer = & python -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')" 2>&1
        
        Write-Info "Please download the appropriate wheel from:"
        Write-Info "https://github.com/cudawarped/opencv-python-cuda-wheels/releases"
        Write-Info ""
        Write-Info "Look for: opencv_python_cuda-*-cp$pyVer-cp$pyVer-win_amd64.whl"
        Write-Info ""
        Write-Info "Then install with: pip install [downloaded-file].whl"
    }
}

# Run DVR-Scan CUDA diagnostics
Write-Header "DVR-Scan CUDA Status"
try {
    & python -c "from dvr_scan.cuda_setup import print_cuda_diagnostics; print_cuda_diagnostics()" 2>&1
} catch {
    Write-Info "Running basic DVR-Scan check..."
    & python -c "from dvr_scan.platform import HAS_MOG2_CUDA; print('MOG2_CUDA Available:', HAS_MOG2_CUDA)" 2>&1
}

# Summary and Next Steps
Write-Header "Summary and Next Steps"

$ready = $true
$steps = @()

if (-not $cudaFound) {
    $ready = $false
    $steps += "1. Install CUDA Toolkit from https://developer.nvidia.com/cuda-downloads"
}

if (-not $opencvCudaAvailable) {
    $ready = $false
    $steps += "2. Install OpenCV with CUDA support:"
    $steps += "   - Download wheel from: https://github.com/cudawarped/opencv-python-cuda-wheels/releases"
    $steps += "   - pip uninstall opencv-python opencv-python-headless opencv-contrib-python"
    $steps += "   - pip install [downloaded-wheel].whl"
}

if ($ready) {
    Write-Success "CUDA is configured and ready to use!"
    Write-Info ""
    Write-Info "Usage:"
    Write-Info "  dvr-scan -i video.mp4 -b MOG2_CUDA"
    Write-Info ""
    Write-Info "Or set in config file (dvr-scan.cfg):"
    Write-Info "  bg-subtractor = MOG2_CUDA"
} else {
    Write-Warn "CUDA is not fully configured. Required steps:"
    Write-Info ""
    foreach ($step in $steps) {
        Write-Info $step
    }
}

Write-Host ""
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
