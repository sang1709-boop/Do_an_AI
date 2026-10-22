$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (!(Test-Path '.\venv\Scripts\python.exe')) {
    Write-Host 'Khong co venv\Scripts\python.exe. Tao venv truoc.' -ForegroundColor Red
    exit 1
}
$python = '.\venv\Scripts\python.exe'
$extra = @()
if (Test-Path '.\data\nguong_phu.json') { $extra += @('--add-data', 'data\nguong_phu.json;data') }
if (!(Test-Path '.\models\model.pkl') -or !(Test-Path '.\models\face_landmarker.task') -or !(Test-Path '.\models\pose_landmarker_lite.task')) {
    Write-Host 'Thieu model.pkl hoac MediaPipe .task trong models. Khong the build.' -ForegroundColor Red
    exit 1
}
& $python -m PyInstaller @extra --noconfirm --clean --onedir --windowed --name AI_Driver_Monitor `
    --collect-all mediapipe `
    --collect-all sklearn `
    --collect-all cv2 `
    --collect-submodules google.protobuf `
    --add-data 'models;models' `
    dashboard.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host 'Da build: dist\AI_Driver_Monitor\AI_Driver_Monitor.exe' -ForegroundColor Green
