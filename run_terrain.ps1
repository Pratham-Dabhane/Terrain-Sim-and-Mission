# PowerShell script to run the Generative AI Terrain Prototype

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Generative AI Terrain Prototype" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Starting the terrain generation prototype..." -ForegroundColor Green
Write-Host ""

try {
    python terrain_prototype.py
}
catch {
    Write-Host "Error running the prototype: $_" -ForegroundColor Red
    Write-Host "Make sure Python is installed and in your PATH" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
