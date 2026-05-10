# Local Windows PyTorch Fig 4 driver (paper-aligned config).
# Picks up cells from hpc-results/exports/ and runs torch_victim with paper defaults.
# Skips cells already done (output JSON exists).
# Designed to run alongside Mac equivalent — both pull from shared HPC export dir.
# Distribution by parity: Windows = odd cells (A01, A03, ..., A29).

$ErrorActionPreference = 'Continue'
$repo    = 'D:\HW\CS 539\final project'
$venv    = "$repo\.venv-torch-local\Scripts\python.exe"
$exports = "$repo\hpc-results\exports"
$outDir  = "$repo\hpc-results\fig4-pytorch-windows"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$env:PYTHONPATH = "$repo\src"

# Cells assigned to Windows (odd-numbered, 15 cells)
$myCells = 1..30 | Where-Object { $_ % 2 -eq 1 } | ForEach-Object { "A{0:D2}" -f $_ }

while ($true) {
    $progress = $false
    foreach ($cell in $myCells) {
        $pkl = "$exports\$cell-poisondataset.pkl"
        $out = "$outDir\$cell.json"
        if (-not (Test-Path $pkl)) { continue }
        if (Test-Path $out) { continue }
        Write-Host "=== $(Get-Date -Format 'HH:mm:ss') Windows running $cell ==="
        & $venv -m metapoison_hpc.torch_victim `
            --dataset $pkl `
            --arch resnet20 `
            --epochs 200 `
            --trials 4 `
            --batch-size 128 `
            --lr 0.1 `
            --momentum 0 `
            --weight-decay 0 `
            --schedule "100,150" `
            --num-workers 4 `
            --output $out 2>&1 | Tee-Object -FilePath "$outDir\$cell.log"
        $progress = $true
    }
    # Check if all 15 done
    $done = (Get-ChildItem $outDir -Filter '*.json' -ErrorAction SilentlyContinue).Count
    if ($done -ge $myCells.Count) {
        Write-Host "=== ALL $($myCells.Count) WINDOWS CELLS DONE ==="
        break
    }
    if (-not $progress) {
        Write-Host "=== $(Get-Date -Format 'HH:mm:ss') waiting for more exports ($done/$($myCells.Count) done) ==="
        Start-Sleep 120
    }
}
