$extensions = ".jpg", ".jpeg", ".png", ".webp"
Get-ChildItem -File | Where-Object { $extensions -contains $_.Extension.ToLower() } | ForEach-Object {
    $img = $_.Name
    if ($img -eq "identify_and_sort.ps1" -or $img -eq "identify_and_sort_improved.ps1") { return }
    
    Write-Host "Analyzing $img..." -ForegroundColor Cyan
    
    # Try to identify the rock
    $rockType = gemini -m "gemini-2.0-flash" "Identify the rock in this image. Reply with ONLY the name (e.g., Sandstone, Granite, Basalt, Pyrite). If it is not a rock, reply 'Not_A_Rock'. If unsure, reply 'Unknown'." "@$img"
    
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($rockType)) {
        Write-Host "Error analyzing $img. Skipping or moving to Error folder." -ForegroundColor Red
        $folderName = "Error"
    } else {
        # Clean up the response
        $folderName = $rockType.Trim().Split("`n")[0].Trim().Replace(" ", "_").Replace(".", "")
        
        # Limit folder name length to avoid issues
        if ($folderName.Length -gt 50) {
            $folderName = "Complex_Response"
        }
    }
    
    if ($folderName -eq "") { $folderName = "Unknown" }
    
    # Create the folder if it doesn't exist
    if (!(Test-Path $folderName)) {
        New-Item -ItemType Directory -Path $folderName | Out-Null
    }
    
    Write-Host "Result: $folderName" -ForegroundColor Green
    Move-Item -Path $img -Destination "$folderName/" -ErrorAction SilentlyContinue
}