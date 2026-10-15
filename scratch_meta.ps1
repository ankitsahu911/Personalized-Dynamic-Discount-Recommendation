$shell = New-Object -COMObject Shell.Application
$folder = $shell.Namespace('C:\Users\anknk\Videos\Screen Recordings')
$file = $folder.ParseName('Screen Recording 2026-10-15 225625.mp4')
for ($i = 0; $i -le 320; $i++) {
    $name = $folder.GetDetailsOf($null, $i)
    $val = $folder.GetDetailsOf($file, $i)
    if ($val -and ($name -match 'Length|Duration|Audio|Video|Bit|Frame|Size|Dimensions')) {
        Write-Host "$name ($i): $val"
    }
}
