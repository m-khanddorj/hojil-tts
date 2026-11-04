param(
 [Parameter(Mandatory=$true)][string]$Python,
 [Parameter(Mandatory=$true)][string]$Model,
 [Parameter(Mandatory=$true)][string]$Profile,
 [Parameter(Mandatory=$true)][string]$Output,
 [int]$Port=8766
)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent $PSScriptRoot
$uri="http://127.0.0.1:$Port"
try {
 $status=Invoke-RestMethod "$uri/api/status" -TimeoutSec 2
 if ($status.application -eq 'hojil-playground') { Start-Process $uri; exit }
 throw 'Port is occupied by another application.'
} catch {
 if ($_.Exception.Message -eq 'Port is occupied by another application.') { throw }
}
New-Item -ItemType Directory -Force -Path $Output | Out-Null
$arguments="-m data.playground --model `"$Model`" --profile `"$Profile`" --output `"$Output`" --port $Port"
$process=Start-Process -FilePath $Python -ArgumentList $arguments -WorkingDirectory $repo -WindowStyle Hidden -RedirectStandardOutput (Join-Path $Output '..\server.log') -RedirectStandardError (Join-Path $Output '..\server-error.log') -PassThru
$process.Id | Set-Content (Join-Path $Output '..\server.pid')
Start-Process $uri
