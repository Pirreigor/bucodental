# Script para inicializar repo, commitear y pushear a GitHub (Windows PowerShell)
# Revisa el script antes de ejecutarlo.

param(
  [string]$remoteUrl = "https://github.com/Pirreigor/bucodental.git",
  [string]$branch = "main"
)

Write-Host "Preparando commit y push al remoto: $remoteUrl"

if (-not (Test-Path .git)) {
  git init
}

git add .
$st = git status --porcelain
if ($st) {
  git commit -m "Proyecto Bucodental - actualizacion desde local"
} else {
  Write-Host "No hay cambios para commitear"
}

# Añadir remote si no existe
$exists = git remote
if ($exists -notcontains 'origin') {
  git remote add origin $remoteUrl
}

git branch -M $branch
Write-Host "Empujando al remoto (origin/$branch) ..."

git push -u origin $branch

Write-Host "Hecho. Si falla el push, revisa credenciales o usa GitHub CLI para autenticar."
