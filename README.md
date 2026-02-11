# Bucodental

Repositorio del proyecto Bucodental (backend + frontend). Este repo contiene el código fuente pero excluye los artefactos pesados (modelos y resultados) por defecto.

Instrucciones rápidas para subir al repositorio remoto:

1. (Opcional) Habilitar Git LFS si quieres subir modelos grandes:

```powershell
git lfs install
git lfs track "backend/modelos/**"
git lfs track "backend/resultados/**"
git add .gitattributes
```

2. Inicializar / confirmar y empujar:

```powershell
cd bucodental_web
git init   # si aún no está inicializado
git add .
git commit -m "Proyecto Bucodental - subir código (excluye modelos/resultados)"
git remote add origin https://github.com/Pirreigor/bucodental.git
git branch -M main
git push -u origin main
```

Nota: los directorios `backend/modelos/` y `backend/resultados/` están ignorados por `.gitignore` para evitar fallos al subir archivos pesados. Si quieres mantener los modelos en el repo usa `git lfs` o súbelos a un storage externo (Drive, S3) y apunta con un archivo `manifest`.

Si quieres que yo haga el commit y push ahora, confirma y lo intento (necesitarás credenciales configuradas en tu entorno git).
