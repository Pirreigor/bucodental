from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import tensorflow as tf
import numpy as np
from PIL import Image
import io, json, os
import threading
from fastapi.responses import JSONResponse

app = FastAPI(title="API Bucodental")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "modelos")
LABELS_DIR = os.path.join(BASE_DIR, "labels")
RESULTS_DIR = os.path.join(BASE_DIR, "resultados")


# Servir imágenes de resultados (accuracy.png, loss.png, confusion_matrix.png, etc.)
if os.path.exists(RESULTS_DIR):
    app.mount("/resultados", StaticFiles(directory=RESULTS_DIR), name="resultados")

# Servir frontend como estáticos en /static
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


from fastapi.responses import FileResponse

# Servir index.html en la raíz
@app.get("/")
def root():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    return FileResponse(index_path)

MODELS_CACHE = {}

def load_labels(model_key: str):
    path = os.path.join(LABELS_DIR, f"{model_key}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def preprocess_image(img: Image.Image, target_size=(224, 224)):
    # AJUSTA target_size y normalización a tu entrenamiento
    img = img.convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def get_model(model_key: str):
    if model_key in MODELS_CACHE:
        return MODELS_CACHE[model_key]
    model_path = os.path.join(MODELS_DIR, model_key, "modelo.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    model = tf.keras.models.load_model(model_path)
    MODELS_CACHE[model_key] = model
    return model


def _preload_models_bg():
    try:
        if not os.path.exists(MODELS_DIR):
            print("[startup] no existe carpeta modelos; nada que precargar")
            return
        models = [d for d in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, d))]
        if not models:
            print("[startup] no hay modelos en modelos/ para precargar")
            return
        print(f"[startup] iniciando precarga de modelos: {models}")
        for m in models:
            try:
                print(f"[startup] cargando modelo: {m}")
                get_model(m)
                print(f"[startup] modelo cargado: {m}")
            except Exception as e:
                print(f"[startup] fallo cargando {m}: {e}")
        print("[startup] precarga de modelos finalizada")
    except Exception as e:
        print(f"[startup] error en precarga: {e}")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/models")
def list_models():
    # lista carpetas dentro de modelos/
    if not os.path.exists(MODELS_DIR):
        return {"models": []}
    models = [d for d in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, d))]
    return {"models": sorted(models)}

@app.post("/predict")
async def predict(model_key: str = Form(...), file: UploadFile = File(...)):
    try:
        print(f"[predict] inicio petición para modelo={model_key}")
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))

        # Intentar obtener modelo (cacheado si ya se precargó)
        model = get_model(model_key)
        print(f"[predict] modelo obtenido: {model_key}")
        x = preprocess_image(img, target_size=(224, 224))
        x = np.asarray(x, dtype=np.float32)
        print("[predict] entrada preprocesada, ejecutando predict()")
        y = model.predict(x)
        y = np.asarray(y, dtype=np.float32)
        if isinstance(y, list):
            y = np.array(y, dtype=np.float32)
        y0 = y[0] if y.ndim > 1 else y

        # binario (sigmoid) o multiclase (softmax)
        if np.ndim(y0) == 0 or (hasattr(y0, "shape") and len(y0.shape) == 0) or (hasattr(y0, "__len__") and len(y0) == 1):
            score = float(y0[0]) if hasattr(y0, "__len__") else float(y0)
            return {"model": model_key, "type": "binary", "score": score}

        y0 = np.array(y0).astype(float)
        idx = int(np.argmax(y0))
        conf = float(y0[idx])

        labels = load_labels(model_key)
        label = labels[idx] if labels and idx < len(labels) else f"clase_{idx}"



        # --- Grad-CAM ---
        from tf_explain.core.grad_cam import GradCAM
        import matplotlib.pyplot as plt
        # Detectar última capa convolucional
        ultima_conv = None
        for capa in reversed(model.layers):
            if isinstance(capa, tf.keras.layers.Conv2D):
                ultima_conv = capa.name
                break
            if isinstance(capa, tf.keras.Model):
                for subcapa in reversed(capa.layers):
                    if isinstance(subcapa, tf.keras.layers.Conv2D):
                        ultima_conv = subcapa.name
                        break
                if ultima_conv:
                    break

        gradcam_url = None
        if ultima_conv:
            try:
                print(f"[predict] generando Grad-CAM (layer={ultima_conv})")
                explainer = GradCAM()
                x_gc = np.asarray(x, dtype=np.float32)
                if isinstance(x_gc, list):
                    x_gc = np.array(x_gc, dtype=np.float32)
                data = (x_gc, None)
                # si el modelo devuelve múltiples salidas, usar la primera salida para el explain
                model_for_explain = model
                try:
                    outputs = getattr(model, 'outputs', None)
                    if isinstance(outputs, (list, tuple)):
                        model_for_explain = tf.keras.Model(inputs=model.inputs, outputs=outputs[0])
                except Exception:
                    model_for_explain = model
                grid = explainer.explain(data, model_for_explain, class_index=int(idx), layer_name=ultima_conv)
                gradcam_dir = os.path.join(RESULTS_DIR, model_key)
                os.makedirs(gradcam_dir, exist_ok=True)
                gradcam_path = os.path.join(gradcam_dir, "gradcam.png")
                plt.imsave(gradcam_path, grid.astype(np.uint8))
                gradcam_url = f"/resultados/{model_key}/gradcam.png"
                print(f"[predict] Grad-CAM guardado en: {gradcam_path}")
            except Exception as e:
                print(f"[predict] fallo generando Grad-CAM: {e}")
                gradcam_url = None

    except Exception as e:
        print(f"[predict] error durante predict: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

    return {
        "model": model_key,
        "type": "multiclass",
        "label": label,
        "confidence": conf,
        "all_scores": y0.tolist(),
        "gradcam_url": gradcam_url
    }


# Permitir ejecutar con: python main.py
if __name__ == "__main__":
    import uvicorn
    # Iniciar hilo de precarga de modelos para evitar bloqueo en la primera petición
    try:
        t = threading.Thread(target=_preload_models_bg, daemon=True)
        t.start()
    except Exception:
        pass
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
