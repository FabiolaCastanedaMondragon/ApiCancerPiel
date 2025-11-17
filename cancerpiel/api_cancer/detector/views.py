# detector/views.py
import os
import zipfile
import gdown
import numpy as np
from django.shortcuts import render
from django.conf import settings
from django.views import View
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from tensorflow.keras.models import load_model
from tensorflow.keras import models
from tensorflow.keras.layers import Conv2D
from PIL import Image, ImageOps, ImageEnhance
import io
import base64
import h5py

# --- 🎯 CONFIGURACIÓN ---
FINAL_ACCURACY = 0.8800
FINAL_F1_SCORE = 0.7300
IMAGE_SIZE = 128

# Directorio de modelos (Render borra todo en cada build, así que se recrea)
MODEL_DIR = os.path.join(settings.BASE_DIR, "detector", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

ZIP_PATH = os.path.join(MODEL_DIR, "models.zip")
URL_ZIP = "https://drive.google.com/uc?export=download&id=1xTaOe3UUKLBvBvFv8IyxWzhD5GwwuqCi"

# Validación del archivo H5
def is_h5_valid(path):
    try:
        with h5py.File(path, "r"):
            return True
    except Exception as e:
        # imprimir para debugging en deploy logs
        print(f"[is_h5_valid] archivo no válido o inexistente {path}: {e}")
        return False

# Descargar ZIP solo si no existe
if not os.path.exists(ZIP_PATH):
    try:
        print("[models] descargando ZIP desde Drive...")
        gdown.download(URL_ZIP, ZIP_PATH, quiet=True)
        print("[models] descarga finalizada (o ya existía).")
    except Exception as e:
        print(f"❌ ERROR: No se pudo descargar el ZIP: {e}")

# Rutas de modelos
ALEXNET_PATH = os.path.join(MODEL_DIR, "best_alexnet_model.h5")
RESNET_PATH = os.path.join(MODEL_DIR, "best_resnet50_model.h5")

# Extraer ZIP si faltan modelos
if (not os.path.exists(ALEXNET_PATH)) or (not os.path.exists(RESNET_PATH)):
    try:
        if os.path.exists(ZIP_PATH):
            with zipfile.ZipFile(ZIP_PATH, "r") as z:
                z.extractall(MODEL_DIR)
            print("[models] ZIP extraído correctamente.")
        else:
            print("[models] ZIP no encontrado; saltando extracción.")
    except Exception as e:
        print(f"❌ ERROR EXTRAYENDO ZIP: {e}")

# Cargar modelos
try:
    if is_h5_valid(ALEXNET_PATH) and is_h5_valid(RESNET_PATH):
        # compile=False para evitar intentar recompilar con optimizadores no disponibles en deploy
        ALEXNET_MODEL = load_model(ALEXNET_PATH, compile=False)
        RESNET_MODEL = load_model(RESNET_PATH, compile=False)
        print("[models] Modelos cargados correctamente.")
    else:
        print("[models] Uno o ambos .h5 no existen o están corruptos.")
        ALEXNET_MODEL = None
        RESNET_MODEL = None
except Exception as e:
    print("❌ ERROR CARGANDO MODELOS:", e)
    ALEXNET_MODEL = None
    RESNET_MODEL = None


# -----------------------------------------------------
# FUNCIONES AUXILIARES PARA ACTIVACIONES
# -----------------------------------------------------

def activation_to_base64_mri(feature_map, out_size=256):
    """
    Convierte una matriz 2D (activación) a PNG base64 en escala de grises.
    Robustecida contra NaNs, rangos cero, y tamaños inesperados.
    """
    try:
        arr = np.array(feature_map, dtype=np.float32)

        # Si la activación está vacía o con shape inesperado, crear matriz neutra
        if arr.size == 0:
            arr = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

        # Normalizar con protección numérica
        minv = np.nanmin(arr)
        maxv = np.nanmax(arr)
        eps = 1e-8
        denom = (maxv - minv) if (maxv - minv) > 0 else eps
        norm = (arr - minv) / denom

        img_uint8 = np.clip((norm * 255.0), 0, 255).astype(np.uint8)
        pil = Image.fromarray(img_uint8, mode="L")

        # Ajustes visuales para estilo "MRI"
        pil = ImageOps.autocontrast(pil, cutoff=0)
        pil = ImageEnhance.Contrast(pil).enhance(2.0)
        pil = pil.resize((out_size, out_size), Image.LANCZOS)

        buff = io.BytesIO()
        pil.save(buff, format="PNG", optimize=True)
        buff.seek(0)
        return base64.b64encode(buff.read()).decode("utf-8")
    except Exception as e:
        print(f"[activation_to_base64_mri] error: {e}")
        # fallback: imagen negra
        blank = Image.new("L", (out_size, out_size), color=0)
        tmp = io.BytesIO()
        blank.save(tmp, format="PNG")
        tmp.seek(0)
        return base64.b64encode(tmp.read()).decode("utf-8")


def _select_conv_layers(model):
    """
    Selecciona capas convolucionales intentando ser tolerante a diferentes nombres/arquitecturas.
    """
    convs = []
    for layer in model.layers:
        try:
            if isinstance(layer, Conv2D) or ("conv" in layer.name.lower()):
                convs.append(layer)
        except Exception:
            # ignorar capas que no expongan nombre u otras propiedades
            continue
    return convs


def get_feature_maps(model, img_array, max_layers=8):
    """
    Devuelve lista de dicts con 'nombre' y 'imagen' (base64) para las primeras `max_layers`
    capas convolucionales del modelo. Si no hay capas o falla, devuelve [].
    """
    if model is None:
        return []

    conv_layers = _select_conv_layers(model)
    if not conv_layers:
        # No hay capas convolucionales detectadas
        print("[get_feature_maps] no se detectaron capas conv en el modelo.")
        return []

    conv_layers = conv_layers[:max_layers]

    try:
        interm = models.Model(inputs=model.input, outputs=[l.output for l in conv_layers])
        activs = interm.predict(img_array)
    except Exception as e:
        print(f"[get_feature_maps] error creando modelo intermedio o predict: {e}")
        return []

    results = []
    for i, act in enumerate(activs):
        try:
            # Convertir activación a 2D para visualizar (media sobre canales)
            if hasattr(act, "ndim"):
                if act.ndim == 4:
                    act2d = np.mean(act[0], axis=-1)
                elif act.ndim == 3:
                    act2d = act[0]
                elif act.ndim == 2:
                    act2d = act
                else:
                    act2d = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
            else:
                act2d = np.zeros((IMAGE_SIZE, IMAGE_SIZE))

            img64 = activation_to_base64_mri(act2d)
            layer_name = conv_layers[i].name if i < len(conv_layers) else f"layer_{i}"
            results.append({"nombre": f"{i+1}: {layer_name}", "imagen": img64})
        except Exception as e:
            print(f"[get_feature_maps] error procesando activacion {i}: {e}")
            results.append({
                "nombre": f"{i+1}: {(conv_layers[i].name if i < len(conv_layers) else 'unknown')}",
                "imagen": activation_to_base64_mri(np.zeros((IMAGE_SIZE, IMAGE_SIZE)))
            })

    return results


# -----------------------------------------------------
# ENDPOINT: MÉTRICAS (JSON)
# -----------------------------------------------------

class MetricsView(APIView):
    def get(self, request):
        if ALEXNET_MODEL is None or RESNET_MODEL is None:
            return Response({"error": "Modelos no disponibles."}, status=status.HTTP_503_SERVICE_UNAVAILABLE)

        return Response({
            "model_type": "Ensamble (AlexNet + ResNet50)",
            "accuracy_final": FINAL_ACCURACY,
            "f1_score_final": FINAL_F1_SCORE,
            "description": "Diagnóstico Binario: 0 (Benigno) / 1 (Maligno)"
        })


# -----------------------------------------------------
# ENDPOINT: PREDICCIÓN + ACTIVACIONES (HTML)
# -----------------------------------------------------

class PredictView(View):

    def get(self, request):
        return render(request, "predict.html")

    def post(self, request):

        if ALEXNET_MODEL is None or RESNET_MODEL is None:
            return render(request, "predict.html", {
                "error_message": "Modelos no cargados o corruptos."
            })

        if "image_file" not in request.FILES:
            return render(request, "predict.html", {
                "error_message": "Seleccione una imagen."
            })

        # --- Imagen subida ---
        file = request.FILES["image_file"]

        try:
            img = Image.open(file).convert("RGB")
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
            arr = np.expand_dims(np.array(img).astype("float32") / 255.0, axis=0)
        except Exception as e:
            print(f"[PredictView] error procesando imagen: {e}")
            return render(request, "predict.html", {
                "error_message": f"Error procesando imagen: {e}"
            })

        # Convertir imagen a b64 (para mantener preview en template)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        uploaded_image_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

        # ---- PREDICCIONES ----
        try:
            p_alex = ALEXNET_MODEL.predict(arr)[0][0]
        except Exception as e:
            print(f"[PredictView] error predict AlexNet: {e}")
            p_alex = float("nan")
        try:
            p_resn = RESNET_MODEL.predict(arr)[0][0]
        except Exception as e:
            print(f"[PredictView] error predict ResNet: {e}")
            p_resn = float("nan")

        # Si alguna predicción no es número válido, manejar como 0.5 (neutral) para no romper la UI
        if not np.isfinite(p_alex):
            p_alex = 0.5
        if not np.isfinite(p_resn):
            p_resn = 0.5

        combined = (p_alex + p_resn) / 2.0
        maligno = combined < 0.25

        result = {
            "prediccion_final_probabilidad": float(combined),
            "diagnostico_codigo": 1 if maligno else 0,
            "diagnostico_texto": "MALIGNO (Posible Cáncer)" if maligno else "BENIGNO (No Cáncer)",
            "detalles_modelo": {
                "Probabilidad_AlexNet": float(p_alex),
                "Probabilidad_ResNet50": float(p_resn)
            }
        }

        # --- MAPAS DE ACTIVACIÓN ---
        alex_maps = get_feature_maps(ALEXNET_MODEL, arr)
        res_maps = get_feature_maps(RESNET_MODEL, arr)

        return render(request, "predict.html", {
            "result": result,
            "alexnet_maps": alex_maps,
            "resnet_maps": res_maps,
            "uploaded_image_url": uploaded_image_url
        })
