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

# Directorio de modelos
MODEL_DIR = os.path.join(settings.BASE_DIR, 'detector', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

ZIP_PATH = os.path.join(MODEL_DIR, 'models.zip')
URL_ZIP  = "https://drive.google.com/uc?export=download&id=1xTaOe3UUKLBvBvFv8IyxWzhD5GwwuqCi"

# Función para validar archivos H5
def is_h5_valid(path):
    try:
        with h5py.File(path, 'r') as f:
            return True
    except Exception as e:
        print(f"Archivo corrupto: {e}")
        return False

# Descargar ZIP si no existe
if not os.path.exists(ZIP_PATH):
    print("📥 Descargando archivos de modelos (ZIP) desde Drive...")
    gdown.download(URL_ZIP, ZIP_PATH, quiet=False)

# Descomprimir ZIP si no están los .h5
ALEXNET_PATH = os.path.join(MODEL_DIR, 'best_alexnet_model.h5')
RESNET_PATH  = os.path.join(MODEL_DIR, 'best_resnet50_model.h5')

if (not os.path.exists(ALEXNET_PATH)) or (not os.path.exists(RESNET_PATH)):
    try:
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(MODEL_DIR)
            print("✔ Descompresión completa.")
    except Exception as e:
        print(f"❌ Error al descomprimir ZIP: {e}")

# Cargar los modelos
try:
    if is_h5_valid(ALEXNET_PATH) and is_h5_valid(RESNET_PATH):
        ALEXNET_MODEL = load_model(ALEXNET_PATH)
        RESNET_MODEL  = load_model(RESNET_PATH)
        print("✅ Modelos AlexNet y ResNet50 cargados correctamente.")
        print("AlexNet capas:", [layer.name for layer in ALEXNET_MODEL.layers])
        print("ResNet50 capas:", [layer.name for layer in RESNET_MODEL.layers])
    else:
        print("⚠️ Uno o ambos archivos .h5 están corruptos o no existen.")
        ALEXNET_MODEL = None
        RESNET_MODEL = None
except Exception as e:
    print(f"❌ Error al cargar modelos: {e}")
    ALEXNET_MODEL = None
    RESNET_MODEL = None

# -----------------------------------------------------
# FUNCIONES AUXILIARES: mapas de activación
# -----------------------------------------------------
def activation_to_base64_mri(feature_map, out_size=256):
    try:
        arr = np.array(feature_map, dtype=np.float32)
        if arr.size == 0:
            arr = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        minv, maxv = np.nanmin(arr), np.nanmax(arr)
        eps = 1e-8
        norm = (arr - minv) / (maxv - minv + eps)
        img_uint8 = (norm * 255.0).astype(np.uint8)
        pil = Image.fromarray(img_uint8, mode='L')
        pil = ImageOps.autocontrast(pil)
        pil = ImageEnhance.Contrast(pil).enhance(2.0)
        pil = pil.resize((out_size, out_size), Image.LANCZOS)
        buff = io.BytesIO()
        pil.save(buff, format='PNG', optimize=True)
        buff.seek(0)
        return base64.b64encode(buff.read()).decode('utf-8')
    except Exception:
        black = Image.new('L', (out_size, out_size), color=0)
        buf = io.BytesIO()
        black.save(buf, format='PNG')
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')

def _select_conv_layers(model):
    conv_layers = []
    for layer in model.layers:
        try:
            if isinstance(layer, Conv2D) or ('conv' in layer.name.lower()):
                conv_layers.append(layer)
        except Exception:
            try:
                if 'conv' in layer.name.lower():
                    conv_layers.append(layer)
            except Exception:
                continue
    return conv_layers

def get_feature_maps(model, img_array, max_layers=8):
    if model is None:
        return []
    conv_layers = _select_conv_layers(model)
    selected = conv_layers[:max_layers]
    try:
        outputs = [layer.output for layer in selected]
        intermediate_model = models.Model(inputs=model.input, outputs=outputs)
        activations = intermediate_model.predict(img_array)
    except Exception:
        return []
    visualizaciones = []
    for idx, act in enumerate(activations):
        try:
            if act.ndim == 4:
                act_2d = np.mean(act[0], axis=-1)
            elif act.ndim == 3:
                act_2d = act[0]
            elif act.ndim == 2:
                act_2d = act
            else:
                act_2d = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
            img_b64 = activation_to_base64_mri(act_2d)
            layer_name = selected[idx].name if idx < len(selected) else f"layer_{idx}"
            visualizaciones.append({"nombre": f"{idx+1}: {layer_name}", "imagen": img_b64})
        except Exception:
            visualizaciones.append({"nombre": f"{idx+1}: {(selected[idx].name if idx < len(selected) else 'unknown')}",
                                     "imagen": activation_to_base64_mri(np.zeros((IMAGE_SIZE, IMAGE_SIZE)))})
    return visualizaciones

# -----------------------------------------------------
# ENDPOINT 1: métricas JSON
# -----------------------------------------------------
class MetricsView(APIView):
    def get(self, request):
        if ALEXNET_MODEL is None or RESNET_MODEL is None:
            return Response({"error": "Modelos no disponibles o corruptos."}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        metrics = {
            "model_type": "Ensamble (AlexNet + ResNet50)",
            "accuracy_final": FINAL_ACCURACY,
            "f1_score_final": FINAL_F1_SCORE,
            "description": "Diagnóstico Binario: 0 (Benigno) / 1 (Maligno).",
        }
        return Response(metrics)

# -----------------------------------------------------
# ENDPOINT 2: predicción HTML
# -----------------------------------------------------
class PredictView(View):
    def get(self, request):
        return render(request, 'predict.html', {})

    def post(self, request):
        if ALEXNET_MODEL is None or RESNET_MODEL is None:
            return render(request, 'predict.html', {'error_message': "Modelos no cargados o corruptos, verifique los archivos."})
        if 'image_file' not in request.FILES:
            return render(request, 'predict.html', {'error_message': "Seleccione un archivo de imagen."})

        img_file = request.FILES['image_file']
        try:
            img = Image.open(img_file).convert('RGB')
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
            img_array = np.expand_dims(np.array(img).astype('float32') / 255.0, axis=0)
        except Exception as e:
            return render(request, 'predict.html', {'error_message': f"Error en preprocesamiento: {e}"})

        # Imagen subida en base64
        buffered = io.BytesIO()
        img.save(buffered, format='PNG')
        uploaded_image_url = f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"

        # Predicciones
        pred_alexnet = ALEXNET_MODEL.predict(img_array)[0][0]
        pred_resnet  = RESNET_MODEL.predict(img_array)[0][0]
        combined_pred = (pred_alexnet + pred_resnet) / 2.0
        is_malignant = combined_pred < 0.25

        result = {
            "prediccion_final_probabilidad": float(combined_pred),
            "diagnostico_codigo": 1 if is_malignant else 0,
            "diagnostico_texto": "MALIGNO (Posible Cáncer)" if is_malignant else "BENIGNO (No Cancer)",
            "detalles_modelo": {
                "Probabilidad_AlexNet": float(pred_alexnet),
                "Probabilidad_ResNet50": float(pred_resnet)
            }
        }

        # Mapas de activación
        alexnet_maps = get_feature_maps(ALEXNET_MODEL, img_array)
        resnet_maps  = get_feature_maps(RESNET_MODEL, img_array)

        context = {
            'result': result,
            'alexnet_maps': alexnet_maps,
            'resnet_maps': resnet_maps,
            'uploaded_image_url': uploaded_image_url
        }
        return render(request, 'predict.html', context)
