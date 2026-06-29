"""Streamlit user interface for the optimized ResNet model."""

from __future__ import annotations

import os
import torch
import torchvision.transforms as T
from PIL import Image
import streamlit as st

from src.config import AppConfig
from src.inference.face_detector import FaceDetector
from src.utils import resolve_device

# Preprocesamiento idéntico al de validación/test
def preprocess_for_inference(image: Image.Image) -> torch.Tensor:
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225])
    ])
    return transform(image.convert("RGB")).unsqueeze(0)

# Carga optimizada del checkpoint de ResNet sin bloquear hilos
@st.cache_resource
def load_optimal_model():
    from src.models.resnet_todo import MultiTaskResNet 
    
    model = MultiTaskResNet()
    # Ruta local de tu experimento ganador desparamentrizado
    checkpoint_path = "artifacts/checkpoints/resnet_finetuning_unfreeze_more/best_model.pt"
    
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        # Si guardaste el diccionario completo de entrenamiento, extraemos solo los pesos
        if "model_state_dict" in state_dict:
            model.load_state_dict(state_dict["model_state_dict"])
        else:
            model.load_state_dict(state_dict)
    else:
        st.error(f"No se encontró el checkpoint en: {checkpoint_path}")
        
    model.eval()
    return model


def run_app() -> None:
    """Render the upload/camera, face detection and prediction workflow."""

    st.set_page_config(
        page_title="UTKFace: genero y edad",
        layout="wide",
    )
    st.title("Clasificación de género y regresión de edad — ResNet18")
    st.write(
        "Ejemplo educativo con una arquitectura ResNet18 Fine-Tuning entrenada en UTKFace. "
        "La aplicación detecta la cara más grande usando OpenCV, la recorta y aplica la inferencia "
        "multitarea optimizada."
    )

    config = AppConfig.from_env()
    device = torch.device("cpu") # Forzamos CPU local para evitar problemas con hilos de GPU
    
    st.sidebar.header("Modelo Activo")
    st.sidebar.code("ResNet18 Fine-Tuning (Unfreeze More)")
    st.sidebar.write(f"Device de Inferencia: `{device}`")

    # Inicialización del modelo
    try:
        model = load_optimal_model()
    except Exception as e:
        st.error(f"Error crítico al inicializar la red ResNet: {e}")
        return

    uploaded_file = st.file_uploader(
        "Sube una imagen",
        type=["jpg", "jpeg", "png"],
    )
    captured_file = st.camera_input("O captura una imagen con la cámara")
    source_file = captured_file if captured_file is not None else uploaded_file

    if source_file is None:
        st.info("Sube o captura una imagen para comenzar.")
        return

    image = Image.open(source_file).convert("RGB")
    detector = FaceDetector()
    detection = detector.detect_largest(image)

    left_column, right_column = st.columns(2)
    with left_column:
        st.subheader("Imagen de entrada")
        if detection is None:
            st.image(image, use_container_width=True)
        else:
            st.image(
                detector.draw_box(image, detection),
                caption="Rostro detectado por OpenCV",
                use_container_width=True,
            )

    if detection is None:
        st.warning(
            "No se detectó un rostro frontal. Prueba una imagen con mejor "
            "iluminación y una cara más visible."
        )
        return

    with right_column:
        st.subheader("Rostro recortado para la Red")
        st.image(detection.crop, use_container_width=True)

    if not st.button("Ejecutar modelo ResNet", type="primary"):
        return

    # Proceso de Inferencia empleando el modelo cargado con 2 salidas
    with st.spinner("Ejecutando inferencia sobre el rostro recortado..."):
        try:
            tensor_x = preprocess_for_inference(detection.crop)
            with torch.no_grad():
                gender_logits, age_pred = model(tensor_x)
                
                # Manejo de clasificación binaria con salida de 2 dimensiones (Softmax)
                probabilities = torch.softmax(gender_logits, dim=1)
                confidence, gender_index = probabilities.max(dim=1)
                
                idx = int(gender_index.item())
                # Mapeo de índices: 0 para Masculino, 1 para Femenino
                gender_label = "Femenino" if idx == 1 else "Masculino"
                gender_confidence = float(confidence.item())
                
                estimated_age = float(age_pred.item())
        except Exception as error:
            st.error(f"Error durante la inferencia: {error}")
            return

    st.subheader("Resultado del Análisis Multitarea")
    metric_gender, metric_age, metric_confidence = st.columns(3)
    metric_gender.metric("Género predicho", gender_label)
    metric_age.metric("Edad estimada", f"{estimated_age:.1f} años")
    metric_confidence.metric(
        "Confianza de género",
        f"{gender_confidence * 100:.1f}%",
    )
    st.caption(
        "Estas salidas reflejan las pautas biométricas aprendidas del dataset "
        "UTKFace durante la etapa de Fine-Tuning."
    )