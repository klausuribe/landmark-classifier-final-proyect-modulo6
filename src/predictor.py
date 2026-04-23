"""
predictor.py
Inferencia con el mejor modelo. Incluye función predict_landmarks
para usarse tanto programáticamente como con Gradio.
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from pathlib import Path

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_inference_transform():
    """Transform determinístico para inferencia (mismo que val/test)."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_model(model_path, device="cpu"):
    """Carga un modelo TorchScript desde disco."""
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model


def predict_landmarks(img_path, model, classes, k=3, device="cpu"):
    """
    Predice los top-k landmarks más probables para una imagen.

    Args:
        img_path: ruta a la imagen (str o Path) o PIL.Image
        model: modelo TorchScript cargado
        classes: lista de nombres de clases (ordenadas según ImageFolder)
        k: número de predicciones top-k a retornar
        device: "cpu" o "cuda"

    Returns:
        lista de tuplas [(clase, probabilidad), ...] ordenada por probabilidad desc
    """
    # Cargar y transformar la imagen
    if isinstance(img_path, (str, Path)):
        img = Image.open(img_path).convert("RGB")
    else:
        img = img_path.convert("RGB") if img_path.mode != "RGB" else img_path

    transform = get_inference_transform()
    img_tensor = transform(img).unsqueeze(0).to(device)  # (1, 3, 224, 224)

    # Inferencia sin gradientes
    with torch.no_grad():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1)[0]
        top_probs, top_idxs = probs.topk(k)

    results = [(classes[idx.item()], prob.item()) for prob, idx in zip(top_probs, top_idxs)]
    return results
