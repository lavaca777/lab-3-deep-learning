import sys
import os

print("--- DIAGNÓSTICO DE IMPORTS ---")
print(f"Directorio de trabajo: {os.getcwd()}")
print(f"Python Path: {sys.path}\n")

try:
    import torch
    print("[OK] PyTorch cargado correctamente.")
    import torchvision
    print("[OK] Torchvision cargado correctamente.")
    import streamlit
    print("[OK] Streamlit cargado correctamente.")
except Exception as e:
    print(f"[ERROR DE ENTORNO]: No se pudo cargar alguna librería base: {e}")
    sys.exit(1)

try:
    print("\nIntentando importar tu modelo...")
    from src.models.resnet_todo import MultiTaskResNet
    print("[OK] Clase MultiTaskResNet18 importada con éxito.")
except Exception as e:
    print(f"\n[ERROR ENCONTRADO EN TU CÓDIGO]: {e}")