import os
import shutil
import zipfile
import tarfile
from pathlib import Path

def extract_and_merge():
    # 1. Definir rutas usando Path para evitar problemas de sistema operativo
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    output_dir = data_dir / "utkface_completo"
    
    # Crear la carpeta final si no existe
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Buscando archivos comprimidos en: {data_dir}")
    
    # Extensiones comunes de compresión
    valid_extensions = ('.zip', '.tar.gz', '.tgz')
    
    # Buscar todos los archivos comprimidos en la carpeta data/
    compressed_files = [f for f in data_dir.iterdir() if f.is_file() and f.name.endswith(valid_extensions)]
    
    if not compressed_files:
        print("No se encontraron archivos comprimidos (.zip o .tar.gz) en la carpeta 'data/'.")
        print("Asegúrate de que tus archivos estén guardados ahí.")
        return

    print(f"Se encontraron {len(compressed_files)} archivos para procesar.")
    
    # 2. Procesar y extraer cada archivo
    for file_path in compressed_files:
        print(f"Extrayendo {file_path.name}...")
        temp_extract_dir = data_dir / f"temp_{file_path.stem}"
        temp_extract_dir.mkdir(exist_ok=True)
        
        # Detectar el tipo de compresión y extraer
        if file_path.name.endswith('.zip'):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(temp_extract_dir)
        elif file_path.name.endswith(('.tar.gz', '.tgz')):
            with tarfile.open(file_path, 'r:gz') as tar_ref:
                tar_ref.extractall(temp_extract_dir)
                
        # 3. Mover las imágenes al directorio unificado
        # Buscamos de forma recursiva cualquier archivo .jpg dentro de lo que se extrajo
        images_found = list(temp_extract_dir.rglob("*.jpg")) + list(temp_extract_dir.rglob("*.jpeg"))
        print(f"-> Encontradas {len(images_found)} imágenes en {file_path.name}. Moviendo...")
        
        for img_path in images_found:
            # Evitar colisiones de nombres si se llaman igual (aunque en UTKFace los nombres son únicos por su timestamp)
            dest_path = output_dir / img_path.name
            shutil.move(str(img_path), str(dest_path))
            
        # Limpiar la carpeta temporal de esa parte
        shutil.rmtree(temp_extract_dir)
        print(f"✔ Finalizada la extracción de {file_path.name} y limpia su carpeta temporal.")

    # 4. Conteo final para verificar
    total_images = len(list(output_dir.glob("*.jpg"))) + len(list(output_dir.glob("*.jpeg")))
    print("\n==================================================")
    print("¡PROCESO FINALIZADO CON ÉXITO!")
    print(f"Todas las imágenes se han unificado en: {output_dir}")
    print(f"Total de imágenes listas para el laboratorio: {total_images}")
    print("==================================================")
    print("\nPróximo paso: Actualiza tu archivo .env con esta ruta:")
    print(f'UTKFACE_DIR="{output_dir}"')

if __name__ == "__main__":
    extract_and_merge()