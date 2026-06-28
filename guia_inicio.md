
# Guía de Inicio

### Fase 1: Inicialización y Configuración en el Servidor (Solo primer ingreso)

Al acceder al servidor por primera vez, se deben ejecutar los siguientes pasos en el orden indicado para preparar el entorno de ejecución y los datos:

**1. Clonar el repositorio y configurar las variables de entorno:**

```bash
# Clonar el proyecto en el directorio personal del servidor
git clone https://github.com/lavaca777/lab-3-deep-learning
cd lab-3-deep-learning

# Crear el archivo .env a partir de la plantilla preconfigurada
cp .env.example .env

```

**2. Instalar y activar el entorno virtual (Conda):**

```bash
# Crear el entorno con las dependencias especificadas en el archivo configuration
conda env create -f environment.yml

# ACTIVAR EL ENTORNO (Requisito obligatorio antes de cada ejecución)
conda activate lab03-dl-2026-01

```

**3. Descomprimir y unificar el conjunto de datos:**

```bash
# Ejecutar el script encargado de extraer y unificar las imágenes de UTKFace
python3 extract_dataset.py

```

---

### Fase 2: Flujo de Trabajo Diario (Windows ↔ Servidor)

Para mantener la sincronización del proyecto, se debe seguir la siguiente rutina:

1. **En Windows (Local):** Modificar el código, implementar las arquitecturas de red correspondientes y realizar las correcciones necesarias en su editor de preferencia.
2. **GitHub Desktop:** Utilizar la interfaz gráfica en Windows para registrar los cambios (**Commit**) y subirlos a la rama principal de GitHub (**Push origin**).
3. **En el Servidor (Terminal):** No se debe editar código directamente en el servidor. Su uso se limita a actualizar los archivos y ejecutar los comandos de entrenamiento:

```bash
# 1. Asegurar la permanencia en el directorio del proyecto
cd ~/lab-3-deep-learning

# 2. Descargar los últimos cambios registrados desde Windows
git pull origin main

# 3. Ejecutar el experimento correspondiente (ejemplo con la CNN base)
python3 main.py --experiment cnn_base

```

---

### ⚠️ Advertencias y Resolución de Conflictos

* **Conservación del directorio:** Bajo ninguna circunstancia se debe eliminar la carpeta completa del proyecto en el servidor para resolver un problema de Git. Eliminar el directorio borrará el conjunto de datos ya extraído, obligando a repetir el proceso de descompresión desde cero de manera innecesaria.
* **Resolución de conflictos con archivos locales (Git Pull Error):** En caso de que se generen archivos temporales o reportes directamente en el servidor que impidan realizar el comando `git pull`, se debe utilizar el almacenamiento temporal (`stash`) para limpiar el estado de trabajo sin perder información:
```bash
# 1. Resguardar temporalmente los archivos modificados en el servidor
git stash

# 2. Descargar la versión actualizada desde GitHub
git pull origin main

# 3. Reincorporar los archivos temporales a su ubicación original
git stash pop

```



---