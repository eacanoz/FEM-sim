import logging
import os
from datetime import datetime

def setup_logger(name="FEM_Simulator", log_dir="Logs"):
    # 1. Crear la carpeta de Logs si no existe
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 2. Generar un nombre de archivo único con fecha y hora
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(log_dir, f"sim_{timestamp}.log")

    # 3. Crear el logger principal
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG) # Nivel base: Captura todo

    # Evitar duplicación de logs si la celda de Jupyter se corre varias veces
    if logger.hasHandlers():
        logger.handlers.clear()

    # 4. Formato del texto (Ej: 2026-04-28 14:30:00 - INFO - Newton converged)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                  datefmt='%Y-%m-%d %H:%M:%S')

    # 5. Handler para el Archivo (Guarda hasta los detalles más mínimos)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # 6. Handler para la Consola / Jupyter (Solo muestra lo importante)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO) # Oculta los DEBUG en la pantalla
    console_handler.setFormatter(formatter)

    # Añadir los handlers al logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# Instancia global para importar en otros archivos
sim_logger = setup_logger()