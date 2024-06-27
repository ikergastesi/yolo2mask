import argparse
from pathlib import Path
import subprocess
import cv2
import numpy as np
from scipy.ndimage import label
from time import time

# Parámetro para definir el porcentaje de la parte inferior que se excluirá
EXCLUSION_PERCENTAGE = 0.1

def convertir_txt_a_mascara(imagen_path, txt_path, output_dir):
    # Cargar la imagen original
    image = cv2.imread(imagen_path)

    # Verificar si la imagen se ha cargado correctamente
    if image is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen en la ruta: {imagen_path}")

    # Obtener las dimensiones de la imagen
    image_height, image_width = image.shape[:2]

    # Leer el archivo de texto
    with open(txt_path, 'r') as file:
        data = file.readlines()

    # Parsear los datos
    segments = []
    for line in data:
        values = list(map(float, line.strip().split()))
        class_id = int(values[0])
        coords = np.array(values[1:]).reshape(-1, 2)  # Cada par de valores representa un punto (x, y)
        segments.append((class_id, coords))

    # Crear una imagen binaria vacía
    mask = np.zeros((image_height, image_width), dtype=np.uint8)

    # Dibujar los polígonos en la máscara
    for class_id, coords in segments:
        # Convertir las coordenadas a píxeles (si están normalizadas)
        coords[:, 0] *= image_width  # Escalar x
        coords[:, 1] *= image_height  # Escalar y
        coords = coords.astype(np.int32)  # Convertir a enteros

        # Dibujar el polígono
        cv2.fillPoly(mask, [coords], color=255)  # Color blanco (255) para la máscara binaria

    return mask

def procesar_imagen(imagen_path, pesos1, pesos2, output_dir):
    # Directorio para las máscaras
    mask_path = output_dir / "mascaras"
    mask_path.mkdir(parents=True, exist_ok=True)

    # Primera inferencia
    subprocess.run(["python", "yolov5/segment/predict.py", "--weights", pesos1, "--img", "1216", "--conf", "0.25", "--source", str(imagen_path), "--save-txt", "--nosave", "--exist-ok", "--project", "data/inferencia", "--name", "inferencia1"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    # Generar máscara para la primera inferencia
    txt_path1 = f"data/inferencia/inferencia1/labels/{imagen_path.stem}.txt"
    mask1 = convertir_txt_a_mascara(str(imagen_path), str(txt_path1), mask_path)
    cv2.imwrite(str(mask_path / "carpetas.png"), mask1)
    
    # Segunda inferencia
    subprocess.run(["python", "yolov5/segment/predict.py", "--weights", pesos2, "--img", "1216", "--conf", "0.25", "--source", str(imagen_path), "--save-txt", "--nosave", "--exist-ok", "--project", "data/inferencia", "--name", "inferencia2"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    
    # Generar máscara para la segunda inferencia
    txt_path2 = f"data/inferencia/inferencia2/labels/{imagen_path.stem}.txt"
    mask2 = convertir_txt_a_mascara(str(imagen_path), str(txt_path2), mask_path)
    cv2.imwrite(str(mask_path / "placas.png"), mask2)

    # Cargar las máscaras binarias generadas
    mask_carpeta = cv2.imread(str(mask_path / "carpetas.png"), 0)  # Máscara de carpetas
    mask_placa = cv2.imread(str(mask_path / "placas.png"), 0)  # Máscara de placas

    # Asegurarse de que las máscaras sean binarias
    _, mask_carpeta = cv2.threshold(mask_carpeta, 127, 255, cv2.THRESH_BINARY)
    _, mask_placa = cv2.threshold(mask_placa, 127, 255, cv2.THRESH_BINARY)

    # Excluir la parte inferior de la máscara de placas
    height, width = mask_placa.shape
    exclusion_height = int(height * (1 - EXCLUSION_PERCENTAGE))
    mask_placa[exclusion_height:] = 0

    # Etiquetar las carpetas
    carpeta_labels, num_carpeta = label(mask_carpeta)

    # Etiquetar las placas
    placa_labels, num_placa = label(mask_placa)

    # Crear una imagen en color para visualizar los resultados
    result_img = np.zeros((*mask_carpeta.shape, 3), dtype=np.uint8)

    # Superponer las placas en la imagen resultante para visualización
    result_img[mask_placa == 255] = [42, 42, 165]  # Marrón para las placas

    # Encontrar los centroides de las carpetas
    centroids = []
    for i in range(1, num_carpeta + 1):
        coords = np.column_stack(np.where(carpeta_labels == i))
        centroid = np.mean(coords, axis=0)
        centroids.append((i, centroid))

    # Ordenar las carpetas por la coordenada x del centroide
    centroids.sort(key=lambda x: x[1][1])  # Ordenar por el valor x (segunda coordenada en el centroid)

    # Lista para almacenar los resultados
    resultados = []

    # Analizar cada carpeta en el orden de izquierda a derecha
    for new_i, (orig_i, _) in enumerate(centroids, start=1):
        carpeta = (carpeta_labels == orig_i)
        placas_en_carpeta = placa_labels * carpeta

        # Contar el número de placas en la carpeta
        placa_ids = np.unique(placas_en_carpeta)
        placa_ids = placa_ids[placa_ids != 0]  # Eliminar el fondo

        # Encontrar el contorno de la carpeta para la visualización
        carpeta_contour, hierarchy = cv2.findContours(carpeta.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(placa_ids) == 0:
            resultados.append(f"Carpeta {new_i}: No hay placa.")
            cv2.drawContours(result_img, carpeta_contour, -1, (0, 0, 255), 2)  # Rojo
        elif len(placa_ids) > 1:
            resultados.append(f"Carpeta {new_i}: Hay mas de una placa.")
            cv2.drawContours(result_img, carpeta_contour, -1, (0, 255, 255), 2)  # Amarillo
        else:
            placa = (placa_labels == placa_ids[0])
            if np.all(placa * carpeta == placa):
                resultados.append(f"Carpeta {new_i}: La placa esta completamente dentro.")
                cv2.drawContours(result_img, carpeta_contour, -1, (0, 255, 0), 2)  # Verde
            else:
                resultados.append(f"Carpeta {new_i}: La placa no esta completamente dentro.")
                cv2.drawContours(result_img, carpeta_contour, -1, (255, 0, 0), 2)  # Azul

    # Superponer result_img en la imagen original
    original_image = cv2.imread(str(imagen_path))
    superposed_img = cv2.addWeighted(original_image, 0.7, result_img, 0.3, 0)

    # Crear un área negra para los resultados
    text_area_width = 600  # Ancho del área de texto
    text_area = np.zeros((superposed_img.shape[0], text_area_width, 3), dtype=np.uint8)

    # Escribir los resultados en el área negra
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_color = (255, 255, 255)
    thickness = 1
    y0, dy = 30, 30  # Coordenadas iniciales y espaciado entre líneas de texto
    for i, line in enumerate(resultados):
        y = y0 + i * dy
        cv2.putText(text_area, line, (10, y), font, font_scale, font_color, thickness)

    # Concatenar la imagen original y el área de texto
    final_img = np.concatenate((superposed_img, text_area), axis=1)

    # Guardar la imagen resultante
    result_path = output_dir / f"{imagen_path.stem}_result.png"
    cv2.imwrite(str(result_path), final_img)

    # Informar que se ha procesado la imagen
    print(f"Inferencia completada para {imagen_path.name}")

def main():
    # Rutas de los pesos
    pesos1 = "yolov5/runs/train-seg/carpetas/weights/best.pt"
    pesos2 = "yolov5/runs/train-seg/placas/weights/best.pt"

    # Directorio de salida para las inferencias
    output_dir = Path("./data/inferencia/resultados")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Argumento para la carpeta de entrada
    parser = argparse.ArgumentParser(description="Ejecutar inferencias de YOLOv5 en todas las imágenes de una carpeta")
    parser.add_argument("carpeta", type=str, help="Ruta de la carpeta de entrada")
    args = parser.parse_args()

    # Ruta de la carpeta
    carpeta_path = Path(args.carpeta).resolve()

    # Procesar todas las imágenes en la carpeta
    for imagen_path in carpeta_path.glob("*.png"):  # Cambiar la extensión si es necesario
        t1 = time()
        procesar_imagen(imagen_path, pesos1, pesos2, output_dir)
        t2 = time()
        print(f"Tiempo de inferencia: {t2-t1}")
if __name__ == "__main__":
    main()
