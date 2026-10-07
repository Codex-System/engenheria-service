import cv2
import numpy as np

def calcular_area_por_pixels(imagem_bytes: bytes, area_total: float) -> dict:
    """
    Recebe os bytes da imagem, aplica os filtros de cor e retorna os cálculos matemáticos.
    """
    nparr = np.frombuffer(imagem_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Formato de imagem inválido.")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Máscaras de cores
    mask_vermelha = cv2.inRange(hsv, np.array([0, 100, 70]), np.array([10, 255, 255])) + cv2.inRange(hsv, np.array([170, 100, 70]), np.array([180, 255, 255]))
    mask_verde = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
    mask_azul = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([140, 255, 255]))

    cores = {
        "vermelho": cv2.countNonZero(mask_vermelha),
        "verde": cv2.countNonZero(mask_verde),
        "azul": cv2.countNonZero(mask_azul)
    }

    cor_detectada = max(cores, key=cores.get)
    pixels_marcados = cores[cor_detectada]

    if pixels_marcados < 50:
         raise ValueError("Não foi possível encontrar uma marcação clara em Vermelho, Verde ou Azul.")

    pixels_totais = img.shape[0] * img.shape[1]
    proporcao = pixels_marcados / pixels_totais
    area_calculada = proporcao * area_total

    # Re-codifica a imagem original para enviar à IA depois
    _, buffer = cv2.imencode('.jpg', img)
    imagem_processada_bytes = buffer.tobytes()

    return {
        "cor_detectada": cor_detectada,
        "proporcao": proporcao,
        "area_calculada": area_calculada,
        "imagem_processada_bytes": imagem_processada_bytes
    }