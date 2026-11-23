from dataclasses import dataclass

import cv2
import numpy as np

SATURACAO_MINIMA = 150
BRILHO_MINIMO = 100
PIXELS_MINIMOS_MARCACAO = 50
FRACAO_MINIMA_REGIAO = 0.001

FAIXAS_HSV_POR_COR = {
    "vermelho": [
        ((0, SATURACAO_MINIMA, BRILHO_MINIMO), (10, 255, 255)),
        ((170, SATURACAO_MINIMA, BRILHO_MINIMO), (180, 255, 255)),
    ],
    "verde": [((35, SATURACAO_MINIMA, BRILHO_MINIMO), (85, 255, 255))],
    "azul": [((100, SATURACAO_MINIMA, BRILHO_MINIMO), (140, 255, 255))],
}

KERNEL_PREENCHER_FALHAS = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


@dataclass(frozen=True)
class ResultadoVisao:
    cor_detectada: str
    proporcao: float
    area_calculada: float
    imagem_jpeg: bytes


def calcular_area_por_pixels(imagem_bytes: bytes, area_total: float) -> ResultadoVisao:
    img = cv2.imdecode(np.frombuffer(imagem_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Formato de imagem inválido.")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    pixels_totais = img.shape[0] * img.shape[1]
    tamanho_minimo_regiao = max(PIXELS_MINIMOS_MARCACAO, int(pixels_totais * FRACAO_MINIMA_REGIAO))

    pixels_por_cor = {
        cor: _contar_pixels_em_regioes_relevantes(_mascara_da_cor(hsv, faixas), tamanho_minimo_regiao)
        for cor, faixas in FAIXAS_HSV_POR_COR.items()
    }
    cor_detectada = max(pixels_por_cor, key=pixels_por_cor.get)
    pixels_marcados = pixels_por_cor[cor_detectada]

    if pixels_marcados < PIXELS_MINIMOS_MARCACAO:
        raise ValueError("Não foi possível encontrar uma marcação clara em Vermelho, Verde ou Azul.")

    proporcao = pixels_marcados / pixels_totais
    _, imagem_jpeg = cv2.imencode(".jpg", img)

    return ResultadoVisao(
        cor_detectada=cor_detectada,
        proporcao=proporcao,
        area_calculada=proporcao * area_total,
        imagem_jpeg=imagem_jpeg.tobytes(),
    )


def _mascara_da_cor(hsv: np.ndarray, faixas: list[tuple[tuple, tuple]]) -> np.ndarray:
    mascara = np.zeros(hsv.shape[:2], np.uint8)
    for minimo, maximo in faixas:
        mascara = cv2.bitwise_or(mascara, cv2.inRange(hsv, np.array(minimo), np.array(maximo)))
    return cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, KERNEL_PREENCHER_FALHAS)


def _contar_pixels_em_regioes_relevantes(mascara: np.ndarray, tamanho_minimo_regiao: int) -> int:
    _, _, estatisticas, _ = cv2.connectedComponentsWithStats(mascara)
    areas_das_regioes = estatisticas[1:, cv2.CC_STAT_AREA]
    return int(areas_das_regioes[areas_das_regioes >= tamanho_minimo_regiao].sum())
