import os

import cv2
import numpy as np
import pytest

os.environ.setdefault("OPENAI_API_KEY", "chave-de-teste")

BGR_POR_NOME = {
    "branco": (255, 255, 255),
    "vermelho": (0, 0, 255),
    "verde": (0, 255, 0),
    "azul": (255, 0, 0),
    "tijolo": (80, 105, 180),
}


@pytest.fixture
def gerar_imagem():
    def _gerar(altura=200, largura=200, fundo="branco", retangulos=()):
        img = np.full((altura, largura, 3), BGR_POR_NOME[fundo], np.uint8)
        for cor, (y0, y1, x0, x1) in retangulos:
            img[y0:y1, x0:x1] = BGR_POR_NOME[cor]
        _, png = cv2.imencode(".png", img)
        return png.tobytes()

    return _gerar
