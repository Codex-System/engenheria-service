from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import cv2
import numpy as np

app = FastAPI(title="Motor de IA e Visão - Construtora Alfa")

@app.post("/api/v1/calcular-area")
async def calcular_area_reforma(
    area_total: float = Form(..., description="Área total do ambiente/terreno como referência"),
    imagem: UploadFile = File(..., description="Foto ou planta com a área marcada")
):
    try:
        # Ler a imagem diretamente da memória RAM (sem salvar em disco)
        contents = await imagem.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse(status_code=400, content={"erro": "Formato de imagem inválido."})

        # Converter a imagem para HSV (Matiz, Saturação, Valor)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Definir a máscara de cor (Isolando a marcação VERMELHA)
        # O vermelho no OpenCV fica nas pontas do espectro HSV (0-10 e 170-180)
        lower_red_1 = np.array([0, 120, 70])
        upper_red_1 = np.array([10, 255, 255])
        lower_red_2 = np.array([170, 120, 70])
        upper_red_2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
        mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
        
        # Junta as duas máscaras para captar qualquer tom de vermelho
        mask_vermelha = mask1 + mask2

        
        # Conta os pixels que compõem a marcação vermelha
        pixels_marcados = cv2.countNonZero(mask_vermelha)
        
        # imagem inteira representa a "Área Total"
        pixels_totais = img.shape[0] * img.shape[1]

        if pixels_totais == 0 or pixels_marcados == 0:
            return JSONResponse(
                status_code=400, 
                content={"erro": "Não foi possível encontrar a marcação vermelha na imagem."}
            )

        
        proporcao = pixels_marcados / pixels_totais
        area_calculada = proporcao * area_total

        # Retorna a resposta limpa para ser consumida futuramente
        return {
            "status": "sucesso",
            "detalhes": {
                "pixels_totais_imagem": pixels_totais,
                "pixels_area_marcada": pixels_marcados,
                "proporcao_ocupada": round(proporcao, 4)
            },
            "area_total_referencia": area_total,
            "area_calculada_reforma": round(area_calculada, 2)
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"erro": str(e)})