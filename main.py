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
        # Ler a imagem diretamente da memória RAM
        contents = await imagem.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse(status_code=400, content={"erro": "Formato de imagem inválido."})

        # Converter para HSV para melhor detecção de cor sob qualquer iluminação
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Definir espectros de cor HSV para Vermelho, Verde e Azul
        
        # VERMELHO (tem duas faixas no OpenCV)
        lower_red_1 = np.array([0, 100, 70])
        upper_red_1 = np.array([10, 255, 255])
        lower_red_2 = np.array([170, 100, 70])
        upper_red_2 = np.array([180, 255, 255])
        mask_vermelha = cv2.inRange(hsv, lower_red_1, upper_red_1) + cv2.inRange(hsv, lower_red_2, upper_red_2)

        # VERDE
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        mask_verde = cv2.inRange(hsv, lower_green, upper_green)

        # AZUL
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([140, 255, 255])
        mask_azul = cv2.inRange(hsv, lower_blue, upper_blue)

        # Avaliar qual cor foi usada para a marcação
        pixels_vermelhos = cv2.countNonZero(mask_vermelha)
        pixels_verdes = cv2.countNonZero(mask_verde)
        pixels_azuis = cv2.countNonZero(mask_azul)

        # Criar um dicionário para descobrir qual tem a maior concentração de pixels
        cores = {
            "vermelho": (pixels_vermelhos, mask_vermelha),
            "verde": (pixels_verdes, mask_verde),
            "azul": (pixels_azuis, mask_azul)
        }

        # Encontra qual cor obteve o maior número de pixels detectados
        cor_detectada = max(cores, key=lambda k: cores[k][0])
        pixels_marcados = cores[cor_detectada][0]
        mask_final = cores[cor_detectada][1]

        # Validação de segurança
        # Se a cor "vencedora" tiver muito poucos pixels (ex: menos de 50),
        # significa que a imagem provavelmente não tem nenhuma marcação clara.
        if pixels_marcados < 50:
             return JSONResponse(
                status_code=400, 
                content={"erro": "Não foi possível encontrar uma marcação clara em Vermelho, Verde ou Azul."}
            )

        # Cálculos Matemáticos (Pixels Totais vs Pixels Marcados)
        pixels_totais = img.shape[0] * img.shape[1]
        
        # Regra de três matemática
        proporcao = pixels_marcados / pixels_totais
        area_calculada = proporcao * area_total

        # Retorno dos dados enriquecido
        return {
            "status": "sucesso",
            "detalhes": {
                "cor_identificada_automaticamente": cor_detectada,
                "pixels_totais_imagem": pixels_totais,
                "pixels_area_marcada": pixels_marcados,
                "proporcao_ocupada": round(proporcao, 4)
            },
            "area_total_referencia": area_total,
            "area_calculada_reforma": round(area_calculada, 2)
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"erro": str(e)})