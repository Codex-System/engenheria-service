from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import base64
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv

# Carrega as variáveis do .env (Sua OPENAI_API_KEY)
load_dotenv()

app = FastAPI(title="Motor de IA e Visão - Construtora Alfa")

# Inicializa o cliente Assíncrono da OpenAI para não travar o FastAPI
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def validar_com_ia(imagem_bytes: bytes, cor: str, proporcao: float, area_calculada: float, area_total: float):
    """
    Envia a imagem e os cálculos do OpenCV para a OpenAI validar o contexto e emitir o laudo.
    """
    # Converte a imagem bruta para Base64 (Formato que a OpenAI exige)
    imagem_base64 = base64.b64encode(imagem_bytes).decode('utf-8')
    
    # Constrói um Prompt de Sistema rigoroso (System Prompt)
    prompt_sistema = """
    Você é um Auditor Técnico Sênior de Projetos de Engenharia.
    Seu trabalho é validar laudos de Visão Computacional que calculam áreas de reforma baseados em marcações feitas em fotos ou plantas.

    O algoritmo calculou a proporção da área marcada assumindo que 100% da imagem (todos os pixels) representam a "Área Total de Referência" informada pelo usuário.

    SUA MISSÃO CRÍTICA DE VALIDAÇÃO:
    Para que o cálculo matemático do algoritmo seja válido, a foto DEVE estar perfeitamente enquadrada APENAS no ambiente total da reforma (ex: apenas a calçada inteira, ou apenas o chão da sala inteira).

    CRITÉRIOS DE REJEIÇÃO IMEDIATA (aprovado = false):
    1. Se a foto capturar "sujeira de fundo" que não faz parte da área total a ser medida (ex: céu, rua, carros, muros vizinhos, móveis que tapam o chão, pessoas). Isso distorce o cálculo total de pixels.
    2. Se a imagem não for do ramo de construção (memes, animais, etc).
    3. Se a marcação colorida parecer flutuar ou não demarcar o chão/superfície fisicamente.

    Se você encontrar qualquer "sujeira de fundo", você deve reprovar e explicar nas "observacoes_tecnicas" que o usuário deve tirar uma foto mais aproximada (fechada/croppada) mostrando APENAS a área total real do ambiente, sem incluir a rua ou o céu, para que a proporção de pixels seja exata.

    Responda ESTRITAMENTE no formato JSON:
    {
        "aprovado": true ou false,
        "tipo_imagem_detectada": "ex: Foto de calçada com rua no fundo (Inválida), Planta Baixa limpa (Válida)",
        "observacoes_tecnicas": "Seu laudo explicando por que a foto está boa ou apontando o erro de enquadramento.",
        "fraude_ou_erro_detectado": true ou false
    }
    """

    # Constrói o Prompt do Usuário com os dados matemáticos do OpenCV
    prompt_usuario = f"""
    O algoritmo de Visão Computacional informou que:
    - A área total de referência é: {area_total}m²
    - A cor da marcação detectada foi: {cor}
    - A marcação ocupa {proporcao * 100:.1f}% da imagem.
    - Portanto, a área de reforma calculada é: {area_calculada}m²
    
    Analise a imagem em anexo e aplique as regras do sistema.
    """

    try:
        # Chamada para o modelo visual da OpenAI
        response = await client.chat.completions.create(
            model="gpt-4o", 
            messages=[
                {
                    "role": "system",
                    "content": prompt_sistema
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_usuario},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{imagem_base64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            response_format={"type": "json_object"}, 
            temperature=0.2 
        )
        
        # Extrai e retorna o JSON puro da IA
        import json
        return json.loads(response.choices[0].message.content)
        
    except Exception as e:
        return {"erro_ia": str(e)}


@app.post("/api/v1/calcular-area")
async def calcular_area_reforma(
    area_total: float = Form(..., description="Área total do ambiente/terreno (m²)"),
    imagem: UploadFile = File(...)
):
    try:
        
        contents = await imagem.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse(status_code=400, content={"erro": "Formato de imagem inválido."})

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # máscaras RGB
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
             return JSONResponse(status_code=400, content={"erro": "Não foi possível encontrar uma marcação clara."})

        pixels_totais = img.shape[0] * img.shape[1]
        proporcao = pixels_marcados / pixels_totais
        area_calculada = proporcao * area_total


        # Enviar para a OpenAI
        _, buffer = cv2.imencode('.jpg', img)
        imagem_para_ia = buffer.tobytes()

        laudo_ia = await validar_com_ia(
            imagem_bytes=imagem_para_ia,
            cor=cor_detectada,
            proporcao=proporcao,
            area_calculada=round(area_calculada, 2),
            area_total=area_total
        )

        # Retorno Final Consolidado
        return {
            "status": "sucesso",
            "calculo_matematico": {
                "cor_identificada": cor_detectada,
                "area_total_referencia": area_total,
                "area_calculada_reforma": round(area_calculada, 2)
            },
            "auditoria_ia": laudo_ia
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"erro": str(e)})