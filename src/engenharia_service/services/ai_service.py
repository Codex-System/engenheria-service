import base64
import json
from openai import AsyncOpenAI
from src.engenharia_service.core.config import settings

# Instancia o cliente usando a chave isolada no arquivo de config
client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

async def validar_com_ia(imagem_bytes: bytes, cor: str, proporcao: float, area_calculada: float, area_total: float) -> dict:
    """
    Envia a imagem e os cálculos para o GPT-4o auditar e emitir o laudo com o prompt ajustado.
    """
    imagem_base64 = base64.b64encode(imagem_bytes).decode('utf-8')
    
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

    prompt_usuario = f"""
    O algoritmo informou que:
    - A área total de referência é: {area_total}m²
    - A cor da marcação detectada foi: {cor}
    - A marcação ocupa {proporcao * 100:.1f}% da imagem.
    - A área calculada é: {area_calculada}m²
    
    Analise a imagem em anexo.
    """

    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt_sistema},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_usuario},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{imagem_base64}", "detail": "high"}
                        }
                    ]
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"erro_ia": str(e)}