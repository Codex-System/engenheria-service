from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import JSONResponse

# Importa as nossas regras de negócio (Serviços)
from src.engenharia_service.services.vision_service import calcular_area_por_pixels
from src.engenharia_service.services.ai_service import validar_com_ia

# Cria o roteador para organizar os endpoints
router = APIRouter()

@router.post("/calcular-area")
async def calcular_area_reforma(
    area_total: float = Form(..., description="Área total do ambiente/terreno (m²)"),
    imagem: UploadFile = File(...)
):
    try:
        # Lê os bytes uma única vez
        contents = await imagem.read()
        
        # Executa o Serviço de Visão 
        resultado_visao = calcular_area_por_pixels(contents, area_total)
        
        # Executa o Serviço de IA
        laudo_ia = await validar_com_ia(
            imagem_bytes=resultado_visao["imagem_processada_bytes"],
            cor=resultado_visao["cor_detectada"],
            proporcao=resultado_visao["proporcao"],
            area_calculada=resultado_visao["area_calculada"],
            area_total=area_total
        )

        # Formata e retorna o JSON final
        return {
            "status": "sucesso",
            "calculo_matematico": {
                "cor_identificada": resultado_visao["cor_detectada"],
                "area_total_referencia": area_total,
                "area_calculada_reforma": round(resultado_visao["area_calculada"], 2)
            },
            "auditoria_ia": laudo_ia
        }

    except ValueError as ve:
        # Captura os erros de validação
        return JSONResponse(status_code=400, content={"erro": str(ve)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"erro": f"Erro interno: {str(e)}"})