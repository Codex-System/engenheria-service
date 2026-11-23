import logging

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from engenharia_service.services.ai_service import AuditoriaIndisponivelError, validar_com_ia
from engenharia_service.services.vision_service import calcular_area_por_pixels

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/calcular-area")
async def calcular_area_reforma(
    area_total: float = Form(..., gt=0, description="Área total do ambiente/terreno (m²)"),
    imagem: UploadFile = File(...),
):
    try:
        conteudo = await imagem.read()
        resultado_visao = await run_in_threadpool(calcular_area_por_pixels, conteudo, area_total)
    except ValueError as erro:
        return JSONResponse(status_code=400, content={"erro": str(erro)})
    except Exception:
        logger.exception("Falha no processamento da imagem")
        return JSONResponse(status_code=500, content={"erro": "Erro interno ao processar a imagem."})

    try:
        laudo_ia = await validar_com_ia(
            imagem_jpeg=resultado_visao.imagem_jpeg,
            cor=resultado_visao.cor_detectada,
            proporcao=resultado_visao.proporcao,
            area_calculada=resultado_visao.area_calculada,
            area_total=area_total,
        )
        status = "aprovado" if laudo_ia["aprovado"] else "reprovado"
    except AuditoriaIndisponivelError:
        logger.exception("Auditoria da IA indisponível")
        laudo_ia = None
        status = "auditoria_indisponivel"

    return {
        "status": status,
        "calculo_matematico": {
            "cor_identificada": resultado_visao.cor_detectada,
            "area_total_referencia": area_total,
            "area_calculada_reforma": round(resultado_visao.area_calculada, 2),
        },
        "auditoria_ia": laudo_ia,
    }
