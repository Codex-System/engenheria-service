import pytest
from fastapi.testclient import TestClient

from engenharia_service.api import routes
from engenharia_service.main import app
from engenharia_service.services.ai_service import AuditoriaIndisponivelError

client = TestClient(app)


def _enviar(imagem: bytes, area_total="40"):
    return client.post(
        "/api/v1/calcular-area",
        data={"area_total": area_total},
        files={"imagem": ("foto.png", imagem, "image/png")},
    )


@pytest.fixture
def imagem_marcada(gerar_imagem):
    return gerar_imagem(retangulos=[("vermelho", (0, 100, 0, 100))])


def _ia_que_responde(aprovado: bool):
    async def validar(**_):
        return {"aprovado": aprovado}

    return validar


@pytest.mark.parametrize("aprovado, status_esperado", [(True, "aprovado"), (False, "reprovado")])
def test_status_reflete_laudo_da_ia(monkeypatch, imagem_marcada, aprovado, status_esperado):
    monkeypatch.setattr(routes, "validar_com_ia", _ia_que_responde(aprovado))

    resposta = _enviar(imagem_marcada)

    assert resposta.status_code == 200
    assert resposta.json()["status"] == status_esperado
    assert resposta.json()["calculo_matematico"]["area_calculada_reforma"] == 10


def test_ia_indisponivel_mantem_calculo_e_sinaliza_status(monkeypatch, imagem_marcada):
    async def falhar(**_):
        raise AuditoriaIndisponivelError("timeout")

    monkeypatch.setattr(routes, "validar_com_ia", falhar)

    resposta = _enviar(imagem_marcada)

    assert resposta.status_code == 200
    assert resposta.json()["status"] == "auditoria_indisponivel"
    assert resposta.json()["auditoria_ia"] is None
    assert resposta.json()["calculo_matematico"]["area_calculada_reforma"] == 10


def test_imagem_invalida_retorna_400():
    resposta = _enviar(b"nao e imagem")

    assert resposta.status_code == 400


@pytest.mark.parametrize("area_total", ["0", "-5"])
def test_area_total_nao_positiva_retorna_422(imagem_marcada, area_total):
    assert _enviar(imagem_marcada, area_total).status_code == 422
