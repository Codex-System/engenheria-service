import asyncio
from types import SimpleNamespace

import pytest

from engenharia_service.services import ai_service


def _resposta_da_openai(conteudo: str):
    mensagem = SimpleNamespace(content=conteudo)
    return SimpleNamespace(choices=[SimpleNamespace(message=mensagem)])


def _validar():
    return asyncio.run(
        ai_service.validar_com_ia(b"jpeg", cor="azul", proporcao=0.1, area_calculada=1.0, area_total=10.0)
    )


def test_retorna_laudo_quando_resposta_valida(monkeypatch):
    async def responder(**_):
        return _resposta_da_openai('{"aprovado": true, "observacoes_tecnicas": "ok"}')

    monkeypatch.setattr(ai_service.client.chat.completions, "create", responder)

    assert _validar()["aprovado"] is True


@pytest.mark.parametrize("conteudo", ["nao e json", '{"observacoes_tecnicas": "sem aprovado"}'])
def test_resposta_malformada_gera_auditoria_indisponivel(monkeypatch, conteudo):
    async def responder(**_):
        return _resposta_da_openai(conteudo)

    monkeypatch.setattr(ai_service.client.chat.completions, "create", responder)

    with pytest.raises(ai_service.AuditoriaIndisponivelError):
        _validar()


def test_falha_na_openai_gera_auditoria_indisponivel(monkeypatch):
    async def falhar(**_):
        raise RuntimeError("timeout")

    monkeypatch.setattr(ai_service.client.chat.completions, "create", falhar)

    with pytest.raises(ai_service.AuditoriaIndisponivelError):
        _validar()
