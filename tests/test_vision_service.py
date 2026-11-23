import pytest

from engenharia_service.services.vision_service import calcular_area_por_pixels


def test_calcula_area_proporcional_a_marcacao(gerar_imagem):
    imagem = gerar_imagem(retangulos=[("vermelho", (0, 100, 0, 100))])

    resultado = calcular_area_por_pixels(imagem, area_total=40)

    assert resultado.cor_detectada == "vermelho"
    assert resultado.proporcao == pytest.approx(0.25)
    assert resultado.area_calculada == pytest.approx(10)


def test_ignora_pontos_soltos_de_outra_cor(gerar_imagem):
    pontos_azuis = [("azul", (y, y + 3, x, x + 3)) for y in range(0, 200, 10) for x in range(150, 200, 10)]
    imagem = gerar_imagem(retangulos=[("verde", (0, 50, 0, 50)), *pontos_azuis])

    resultado = calcular_area_por_pixels(imagem, area_total=100)

    assert resultado.cor_detectada == "verde"
    assert resultado.area_calculada == pytest.approx(6.25)


def test_fundo_de_tijolo_nao_e_confundido_com_marcacao(gerar_imagem):
    imagem = gerar_imagem(fundo="tijolo", retangulos=[("azul", (0, 40, 0, 40))])

    resultado = calcular_area_por_pixels(imagem, area_total=100)

    assert resultado.cor_detectada == "azul"
    assert resultado.area_calculada == pytest.approx(4)


def test_rejeita_imagem_sem_marcacao(gerar_imagem):
    with pytest.raises(ValueError, match="marcação"):
        calcular_area_por_pixels(gerar_imagem(), area_total=10)


def test_rejeita_bytes_que_nao_sao_imagem():
    with pytest.raises(ValueError, match="inválido"):
        calcular_area_por_pixels(b"isso nao e uma imagem", area_total=10)
