import os
import json
import joblib
import pandas as pd

from dados import FEATURES, INFO_COLUNAS

# Caminhos para os arquivos do modelo treinado e seus metadados na pasta models/ (fora da pasta projeto/)
DIR_PROJETO = os.path.dirname(os.path.abspath(__file__))
DIR_RAIZ = os.path.dirname(DIR_PROJETO)
DIR_MODELS = os.path.join(DIR_RAIZ, "models")


# Função que carrega o modelo treinado (.joblib) e o arquivo de metadados (info_modelo.json)
def carregarModelo():
    caminho_mod = os.path.join(DIR_MODELS, "modelo.joblib")
    caminho_inf = os.path.join(DIR_MODELS, "info_modelo.json")

    # Caso o modelo ainda não tenha sido gerado em disco, executa o treinamento automaticamente
    if not os.path.exists(caminho_mod) or not os.path.exists(caminho_inf):
        from treinar import treinarModelos
        treinarModelos()

    modelo = joblib.load(caminho_mod)
    with open(caminho_inf, "r", encoding="utf-8") as f:
        info = json.load(f)

    return modelo, info


# Função que valida os dados enviados pelo formulário HTML, convertendo para float e checando as faixas permitidas
def validarEntrada(form):
    valores = {}
    erros = []

    for col in FEATURES:
        meta = INFO_COLUNAS[col]
        valor_bruto = form.get(col, "")

        # Verifica se o campo foi deixado em branco
        if valor_bruto is None or str(valor_bruto).strip() == "":
            erros.append(f"O campo '{meta['nome']}' é obrigatório e não pode ficar vazio.")
            continue

        # Permite tanto vírgula quanto ponto como separador decimal
        texto_limpo = str(valor_bruto).strip().replace(",", ".")

        # Tenta converter para número de ponto flutuante (float)
        try:
            numero = float(texto_limpo)
        except ValueError:
            erros.append(
                f"Valor inválido em '{meta['nome']}': '{valor_bruto}' não é um número válido."
            )
            continue

        # Verifica se o valor está dentro da faixa química/física aceita para o vinho
        min_val = meta["min"]
        max_val = meta["max"]
        if numero < min_val or numero > max_val:
            erros.append(
                f"O valor de '{meta['nome']}' ({numero} {meta['unidade']}) está fora da faixa permitida "
                f"(de {min_val} a {max_val} {meta['unidade']})."
            )
            continue

        valores[col] = numero

    if erros:
        return None, " ".join(erros)

    return valores, None


# Função que monta um DataFrame de 1 linha na ordem exata de FEATURES e executa predict() e predict_proba()
def prever(valores):
    modelo, info = carregarModelo()

    # Cria um DataFrame de 1 linha preservando os nomes e a ordem das colunas usadas no treinamento
    df_entrada = pd.DataFrame([valores], columns=FEATURES)

    # Realiza a previsão da classe (0 = Qualidade comum, 1 = Alta qualidade)
    classe_num = int(modelo.predict(df_entrada)[0])

    # Calcula as probabilidades de cada classe usando predict_proba
    probabilidades = modelo.predict_proba(df_entrada)[0]
    prob_comum = round(float(probabilidades[0]) * 100, 2)
    prob_alta = round(float(probabilidades[1]) * 100, 2)

    if classe_num == 1:
        rotulo = "Alta qualidade"
        prob_principal = prob_alta
    else:
        rotulo = "Qualidade comum"
        prob_principal = prob_comum

    return {
        "classe_num": classe_num,
        "classe": rotulo,
        "probabilidade": prob_principal,
        "prob_alta": prob_alta,
        "prob_comum": prob_comum,
        "modelo_usado": info.get("modelo_escolhido", "Random Forest"),
        "base_usada": info.get("base_usada", "data/winequality-red.csv"),
        "versao_modelo": info.get("versao_modelo", "1.0")
    }
