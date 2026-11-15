import os
import time
from flask import Flask, render_template, request, flash
from werkzeug.utils import secure_filename

from dados import (
    FEATURES,
    COL_QUALIDADE,
    LIMITE_ALTA,
    INFO_COLUNAS,
    DIR_GRAFICOS,
    lerCsv,
    validar,
    limpar,
    criarAlvo,
    gerarAchados,
    gerarGraficos
)
from modelo import carregarModelo, validarEntrada, prever

# Caminhos de diretórios do projeto (models/ e static/ ficam na raiz, fora da pasta projeto/)
DIR_PROJETO = os.path.dirname(os.path.abspath(__file__))
DIR_RAIZ = os.path.dirname(DIR_PROJETO)
DIR_STATIC = os.path.join(DIR_RAIZ, "static")
DIR_UPLOADS = os.path.join(DIR_RAIZ, "data", "uploads")
CAMINHO_CSV_PADRAO = os.path.join(DIR_RAIZ, "data", "winequality-red.csv")

app = Flask(__name__, static_folder=DIR_STATIC)
# Define secret_key obrigatória para utilização do sistema de mensagens flash() do Flask
app.secret_key = "chave_secreta_qualidade_vinho_uci_2026"

# Garante que as pastas de uploads e de gráficos existam
os.makedirs(DIR_UPLOADS, exist_ok=True)
os.makedirs(DIR_GRAFICOS, exist_ok=True)

# Armazena o caminho do último CSV válido enviado pelo usuário (inicia com a base padrão de data/)
ESTADO_ATUAL = {
    "caminho_csv": CAMINHO_CSV_PADRAO,
    "nome_arquivo": "winequality-red.csv (Base Padrão)"
}

# Gera os gráficos na pasta static/graficos/ assim que a aplicação é iniciada
if os.path.exists(CAMINHO_CSV_PADRAO):
    _df_inicial = lerCsv(CAMINHO_CSV_PADRAO)
    if _df_inicial is not None and not validar(_df_inicial):
        _df_limpo_inicial, _ = limpar(_df_inicial)
        gerarGraficos(_df_limpo_inicial)


# Rota inicial (/): recebe o upload de CSV, salva o original sem alterá-lo em data/uploads/, valida e mostra amostra
@app.route("/", methods=["GET", "POST"])
def inicio():
    amostra_colunas = None
    amostra_linhas = None
    qtd_linhas = 0
    qtd_colunas = 0
    nome_arquivo_atual = None
    sucesso_upload = False

    if request.method == "POST":
        # Permite também carregar a base padrão diretamente caso o usuário clique no botão de demonstração
        if request.form.get("usar_padrao") == "1":
            df = lerCsv(CAMINHO_CSV_PADRAO)
            erros = validar(df)
            if not erros:
                ESTADO_ATUAL["caminho_csv"] = CAMINHO_CSV_PADRAO
                ESTADO_ATUAL["nome_arquivo"] = "winequality-red.csv (Base Padrão)"
                df_limpo_padrao, _ = limpar(df)
                gerarGraficos(df_limpo_padrao)
                amostra_colunas = list(df.columns)
                amostra_linhas = df.head(10).to_dict(orient="records")
                qtd_linhas = int(len(df))
                qtd_colunas = int(len(df.columns))
                nome_arquivo_atual = ESTADO_ATUAL["nome_arquivo"]
                sucesso_upload = True
                flash("Base padrão 'winequality-red.csv' carregada e validada com sucesso!", "sucesso")
        else:
            arquivo = request.files.get("arquivo")

            # 1. Verifica se algum arquivo foi selecionado no formulário
            if not arquivo or not arquivo.filename or arquivo.filename.strip() == "":
                flash("Erro: Nenhum arquivo selecionado. Por favor, escolha um arquivo .csv para enviar.", "erro")
            # 2. Verifica a extensão do arquivo (.csv)
            elif not arquivo.filename.lower().endswith(".csv"):
                flash(
                    f"Erro de formato: O arquivo '{arquivo.filename}' não é um arquivo CSV (.csv). Envie apenas arquivos com extensão .csv.",
                    "erro"
                )
            else:
                # Salva o arquivo original em data/uploads/ preservando seu conteúdo intacto
                nome_seguro = secure_filename(arquivo.filename)
                if not nome_seguro:
                    nome_seguro = "upload_vinho.csv"
                caminho_salvo = os.path.join(DIR_UPLOADS, nome_seguro)
                arquivo.save(caminho_salvo)

                # Lê e valida o arquivo salvo
                df = lerCsv(caminho_salvo)
                erros = validar(df)

                if erros:
                    for erro in erros:
                        flash(erro, "erro")
                else:
                    # Atualiza o CSV ativo e já gera os novos gráficos em static/graficos/
                    ESTADO_ATUAL["caminho_csv"] = caminho_salvo
                    ESTADO_ATUAL["nome_arquivo"] = nome_seguro
                    df_limpo_upload, _ = limpar(df)
                    gerarGraficos(df_limpo_upload)

                    amostra_colunas = list(df.columns)
                    amostra_linhas = df.head(10).to_dict(orient="records")
                    qtd_linhas = int(len(df))
                    qtd_colunas = int(len(df.columns))
                    nome_arquivo_atual = nome_seguro
                    sucesso_upload = True
                    flash(
                        f"Arquivo '{nome_seguro}' validado com sucesso! As análises e gráficos foram atualizados.",
                        "sucesso"
                    )

    return render_template(
        "index.html",
        features=FEATURES,
        col_qualidade=COL_QUALIDADE,
        limite_alta=LIMITE_ALTA,
        info_colunas=INFO_COLUNAS,
        sucesso_upload=sucesso_upload,
        nome_arquivo=nome_arquivo_atual,
        qtd_linhas=qtd_linhas,
        qtd_colunas=qtd_colunas,
        amostra_colunas=amostra_colunas,
        amostra_linhas=amostra_linhas
    )


# Rota de análise (/analise): limpa o arquivo enviado, gera os 3 gráficos (via dados.py) e renderiza com relatório e info_modelo.json
@app.route("/analise", methods=["POST", "GET"])
def analises():
    caminho_csv = ESTADO_ATUAL.get("caminho_csv", CAMINHO_CSV_PADRAO)
    nome_arquivo = ESTADO_ATUAL.get("nome_arquivo", "winequality-red.csv (Base Padrão)")

    if not os.path.exists(caminho_csv):
        caminho_csv = CAMINHO_CSV_PADRAO
        nome_arquivo = "winequality-red.csv (Base Padrão)"

    df_bruto = lerCsv(caminho_csv)
    erros = validar(df_bruto)
    if erros:
        flash("O arquivo selecionado continha erros; exibindo análise da base padrão.", "erro")
        df_bruto = lerCsv(CAMINHO_CSV_PADRAO)
        nome_arquivo = "winequality-red.csv (Base Padrão)"

    # Executa limpeza, gera achados estatísticos e salva os 3 gráficos atualizados usando funções de dados.py
    df_limpo, relatorio = limpar(df_bruto)
    achados = gerarAchados(df_limpo)
    gerarGraficos(df_limpo)

    # Calcula tabela resumo por classe (groupby) para exibir na página de análise
    df_temp = df_limpo.copy()
    df_temp["classe_alta"] = criarAlvo(df_temp)
    medias_por_classe = df_temp.groupby("classe_alta")[FEATURES].mean().round(4).to_dict(orient="index")
    medianas_por_classe = df_temp.groupby("classe_alta")[FEATURES].median().round(4).to_dict(orient="index")

    # Carrega os metadados do treinamento e comparação de modelos (info_modelo.json)
    _, info_modelo = carregarModelo()

    return render_template(
        "analise.html",
        nome_arquivo=nome_arquivo,
        relatorio=relatorio,
        achados=achados,
        medias_por_classe=medias_por_classe,
        medianas_por_classe=medianas_por_classe,
        info_colunas=INFO_COLUNAS,
        features=FEATURES,
        limite_alta=LIMITE_ALTA,
        info_modelo=info_modelo,
        versao_grafico=int(time.time())
    )


# Rota de previsão (/previsao): exibe o formulário das 4 variáveis X e apresenta a previsão com probabilidade
@app.route("/previsao", methods=["POST", "GET"])
def previsao():
    _, info_modelo = carregarModelo()
    resultado = None
    valores_form = {col: "" for col in FEATURES}

    if request.method == "POST":
        # Preserva os valores digitados pelo usuário no formulário
        for col in FEATURES:
            valores_form[col] = request.form.get(col, "")

        valores_validos, erro = validarEntrada(request.form)
        if erro:
            flash(erro, "erro")
        else:
            resultado = prever(valores_validos)

    return render_template(
        "previsao.html",
        features=FEATURES,
        info_colunas=INFO_COLUNAS,
        limite_alta=LIMITE_ALTA,
        info_modelo=info_modelo,
        valores_form=valores_form,
        resultado=resultado
    )


if __name__ == "__main__":
    app.run(debug=True)