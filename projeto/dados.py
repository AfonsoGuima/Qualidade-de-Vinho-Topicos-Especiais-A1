import os
import pandas as pd
import matplotlib
# Configura o backend 'Agg' do Matplotlib para gerar imagens sem precisar de interface gráfica (GUI)
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Caminhos de diretórios para salvar os gráficos na pasta static/graficos/ (fora da pasta projeto/)
DIR_PROJETO = os.path.dirname(os.path.abspath(__file__))
DIR_RAIZ = os.path.dirname(DIR_PROJETO)
DIR_GRAFICOS = os.path.join(DIR_RAIZ, "static", "graficos")

# Constantes compartilhadas que definem as variáveis de entrada (X), a coluna alvo original e o ponto de corte
FEATURES = ["alcohol", "volatile acidity", "density", "sulphates"]
COL_QUALIDADE = "quality"
LIMITE_ALTA = 7

# Metadados das colunas para exibição na interface e validação de faixas
INFO_COLUNAS = {
    "alcohol": {
        "nome": "Álcool (alcohol)",
        "unidade": "% vol",
        "min": 8.0,
        "max": 16.0,
        "passo": 0.1,
        "exemplo": 10.5,
        "explicacao": "Teor alcoólico do vinho (% em volume). Vinhos de safra superior frequentemente apresentam maior graduação alcoólica resultante de uvas mais maduras."
    },
    "volatile acidity": {
        "nome": "Acidez Volátil (volatile acidity)",
        "unidade": "g/L",
        "min": 0.08,
        "max": 2.00,
        "passo": 0.01,
        "exemplo": 0.45,
        "explicacao": "Quantidade de ácido acético no vinho. Em níveis elevados, causa um gosto desagradável de vinagre, reduzindo diretamente a nota sensorial."
    },
    "density": {
        "nome": "Densidade (density)",
        "unidade": "g/cm³",
        "min": 0.9800,
        "max": 1.0400,
        "passo": 0.0001,
        "exemplo": 0.9965,
        "explicacao": "Densidade da bebida próxima à da água, influenciada diretamente pelo percentual de álcool e pelo teor de açúcar residual."
    },
    "sulphates": {
        "nome": "Sulfatos (sulphates)",
        "unidade": "g/L",
        "min": 0.20,
        "max": 2.50,
        "passo": 0.01,
        "exemplo": 0.65,
        "explicacao": "Aditivo enológico que contribui para os níveis de dióxido de enxofre (SO₂), atuando como antimicrobiano e antioxidante para preservar o frescor."
    },
    "quality": {
        "nome": "Qualidade (quality)",
        "unidade": "nota (0 a 10)",
        "min": 0,
        "max": 10,
        "passo": 1,
        "exemplo": 7,
        "explicacao": "Nota sensorial atribuída por especialistas (de 0 a 10). Na aplicação, notas >= 7 definem a classe 'Alta Qualidade' (1) e notas < 7 definem 'Qualidade Comum' (0)."
    }
}


# Função responsável por ler um arquivo CSV tentando o separador ';' e depois ',', tratando arquivos vazios
def lerCsv(caminho):
    # Verifica se o caminho existe e se o arquivo não está vazio (0 bytes)
    if not caminho or not os.path.exists(caminho):
        return None
    if os.path.getsize(caminho) == 0:
        return pd.DataFrame()

    # Tenta ler primeiramente com ponto e vírgula (;)
    try:
        df_ponto_virgula = pd.read_csv(caminho, sep=";")
        # Se encontrou mais de 1 coluna, o separador ';' funcionou
        if df_ponto_virgula.shape[1] > 1:
            return df_ponto_virgula
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except Exception:
        pass

    # Caso contrário, tenta ler com vírgula (,)
    try:
        df_virgula = pd.read_csv(caminho, sep=",")
        return df_virgula
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except Exception:
        return None


# Função que valida o DataFrame lido, conferindo arquivo vazio, separador, colunas obrigatórias e tipos numéricos
def validar(df):
    erros = []

    # 1. Verifica se o DataFrame é nulo ou vazio
    if df is None:
        erros.append("Não foi possível ler o arquivo enviado. Verifique se é um arquivo CSV válido.")
        return erros

    if df.empty or len(df) == 0:
        erros.append("O arquivo enviado está vazio (não possui registros).")
        return erros

    # 2. Verifica se o separador usado no arquivo é inválido (apenas 1 coluna lida com cabeçalho agrupado)
    if df.shape[1] <= 1:
        erros.append(
            "Separador incorreto detectado! Utilize vírgula (,) ou ponto e vírgula (;) para separar as colunas do CSV."
        )
        return erros

    # 3. Verifica se todas as colunas obrigatórias (FEATURES + quality) estão presentes
    colunas_obrigatorias = FEATURES + [COL_QUALIDADE]
    colunas_presentes = [str(c).strip() for c in df.columns]
    faltantes = [col for col in colunas_obrigatorias if col not in colunas_presentes]

    if faltantes:
        erros.append(
            f"Colunas obrigatórias ausentes no CSV: {', '.join(faltantes)}. "
            f"O arquivo deve conter obrigatoriamente: {', '.join(colunas_obrigatorias)}."
        )
        return erros

    # 4. Verifica se os valores nas colunas obrigatórias são numéricos
    for col in colunas_obrigatorias:
        serie_original = df[col]
        # Converte para numérico transformando textos inválidos em NaN
        serie_numerica = pd.to_numeric(serie_original, errors="coerce")

        # Conta quantos valores não eram vazios originalmente, mas falharam na conversão numérica
        valores_nao_numericos = int((serie_original.notna() & serie_numerica.isna()).sum())
        if valores_nao_numericos > 0:
            erros.append(
                f"A coluna '{col}' contém {valores_nao_numericos} valor(es) não numérico(s) incompatível(is)."
            )
        elif serie_numerica.notna().sum() == 0:
            erros.append(f"A coluna '{col}' está inteiramente vazia (sem valores numéricos válidos).")

    return erros


# Função que realiza a limpeza dos dados: conversões, remoção de duplicatas, tratamento de ausentes e filtro de valores impossíveis
def limpar(df):
    df_limpo = df.copy()
    # Padroniza nomes de colunas removendo espaços extras nas pontas
    df_limpo.columns = [str(c).strip() for c in df_limpo.columns]

    colunas_obrigatorias = FEATURES + [COL_QUALIDADE]
    linhas_antes = int(len(df_limpo))
    colunas_total = int(len(df_limpo.columns))

    # 1. Conversão garantida das colunas obrigatórias para tipo numérico (float/int)
    for col in colunas_obrigatorias:
        df_limpo[col] = pd.to_numeric(df_limpo[col], errors="coerce")

    # 2. Remoção de linhas duplicadas (essencial antes do train_test_split para evitar vazamento de dados)
    duplicatas_removidas = int(df_limpo.duplicated().sum())
    df_limpo = df_limpo.drop_duplicates().copy()

    # 3. Tratamento de valores ausentes (NaN)
    ausentes_tratados = int(df_limpo[colunas_obrigatorias].isna().sum().sum())
    linhas_com_ausentes = int(df_limpo[colunas_obrigatorias].isna().any(axis=1).sum())

    # Se o alvo (quality) estiver ausente, a linha é removida para não inventar rótulos
    df_limpo = df_limpo.dropna(subset=[COL_QUALIDADE]).copy()

    # Para as variáveis de entrada (FEATURES), preenche eventuais ausentes com a mediana da coluna
    for col in FEATURES:
        if df_limpo[col].isna().any():
            mediana = df_limpo[col].median()
            df_limpo[col] = df_limpo[col].fillna(mediana)

    # 4. Remoção de valores fisicamente ou quimicamente impossíveis. Regras: álcool > 0 (e <= 25), densidade entre 0.98 e 1.04, quality entre 0 e 10, acidez e sulfatos > 0
    mascara_valida = (
        (df_limpo["alcohol"] > 0) & (df_limpo["alcohol"] <= 25) &
        (df_limpo["density"] >= 0.98) & (df_limpo["density"] <= 1.04) &
        (df_limpo[COL_QUALIDADE] >= 0) & (df_limpo[COL_QUALIDADE] <= 10) &
        (df_limpo["volatile acidity"] > 0) &
        (df_limpo["sulphates"] > 0)
    )
    impossiveis_removidos = int((~mascara_valida).sum())
    df_limpo = df_limpo[mascara_valida].copy()

    # Garante que a coluna quality seja inteira após a limpeza
    df_limpo[COL_QUALIDADE] = df_limpo[COL_QUALIDADE].astype(int)

    linhas_depois = int(len(df_limpo))

    # Monta o relatório detalhado da limpeza para exibição na página de análise
    relatorio = {
        "linhas_antes": linhas_antes,
        "linhas_depois": linhas_depois,
        "colunas": colunas_total,
        "duplicatas_removidas": duplicatas_removidas,
        "ausentes_tratados": ausentes_tratados,
        "linhas_com_ausentes": linhas_com_ausentes,
        "impossiveis_removidos": impossiveis_removidos,
        "total_removidos": linhas_antes - linhas_depois,
        "explicacoes": [
            f"Conversão de tipos: As colunas {', '.join(colunas_obrigatorias)} foram validadas e convertidas para tipos numéricos (float/int).",
            f"Remoção de duplicatas ({duplicatas_removidas} linha(s) removida(s)): A base original de vinhos possui registros repetidos. Removê-los antes da divisão de treino e teste evita vazamento de dados (data leakage) e impede que a acurácia fique artificialmente inflada.",
            f"Tratamento de ausentes ({ausentes_tratados} valor(es) tratado(s)): Registros sem a nota alvo ('{COL_QUALIDADE}') são descartados e valores faltantes nas variáveis de entrada (X) recebem a mediana da respectiva coluna.",
            f"Filtro de valores impossíveis ({impossiveis_removidos} linha(s) removida(s)): Foram descartados registros com álcool <= 0%, densidade fora da faixa física de 0,98 a 1,04 g/cm³, acidez/sulfatos <= 0 ou nota '{COL_QUALIDADE}' fora da escala de 0 a 10."
        ]
    }

    return df_limpo, relatorio


# Função que cria a variável alvo binária y: 1 para Alta Qualidade (quality >= 7) e 0 para Qualidade Comum (quality < 7)
def criarAlvo(df):
    return (df[COL_QUALIDADE] >= LIMITE_ALTA).astype(int)


# Alias em snake_case para manter compatibilidade com qualquer chamada a criar_alvo(df)
criar_alvo = criarAlvo


# Função que calcula estatísticas e agrupamentos (groupby) para gerar achados analíticos em texto sobre a base
def gerarAchados(df):
    df_temp = df.copy()
    df_temp["classe_alta"] = criarAlvo(df_temp)

    total = len(df_temp)
    qtd_alta = int((df_temp["classe_alta"] == 1).sum())
    qtd_comum = int((df_temp["classe_alta"] == 0).sum())
    perc_alta = (qtd_alta / total * 100) if total > 0 else 0.0
    perc_comum = (qtd_comum / total * 100) if total > 0 else 0.0

    # Agrupamento (groupby) por classe para comparar médias e medianas das variáveis X
    agrupado_media = df_temp.groupby("classe_alta")[FEATURES].mean()
    agrupado_mediana = df_temp.groupby("classe_alta")[FEATURES].median()

    alcool_comum = agrupado_media.loc[0, "alcohol"] if 0 in agrupado_media.index else 0.0
    alcool_alta = agrupado_media.loc[1, "alcohol"] if 1 in agrupado_media.index else 0.0
    mediana_alcool_comum = agrupado_mediana.loc[0, "alcohol"] if 0 in agrupado_mediana.index else 0.0
    mediana_alcool_alta = agrupado_mediana.loc[1, "alcohol"] if 1 in agrupado_mediana.index else 0.0

    acidez_comum = agrupado_media.loc[0, "volatile acidity"] if 0 in agrupado_media.index else 0.0
    acidez_alta = agrupado_media.loc[1, "volatile acidity"] if 1 in agrupado_media.index else 0.0

    sulfatos_comum = agrupado_media.loc[0, "sulphates"] if 0 in agrupado_media.index else 0.0
    sulfatos_alta = agrupado_media.loc[1, "sulphates"] if 1 in agrupado_media.index else 0.0

    densidade_comum = agrupado_media.loc[0, "density"] if 0 in agrupado_media.index else 0.0
    densidade_alta = agrupado_media.loc[1, "density"] if 1 in agrupado_media.index else 0.0

    achados = [
        (
            f"Desequilíbrio de Classes (Corte quality >= {LIMITE_ALTA}): Apenas {perc_alta:.1f}% dos vinhos ({qtd_alta} de {total} registros limpos) "
            f"atingem nota >= {LIMITE_ALTA} ('Alta Qualidade'), enquanto {perc_comum:.1f}% ({qtd_comum} registros) pertencem à classe 'Qualidade Comum' (notas < {LIMITE_ALTA}). "
            f"Esse ponto de corte é justificado para destacar vinhos realmente superiores da grande massa mediana."
        ),
        (
            f"Teor Alcoólico Maior em Vinhos Superiores: No agrupamento por classe, vinhos de 'Alta Qualidade' possuem teor alcoólico médio de "
            f"{alcool_alta:.2f}% vol (mediana {mediana_alcool_alta:.2f}% vol) em comparação aos vinhos de 'Qualidade Comum' "
            f"com média de {alcool_comum:.2f}% vol (mediana {mediana_alcool_comum:.2f}% vol)."
        ),
        (
            f"Acidez Volátil Elevada Penaliza a Qualidade: Vinhos comuns apresentam acidez volátil média de {acidez_comum:.3f} g/L, "
            f"enquanto nos vinhos de alta qualidade a média cai para {acidez_alta:.3f} g/L. Como o excesso de ácido acético gera aroma avinagrado, "
            f"valores baixos de acidez volátil são fortes indicadores de qualidade superior."
        ),
        (
            f"Sulfatos e Densidade como Diferenciais Químicos: Vinhos de alta qualidade exibem maior concentração média de sulfatos "
            f"({sulfatos_alta:.3f} g/L contra {sulfatos_comum:.3f} g/L nos comuns), favorecendo a conservação antioxidante, e densidade média ligeiramente menor "
            f"({densidade_alta:.5f} g/cm³ contra {densidade_comum:.5f} g/cm³), reflexo direto da maior graduação alcoólica."
        )
    ]

    return achados


# Função que gera as 3 visualizações relevantes com Matplotlib e salva em static/graficos/ (na raiz)
def gerarGraficos(df_limpo):
    os.makedirs(DIR_GRAFICOS, exist_ok=True)
    y_alvo = criarAlvo(df_limpo)

    # Configuração visual em tema escuro para os gráficos com destaque na cor vinho
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.facecolor": "#18181c",
        "axes.facecolor": "#121215",
        "axes.edgecolor": "#44444a",
        "axes.labelcolor": "#f5f5f7",
        "xtick.color": "#e0e0e5",
        "ytick.color": "#e0e0e5",
        "text.color": "#f5f5f7",
        "legend.facecolor": "#1f1f24",
        "legend.edgecolor": "#44444a"
    })

    # GRÁFICO 1: Distribuição das classes (barras) + Histograma de quality com limite 7
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    # Subplot 1A: Barras da distribuição binária das classes
    qtd_comum = int((y_alvo == 0).sum())
    qtd_alta = int((y_alvo == 1).sum())
    total = len(y_alvo)
    perc_comum = (qtd_comum / total * 100) if total > 0 else 0
    perc_alta = (qtd_alta / total * 100) if total > 0 else 0

    classes_nomes = ["Qualidade Comum\n(quality < 7)", "Alta Qualidade\n(quality ≥ 7)"]
    contagens = [qtd_comum, qtd_alta]
    cores_classes = ["#5c636a", "#800020"]

    barras = ax1.bar(classes_nomes, contagens, color=cores_classes, width=0.55, edgecolor="#f5f5f7")
    ax1.set_title("Distribuição das Classes do Alvo (y)", fontweight="bold")
    ax1.set_ylabel("Quantidade de Vinhos")
    ax1.grid(axis="y", linestyle="--", alpha=0.25, color="#888")

    for barra, qtd, perc in zip(barras, contagens, [perc_comum, perc_alta]):
        ax1.text(
            barra.get_x() + barra.get_width() / 2,
            barra.get_height() + (max(contagens) * 0.02),
            f"{qtd} ({perc:.1f}%)",
            ha="center",
            va="bottom",
            fontweight="bold",
            color="#ffffff"
        )
    ax1.set_ylim(0, max(contagens) * 1.15 if contagens else 10)

    # Subplot 1B: Histograma/Distribuição das notas de quality com linha vertical no limite 7
    contagem_notas = df_limpo[COL_QUALIDADE].value_counts().sort_index()
    notas = contagem_notas.index.tolist()
    valores_notas = contagem_notas.values.tolist()
    cores_notas = ["#800020" if n >= LIMITE_ALTA else "#5c636a" for n in notas]

    ax2.bar(notas, valores_notas, color=cores_notas, width=0.65, edgecolor="#f5f5f7")
    ax2.axvline(
        x=LIMITE_ALTA - 0.5,
        color="#ff4d6d",
        linestyle="--",
        linewidth=2.2,
        label=f"Limite Alta Qualidade (quality ≥ {LIMITE_ALTA})"
    )
    ax2.set_title("Distribuição da Nota Original (quality)", fontweight="bold")
    ax2.set_xlabel("Nota Sensorial (quality)")
    ax2.set_ylabel("Quantidade de Vinhos")
    ax2.set_xticks(notas)
    ax2.grid(axis="y", linestyle="--", alpha=0.25, color="#888")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    caminho_g1 = os.path.join(DIR_GRAFICOS, "distribuicao_classes.png")
    fig.savefig(caminho_g1, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)

    # GRÁFICO 2: Boxplot e Histograma de Álcool (alcohol) por Classe
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    alcool_comum = df_limpo.loc[y_alvo == 0, "alcohol"]
    alcool_alta = df_limpo.loc[y_alvo == 1, "alcohol"]

    # Subplot 2A: Boxplot de álcool por classe
    bp = ax1.boxplot(
        [alcool_comum, alcool_alta],
        tick_labels=["Qualidade Comum (< 7)", "Alta Qualidade (≥ 7)"],
        patch_artist=True,
        widths=0.45,
        boxprops=dict(edgecolor="#f5f5f7"),
        whiskerprops=dict(color="#f5f5f7"),
        capprops=dict(color="#f5f5f7"),
        flierprops=dict(markeredgecolor="#f5f5f7")
    )
    for patch, cor in zip(bp["boxes"], ["#5c636a", "#a4161a"]):
        patch.set_facecolor(cor)
        patch.set_alpha(0.9)
    for median in bp["medians"]:
        median.set_color("#ffffff")
        median.set_linewidth(2)

    ax1.set_title("Boxplot do Teor Alcoólico por Classe", fontweight="bold")
    ax1.set_ylabel("Álcool (% vol)")
    ax1.grid(axis="y", linestyle="--", alpha=0.25, color="#888")

    # Subplot 2B: Histograma de álcool por classe
    ax2.hist(
        alcool_comum,
        bins=18,
        alpha=0.65,
        color="#6c757d",
        edgecolor="#f5f5f7",
        label="Qualidade Comum (< 7)"
    )
    ax2.hist(
        alcool_alta,
        bins=18,
        alpha=0.85,
        color="#800020",
        edgecolor="#f5f5f7",
        label="Alta Qualidade (≥ 7)"
    )
    ax2.set_title("Histograma de Álcool (% vol) por Classe", fontweight="bold")
    ax2.set_xlabel("Álcool (% vol)")
    ax2.set_ylabel("Frequência")
    ax2.grid(axis="y", linestyle="--", alpha=0.25, color="#888")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    caminho_g2 = os.path.join(DIR_GRAFICOS, "alcool_por_classe.png")
    fig.savefig(caminho_g2, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)

    # GRÁFICO 3: Dispersão Álcool (alcohol) × Acidez Volátil (volatile acidity) por Classe
    fig, ax = plt.subplots(figsize=(9.5, 5.2))

    mascara_comum = (y_alvo == 0)
    mascara_alta = (y_alvo == 1)

    ax.scatter(
        df_limpo.loc[mascara_comum, "alcohol"],
        df_limpo.loc[mascara_comum, "volatile acidity"],
        c="#8a939b",
        alpha=0.55,
        s=36,
        label="Qualidade Comum (quality < 7)",
        edgecolors="none"
    )
    ax.scatter(
        df_limpo.loc[mascara_alta, "alcohol"],
        df_limpo.loc[mascara_alta, "volatile acidity"],
        c="#a4161a",
        alpha=0.95,
        s=55,
        label="Alta Qualidade (quality ≥ 7)",
        edgecolors="#ffffff",
        linewidths=0.5
    )

    ax.set_title(
        "Dispersão: Teor Alcoólico × Acidez Volátil Colorida por Classe",
        fontweight="bold"
    )
    ax.set_xlabel("Álcool (% vol)")
    ax.set_ylabel("Acidez Volátil (g/L)")
    ax.grid(True, linestyle="--", alpha=0.25, color="#888")
    ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    caminho_g3 = os.path.join(DIR_GRAFICOS, "dispersao_alcool_acidez.png")
    fig.savefig(caminho_g3, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)