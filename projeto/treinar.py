import os
import json
from datetime import datetime
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

from dados import FEATURES, COL_QUALIDADE, LIMITE_ALTA, lerCsv, limpar, criarAlvo

# Diretórios base do projeto para garantir funcionamento independente da pasta de execução
DIR_PROJETO = os.path.dirname(os.path.abspath(__file__))
DIR_RAIZ = os.path.dirname(DIR_PROJETO)

CAMINHO_CSV_PADRAO = os.path.join(DIR_RAIZ, "data", "winequality-red.csv")
DIR_MODELS = os.path.join(DIR_RAIZ, "models")

# Carrega e limpa a base padrão para definir X e y globalmente
_df_bruto = lerCsv(CAMINHO_CSV_PADRAO)
if _df_bruto is not None and not _df_bruto.empty:
    df_base, relatorio_treino = limpar(_df_bruto)
    X = df_base[FEATURES]
    y = criarAlvo(df_base)
else:
    df_base, relatorio_treino = None, {}
    X, y = None, None


# Função que realiza a divisão estratificada em treino (80%) e teste (20%) prevenindo vazamento de dados
def trainTestSplit(test_size=0.20, random_state=42, stratify=y):
    # Utiliza train_test_split do scikit-learn mantendo a proporção das classes (stratify)
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )


# Função que avalia um modelo treinado no conjunto de teste e retorna dicionário com métricas e matriz de confusão
def avaliarModelo(nome, descricao, modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)

    acuracia = float(accuracy_score(y_test, y_pred))
    precisao = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    matriz = confusion_matrix(y_test, y_pred).tolist()

    # Extrai Verdadeiros Negativos (VN), Falsos Positivos (FP), Falsos Negativos (FN) e Verdadeiros Positivos (VP)
    vn, fp = int(matriz[0][0]), int(matriz[0][1])
    fn, vp = int(matriz[1][0]), int(matriz[1][1])

    return {
        "nome": nome,
        "descricao": descricao,
        "acuracia": round(acuracia * 100, 2),
        "precisao": round(precisao * 100, 2),
        "recall": round(recall * 100, 2),
        "f1": round(f1 * 100, 2),
        "matriz_confusao": matriz,
        "vn": vn,
        "fp": fp,
        "fn": fn,
        "vp": vp
    }


# Função principal que treina Decision Tree e Random Forest, compara métricas e salva o melhor modelo e metadados
def treinarModelos():
    global df_base, relatorio_treino, X, y

    # Recarrega e limpa os dados caso necessário
    df_bruto = lerCsv(CAMINHO_CSV_PADRAO)
    df_base, relatorio_treino = limpar(df_bruto)
    X = df_base[FEATURES]
    y = criarAlvo(df_base)

    test_size = 0.20
    random_state = 42

    # Divide a base limpa em 80% treino e 20% teste com estratificação pela classe alvo
    X_train, X_test, y_train, y_test = trainTestSplit(
        test_size=test_size, random_state=random_state, stratify=y
    )

    # 1. Treinamento do Modelo 1: Decision Tree Classifier (conforme exemplo mínimo da atividade: max_depth=4)
    modelo_dt = DecisionTreeClassifier(max_depth=4, random_state=random_state)
    modelo_dt.fit(X_train, y_train)
    metricas_dt = avaliarModelo(
        nome="Decision Tree (Árvore de Decisão)",
        descricao="Árvore de decisão única com profundidade máxima (max_depth=4, random_state=42).",
        modelo=modelo_dt,
        X_test=X_test,
        y_test=y_test
    )

    # 2. Treinamento do Modelo 2: Random Forest Classifier
    modelo_rf = RandomForestClassifier(
        n_estimators=100, random_state=random_state
    )
    modelo_rf.fit(X_train, y_train)
    metricas_rf = avaliarModelo(
        nome="Random Forest (Floresta Aleatória)",
        descricao="Comitê (ensemble) de 100 árvores de decisão independentes (n_estimators=100, random_state=42).",
        modelo=modelo_rf,
        X_test=X_test,
        y_test=y_test
    )

    # Cálculo da linha de base (Baseline): proporção da classe mais frequente ("Qualidade Comum" = 0) no teste
    total_teste = int(len(y_test))
    qtd_comum_teste = int((y_test == 0).sum())
    qtd_alta_teste = int((y_test == 1).sum())
    perc_comum_teste = round((qtd_comum_teste / total_teste) * 100, 2)
    perc_alta_teste = round((qtd_alta_teste / total_teste) * 100, 2)

    # Seleciona o melhor modelo comparando F1-Score e Acurácia no conjunto de teste
    if (metricas_rf["f1"], metricas_rf["acuracia"]) >= (metricas_dt["f1"], metricas_dt["acuracia"]):
        melhor_modelo = modelo_rf
        melhor_info = metricas_rf
        outro_info = metricas_dt
    else:
        melhor_modelo = modelo_dt
        melhor_info = metricas_dt
        outro_info = metricas_rf

    # Monta o JSON de metadados com todas as informações exigidas pela página analise.html e previsao.html
    info_modelo = {
        "base_usada": "data/winequality-red.csv (UCI Wine Quality - Vinhos Tintos)",
        "versao_modelo": "1.0",
        "data_treino": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "features": FEATURES,
        "coluna_alvo": COL_QUALIDADE,
        "limite_alta": LIMITE_ALTA,
        "proporcao_treino": "80%",
        "proporcao_teste": "20%",
        "test_size": test_size,
        "random_state": random_state,
        "linhas_brutas_treino": relatorio_treino["linhas_antes"],
        "linhas_limpas_treino": relatorio_treino["linhas_depois"],
        "duplicatas_removidas_treino": relatorio_treino["duplicatas_removidas"],
        "tamanho_treino": int(len(X_train)),
        "tamanho_teste": total_teste,
        "qtd_comum_teste": qtd_comum_teste,
        "qtd_alta_teste": qtd_alta_teste,
        "perc_comum_teste": perc_comum_teste,
        "perc_alta_teste": perc_alta_teste,
        "linha_de_base": perc_comum_teste,
        "modelos": [metricas_dt, metricas_rf],
        "modelo_escolhido": melhor_info["nome"],
        "metricas_escolhido": melhor_info,
        "justificativa_escolha": (
            f"O modelo {melhor_info['nome']} foi selecionado para a aplicação final por obter o melhor desempenho "
            f"no conjunto de teste, alcançando Acurácia de {melhor_info['acuracia']}% (superando a linha de base de {perc_comum_teste}%), "
            f"Precisão de {melhor_info['precisao']}%, Recall de {melhor_info['recall']}% e F1-Score de {melhor_info['f1']}% "
            f"(contra Acurácia de {outro_info['acuracia']}%, Recall de {outro_info['recall']}% e F1-Score de {outro_info['f1']}% do modelo {outro_info['nome']}). "
            f"Além de identificar mais vinhos de alta qualidade ({melhor_info['vp']} Verdadeiros Positivos contra {outro_info['vp']}), "
            f"o modelo oferece estimativas de probabilidade (predict_proba) consistentes para o formulário interativo."
        ),
        "interpretacao_erros": (
            f"Na classificação de vinhos, um Falso Positivo (FP) ocorre quando um vinho comum (quality < 7) é previsto como 'Alta Qualidade', "
            f"gerando risco à reputação caso um lote mediano seja comercializado como premium. Já um Falso Negativo (FN) ocorre quando um vinho "
            f"realmente superior (quality >= 7) é classificado como comum, representando perda de oportunidade comercial (e ocorre com maior frequência "
            f"porque apenas {perc_alta_teste}% da base pertence à classe alta). No conjunto de teste ({total_teste} vinhos únicos), "
            f"o Random Forest registrou {metricas_rf['fp']} Falsos Positivos e {metricas_rf['fn']} Falsos Negativos (acertando {metricas_rf['vp']} de {qtd_alta_teste} vinhos superiores), "
            f"enquanto a Decision Tree registrou {metricas_dt['fp']} Falsos Positivos e {metricas_dt['fn']} Falsos Negativos (acertando {metricas_dt['vp']} de {qtd_alta_teste} vinhos superiores)."
        )
    }

    # Salva o modelo (.joblib) e os metadados (.json) na pasta models/ na raiz do projeto (fora de projeto/)
    os.makedirs(DIR_MODELS, exist_ok=True)
    joblib.dump(melhor_modelo, os.path.join(DIR_MODELS, "modelo.joblib"))
    with open(os.path.join(DIR_MODELS, "info_modelo.json"), "w", encoding="utf-8") as f:
        json.dump(info_modelo, f, ensure_ascii=False, indent=4)

    return info_modelo


if __name__ == "__main__":
    resultado = treinarModelos()
    print("Treinamento concluído com sucesso!")
    print(f"Modelo escolhido: {resultado['modelo_escolhido']}")
    for m in resultado["modelos"]:
        print(f" - {m['nome']}: Acurácia={m['acuracia']}%, Recall={m['recall']}%, F1={m['f1']}%")
