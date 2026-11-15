# Qualidade de Vinhos (UCI Wine Quality)

Aplicação web desenvolvida em **Python** com **Flask**, **Pandas**, **Matplotlib** e **Scikit-Learn** para análise exploratória, limpeza de dados, visualização gráfica e classificação supervisionada da qualidade de vinhos tintos a partir de propriedades físico-químicas.

**Integrantes:** Afonso Guimarães, Eduardo Xavier e Luis Arthur.

---

## Estrutura do Projeto

```text
Qualidade-de-Vinho-Topicos-Especiais-A1/
├── data/
│   ├── winequality-red.csv       # Base de dados padrão (UCI Wine Quality - Red Wine)
│   └── uploads/                  # Arquivos CSV enviados pelos usuários via interface
├── models/
│   ├── modelo.joblib             # Modelo de Machine Learning treinado e salvo
│   └── info_modelo.json          # Metadados do treino, métricas e matrizes de confusão
├── static/
│   └── graficos/                 # Gráficos gerados automaticamente pelo Matplotlib
├── projeto/
│   ├── app.py                    # Rotas Flask (/, /analise, /previsao)
│   ├── dados.py                  # Leitura, validação, limpeza, achados e geração de gráficos
│   ├── treinar.py                # Divisão treino/teste, treinamento e comparação dos modelos
│   ├── modelo.py                 # Carregamento do modelo, validação do formulário e previsão
│   └── templates/
│       ├── index.html            # Página de Upload de CSV, Validação e Amostra
│       ├── analise.html          # Página de Limpeza, Achados, Gráficos e Comparação de Modelos
│       └── previsao.html         # Página de Formulário Interativo e Previsão (predict_proba)
├── requirements.txt              # Bibliotecas necessárias para rodar o projeto
├── .gitignore                    # Arquivo que lista arquivos e diretórios que devem ser ignorados pelo Git
└── README.md                     # Documentação do projeto
```

---

## Instalação e Execução (Passo a Passo)

Siga os passos abaixo no terminal (**PowerShell** ou **Prompt de Comando**) a partir da pasta raiz `Python-QUALIDADE-VINHO`:

### 1. Criar e ativar o ambiente virtual (opcional, mas recomendado)

**No Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**No Windows (CMD):**
```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
```

### 2. Instalar as dependências do projeto

Instale todas as bibliotecas listadas no arquivo `requirements.txt`:
```powershell
pip install -r requirements.txt
```
*(Bibliotecas instaladas: `flask`, `pandas`, `scikit-learn`, `matplotlib` e `joblib`).*

### 3. Treinar os modelos de Machine Learning

O repositório já acompanha o modelo treinado em `models/modelo.joblib`, mas caso queira executar novamente o treinamento e a comparação dos modelos do zero:
```powershell
python projeto/treinar.py
```

### 4. Iniciar a aplicação Flask

Execute o arquivo principal do servidor web:
```powershell
python projeto/app.py
```

### 5. Acessar no navegador

Com o servidor rodando no terminal, abra o navegador e acesse:
```text
http://127.0.0.1:5000
```

- **Página 1 (`/`) — Upload e Validação:** Envie um arquivo `.csv` ou clique em **"Carregar Base Padrão (UCI)"** para visualizar a amostra dos dados.
- **Página 2 (`/analise`) — Análise, Gráficos e Modelos:** Veja o relatório antes/depois da limpeza, os achados com `groupby`, os 3 gráficos atualizados e a comparação dos modelos.
- **Página 3 (`/previsao`) — Previsão no Navegador:** Preencha os 4 atributos químicos do vinho no formulário para prever se ele é de **Alta qualidade** ou **Qualidade comum** com a respectiva probabilidade.

---

## Esquema do CSV Aceito, Validação e Limpeza

A aplicação aceita arquivos `.csv` separados por **vírgula (`,`)** ou **ponto e vírgula (`;`)**. O arquivo enviado é preservado intacto em `data/uploads/` e deve conter obrigatoriamente as **5 colunas numéricas** abaixo:

| Coluna Obrigatória | Papel | Unidade | Faixa Física/Química Aceita | Descrição |
| :--- | :---: | :---: | :---: | :--- |
| `alcohol` | Entrada (`X`) | `% vol` | `8.0` a `16.0 % vol` | Graduação alcoólica percentual em volume. |
| `volatile acidity` | Entrada (`X`) | `g/L` | `0.08` a `2.00 g/L` | Quantidade de ácido acético (excesso gera gosto avinagrado). |
| `density` | Entrada (`X`) | `g/cm³` | `0.9800` a `1.0400 g/cm³` | Densidade da bebida (varia conforme álcool e açúcar residual). |
| `sulphates` | Entrada (`X`) | `g/L` | `0.20` a `2.50 g/L` | Aditivo que contribui para o $\text{SO}_2$ (conservante e antioxidante). |
| `quality` | Alvo original (`y`) | Nota (`0–10`) | `0` a `10` (inteiro) | Nota sensorial atribuída por especialistas. |

### Regras de Validação e Pipeline de Limpeza (`dados.py`)
1. **Validação no Upload (`validar`):** Verifica se o arquivo existe, se possui extensão `.csv`, se não está vazio, se o separador é válido (`,` ou `;`), se todas as 5 colunas obrigatórias existem e se todas contêm valores numéricos válidos.
2. **Remoção de Duplicatas (`limpar`):** Na base padrão (`1.599` registros brutos), são removidas **240 linhas duplicadas**, restando **1.359 registros únicos**. Remover duplicatas antes do `train_test_split` é essencial para evitar **vazamento de dados (data leakage)** entre treino e teste.
3. **Tratamento de Ausentes:** Linhas sem o rótulo `quality` são descartadas; eventuais valores ausentes nas variáveis `X` são preenchidos com a **mediana** da respectiva coluna.
4. **Remoção de Valores Impossíveis:** Filtra registros com `alcohol <= 0`, `density` fora de `0,98–1,04 g/cm³`, `volatile acidity <= 0`, `sulphates <= 0` ou `quality` fora do intervalo `0–10`.

---

## Definição de `X` (Variáveis de Entrada) e `y` (Alvo)

### Variáveis de Entrada (`X`)
Foram selecionadas 4 características físico-químicas medidas em laboratório disponíveis no momento da previsão:
1. **`alcohol` (Álcool):** Uvas mais maduras e processos de fermentação bem conduzidos geram maior teor alcoólico, fortemente associado a vinhos mais encorpados e bem avaliados (`11,52% vol` de média na classe alta contra `10,25% vol` na comum).
2. **`volatile acidity` (Acidez Volátil):** Mede a volatilidade ácida; valores altos indicam deterioração acética (aroma de vinagre), reduzindo drasticamente a nota do vinho (`0,405 g/L` na classe alta contra `0,547 g/L` na comum).
3. **`density` (Densidade):** Relaciona-se inversamente com o teor alcoólico e diretamente com o extrato seco e açúcar residual.
4. **`sulphates` (Sulfatos):** Protege o vinho contra oxidação e bactérias, preservando aromas e estabilidade (`0,743 g/L` na classe alta contra `0,645 g/L` na comum).

> **Prevenção de Vazamento:** A coluna `quality` nunca entra em `X`, pois ela é a própria resposta que origina o alvo `y`.

### Variável Alvo (`y`) e Justificativa do Limite (`quality >= 7`)
- **`y = 1` (Alta Qualidade):** `quality >= 7` (representa **13,5%** da base limpa — 184 vinhos).
- **`y = 0` (Qualidade Comum):** `quality < 7` (representa **86,5%** da base limpa — 1.175 vinhos, concentrados nas notas 5 e 6).
- **Justificativa:** Como mais de 80% dos vinhos recebem notas medianas (5 ou 6), o corte em `quality >= 7` permite distinguir os vinhos verdadeiramente superiores/premium dos vinhos cotidianos.

---

## Modelos Testados, Métricas, Comparação e Escolha Final

### Divisão de Treino e Teste
- Utilizou-se `train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)` sobre a base limpa (`1.359` registros):
  - **Conjunto de Treino (80%):** `1.087` vinhos.
  - **Conjunto de Teste (20%):** `272` vinhos (`235` de Qualidade Comum e `37` de Alta Qualidade).
- **Linha de Base (Baseline):** Como `86,40%` (`235 / 272`) dos exemplos de teste pertencem à classe mais frequente (*Qualidade Comum*), um classificador ingênuo que chutasse sempre "Qualidade Comum" teria **86,40% de acurácia**, porém **0% de Recall** para identificar vinhos de alta qualidade.

### Tabela Comparativa no Conjunto de Teste (`N = 272`)

| Modelo Testado | Configuração | Acurácia | Ganho sobre Baseline | Precisão (Classe Alta) | Recall (Classe Alta) | F1-Score (Classe Alta) | Matriz de Confusão `[[VN, FP], [FN, VP]]` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree (Árvore de Decisão)** | `max_depth=4, random_state=42` | **87,87%** | `+1,47 p.p.` | `57,69%` | **40,54%** | **47,62%** | `[[224, 11], [22, 15]]` |
| **Random Forest (Floresta Aleatória)** | `n_estimators=100, random_state=42` | `86,76%` | `+0,36 p.p.` | `52,00%` | `35,14%` | `41,94%` | `[[223, 12], [24, 13]]` |

### Interpretação dos Erros e Justificativa da Escolha Final
- **Significado dos Erros no Problema:**
  - **Falso Positivo (`FP`):** Classificar um vinho comum como sendo de *Alta Qualidade*. Impacta a confiança do consumidor caso um lote comum seja vendido como linha especial.
  - **Falso Negativo (`FN`):** Classificar um vinho de *Alta Qualidade* como comum. Representa perda de oportunidade comercial de valorizar um rótulo superior.
- **Justificativa do Modelo Escolhido (`Decision Tree Classifier`):**
  - O modelo **Decision Tree (`max_depth=4`)** foi escolhido para a aplicação final por superar o Random Forest em todas as métricas principais no conjunto de teste: maior **Acurácia (`87,87%` vs `86,76%`)**, maior **Precisão (`57,69%` vs `52,00%`)**, maior **Recall na classe de alta qualidade (`40,54%` vs `35,14%`)** e maior **F1-Score (`47,62%` vs `41,94%`)**, além de cometer menos Falsos Positivos (`11` contra `12`) e menos Falsos Negativos (`22` contra `24`). A poda em `max_depth=4` evitou o sobreajuste (*overfitting*) e garantiu boa capacidade de generalização.

---

## Limitações do Projeto

1. **Desequilíbrio Natural das Classes:** Apenas `13,5%` dos registros pertencem à classe *Alta Qualidade (`quality >= 7`)*, o que torna o Recall da classe minoritária naturalmente mais desafiador sem técnicas de reamostragem.
2. **Restrição a 4 Variáveis Físico-Químicas:** O modelo utiliza apenas `alcohol`, `volatile acidity`, `density` e `sulphates`. Outras variáveis químicas (como acidez cítrica, cloretos e dióxido de enxofre livre) e fatores enológicos externos (casta da uva, safra, região e tempo de barrica) não estão incluídos no preditor.
3. **Subjetividade do Alvo (`quality`):** A nota original deriva da mediana de avaliações sensoriais humanas, que podem conter variações subjetivas entre degustadores.
4. **Escopo Específico:** O modelo foi treinado com vinhos tintos da variante portuguesa *Vinho Verde* (base UCI), podendo não generalizar diretamente para vinhos brancos, espumantes ou de outras regiões sem novo treinamento.

---

## Fontes Utilizadas e Papel da Inteligência Artificial

### Fonte da Base de Dados (CSV)
- **Nome da Base:** UCI Wine Quality Dataset (`winequality-red.csv`)
- **Site / Repositório de onde o CSV foi obtido:** [www.kaggle.com/datasets/anairamcosta/winequality-red-csv]
- **Referência Original:** Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). *Modeling wine preferences by data mining from physicochemical properties*. Decision Support Systems, 47(4), 547-553.

### Papel da Inteligência Artificial no Projeto
- **Ferramenta de IA utilizada:** Assistente de IA - Gemini 4 Argon
- **Como a IA auxiliou o grupo:**
  - Estilização dos arquivos html na pasta de templates. Validação dos modelos de Machine Learning escolhidos pelo grupo. Análise dos melhores temas para os gráficos e auxílio na escolha dos tipos de gráficos. Auxílio na estruturação do README.md. Auxílio para escrever os comentários presentes nos códigos python e html.
- **Validação humana realizada pela equipe:**
  - A equipe escolheu o CSV da qualidade de vinhos no site Kaggle, realizou a limpeza e análise dos dados, escolheu os modelos de Machine Learning a serem comparados e validou os resultados obtidos pela IA. Validação dos temas de gráficos sugeridos pela IA e revisão completa do README.md conforme a estruturação sugerida pelo Gemini. Validação dos comentários presentes nos códigos python e html.

---

## Participação de Cada Integrante

- **Afonso Guimarães:** *Escolha do CSV no site Kaggle, Treinamento dos modelos de Machine Learning, carregamento e criação do formulário de previsão, criação e revisão do README.md*
- **Eduardo Xavier:** *Rotas Flask no app.py e estruturação dos arquivos html*
- **Luis Arthur:** *Leitura, validação, limpeza, gráficos, estruturação do dados.py*

**O projeto foi feito via chamada, com compartilhamento de tela. Portanto, todos os integrantes participaram de todas as etapas. Mas alguns focaram mais em uma etapa ao qual foi designada a ele. A escolha dos temas do gráfico foi feita em conjunto e com o auxilio de IA para saber se os temas faziam sentido serem aplicados ao projeto.**