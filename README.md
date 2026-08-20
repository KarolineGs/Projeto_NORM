# Predição de Risco de NORM

Projeto de Machine Learning para análise e predição da ocorrência de NORM
(Naturally Occurring Radioactive Material) em plataformas, utilizando
variáveis químicas associadas à água produzida.

## Objetivo

O projeto utiliza dados históricos para estimar a probabilidade de ocorrência
de NORM a partir das seguintes variáveis:

- Salinidade
- Bário
- Estrôncio

O modelo final utiliza Regressão Logística.

## Estrutura do projeto

```text
PROJETO_NORM/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── outputs/
│   ├── figures/
│   └── models/
│
├── src/
│   ├── __init__.py
│   ├── pre_processamento.py
│   ├── processing.py
│   ├── modelling.py
│   └── graficos.py
│
├── tests/
│
├── main.py
├── train.py
├── predict.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Pipeline

O fluxo de treinamento é:

```text
Dados brutos
    ↓
Pré-processamento
    ↓
Base analítica
    ↓
Preparação dos dados
    ↓
Transformação log10
    ↓
StandardScaler
    ↓
Regressão Logística
    ↓
Modelo treinado
```

O pipeline do modelo contém:

1. transformação logarítmica (`log10`);
2. padronização com `StandardScaler`;
3. classificação com `LogisticRegression`.

Isso garante que as mesmas transformações utilizadas durante o treinamento
também sejam aplicadas durante a inferência.

## Ambiente

Criar um ambiente virtual:

```bash
python -m venv .venv
```

Ativar no Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Instalar as dependências:

```bash
pip install -r requirements.txt
```

## Treinamento

Para executar o pipeline de treinamento:

```bash
python train.py
```

O modelo treinado é salvo em:

```text
outputs/models/modelo_norm.joblib
```

## Inferência

Uma previsão pode ser realizada pelo `predict.py`.

Exemplo:

```bash
python predict.py --salinidade 100000 --bario 20 --estroncio 150
```

A saída contém:

```text
Classe prevista
Probabilidade sem NORM
Probabilidade com NORM
```

As três variáveis precisam receber valores numéricos maiores que zero.

## Pipeline de inferência

```text
Salinidade
Bário
Estrôncio
    ↓
Validação
    ↓
Pipeline treinado
    ↓
log10
    ↓
StandardScaler
    ↓
LogisticRegression
    ↓
Probabilidade de NORM
```

## Execução analítica

O arquivo:

```text
main.py
```

executa o pipeline analítico completo, incluindo processamento, modelagem,
avaliação e geração dos gráficos.

O arquivo:

```text
train.py
```

é responsável pelo fluxo operacional de treinamento.

O arquivo:

```text
predict.py
```

é responsável pelo fluxo de inferência utilizando um modelo previamente
treinado.

## Artefatos

Os principais artefatos gerados pelo projeto são armazenados em:

```text
outputs/
├── figures/
└── models/
```

Esses artefatos não são versionados diretamente pelo Git.

## Próximas etapas

O projeto está sendo preparado para execução em ambiente de produção.

Próximas etapas:

- containerização com Docker;
- publicação da imagem no Google Artifact Registry;
- treinamento no Google Cloud;
- integração com BigQuery;
- deploy do modelo;
- CI/CD;
- monitoramento.

## Tecnologias

- Python
- pandas
- NumPy
- scikit-learn
- Joblib
- Matplotlib
- Plotly
- GeoPandas
- Google Cloud Platform
