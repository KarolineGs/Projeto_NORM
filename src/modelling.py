
import logging
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer


logger = logging.getLogger(__name__)


# ==========================================================
# CONFIGURAÇÕES
# ==========================================================

COLUNA_ALVO = "TEM_NORM"

VARIAVEIS_MODELO = [
    "MEDIANA_SALINIDADE_PLAT",
    "MEDIANA_BARIO_PLAT",
    "MEDIANA_ESTRONCIO_PLAT",
]

def preparar_dados_modelo(
    df_analise: pd.DataFrame
) -> pd.DataFrame:
    """
    Prepara o dataframe de análise para a modelagem.

    Seleciona as variáveis químicas, remove valores
    inválidos e aplica transformação log10.

    Parameters
    ----------
    df_analise : pd.DataFrame
        Dataframe gerado pelo pipeline de processamento.

    Returns
    -------
    pd.DataFrame
        Base preparada para modelagem.
    """

    logger.info(
        "Preparando dados para modelagem"
    )

    colunas = [
        "LOCAL DA GERAÇÃO",
        COLUNA_ALVO,
        *VARIAVEIS_MODELO,
    ]

    dados = (
        df_analise[colunas]
        .copy()
    )

    # Nomes mais simples para modelagem
    dados = dados.rename(
        columns={
            "MEDIANA_SALINIDADE_PLAT": "SALINIDADE",
            "MEDIANA_BARIO_PLAT": "BARIO",
            "MEDIANA_ESTRONCIO_PLAT": "ESTRONCIO",
        }
    )

    variaveis = [
        "SALINIDADE",
        "BARIO",
        "ESTRONCIO",
    ]

    # Conversão numérica
    for coluna in [
        COLUNA_ALVO,
        *variaveis,
    ]:
        dados[coluna] = pd.to_numeric(
            dados[coluna],
            errors="coerce",
        )

    # Infinitos -> NaN
    dados = dados.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Remove incompletos
    dados = dados.dropna(
        subset=[
            COLUNA_ALVO,
            *variaveis,
        ]
    )

    # Log10 exige valores > 0
    dados = dados[
        (dados["SALINIDADE"] > 0)
        & (dados["BARIO"] > 0)
        & (dados["ESTRONCIO"] > 0)
    ].copy()

    dados[COLUNA_ALVO] = (
        dados[COLUNA_ALVO]
        .astype(int)
    )

    # Transformação logarítmica
    for coluna in variaveis:

        dados[f"LOG_{coluna}"] = np.log10(
            dados[coluna]
        )

    logger.info(
        "Dados preparados | %d plataformas",
        len(dados)
    )

    return dados.reset_index(drop=True)

def criar_modelo_logistico() -> Pipeline:
    """
    Cria o pipeline da regressão logística.

    Etapas:
    - transformação log10
    - padronização
    - regressão logística
    """

    modelo = Pipeline(
        steps=[
            (
                "log10",
                FunctionTransformer(
                    np.log10,
                    feature_names_out="one-to-one",
                ),
            ),
            (
                "padronizacao",
                StandardScaler(),
            ),
            (
                "modelo",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=42,
                ),
            ),
        ]
    )

    return modelo

def calcular_especificidade(
    y_real,
    y_previsto
) -> float:
    """
    Calcula a especificidade da classificação.
    """

    tn, fp, fn, tp = confusion_matrix(
        y_real,
        y_previsto,
        labels=[0, 1],
    ).ravel()

    if (tn + fp) == 0:
        return np.nan

    return tn / (tn + fp)

def avaliar_modelo(
    dados_modelo: pd.DataFrame,
    features: list[str],
    n_splits: int = 4,
    limite: float = 0.45,
) -> dict:
    """
    Avalia uma regressão logística utilizando
    validação cruzada estratificada.

    Parameters
    ----------
    dados_modelo : pd.DataFrame
        Base preparada para modelagem.

    features : list[str]
        Variáveis utilizadas pelo modelo.

    n_splits : int, padrão=5
        Número máximo de folds.

    limite : float, padrão=0.5
        Limite para classificação.

    Returns
    -------
    dict
        Métricas e previsões do modelo.
    """

    X = dados_modelo[features]

    y = dados_modelo[
        COLUNA_ALVO
    ]

    # Garante que não haja mais folds
    # que observações na menor classe
    menor_classe = int(
        y.value_counts().min()
    )

    folds = min(
        n_splits,
        menor_classe,
    )

    if folds < 2:
        raise ValueError(
            "Quantidade insuficiente de observações "
            "para validação cruzada."
        )

    cv = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=42,
    )

    modelo = criar_modelo_logistico()

    probabilidades = cross_val_predict(
        modelo,
        X,
        y,
        cv=cv,
        method="predict_proba",
    )[:, 1]

    previsoes = (
        probabilidades >= limite
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        previsoes,
        labels=[0, 1],
    ).ravel()

    resultado = {
        "AUC": roc_auc_score(
            y,
            probabilidades,
        ),

        "ACURACIA": accuracy_score(
            y,
            previsoes,
        ),

        "PRECISAO": precision_score(
            y,
            previsoes,
            zero_division=0,
        ),

        "SENSIBILIDADE": recall_score(
            y,
            previsoes,
            zero_division=0,
        ),

        "ESPECIFICIDADE": calcular_especificidade(
            y,
            previsoes,
        ),

        "F1_SCORE": f1_score(
            y,
            previsoes,
            zero_division=0,
        ),

        "VP": tp,
        "FP": fp,
        "VN": tn,
        "FN": fn,

        "probabilidades": probabilidades,
        "previsoes": previsoes,
        "features": features,
        "folds": folds,
    }

    return resultado

def comparar_modelos(
    dados_modelo: pd.DataFrame,
    n_splits: int = 4,
    limite: float = 0.45,
) -> tuple[pd.DataFrame, dict]:
    """
    Compara diferentes combinações de variáveis
    químicas utilizando regressão logística.

    Returns
    -------
    resultado_modelos : pd.DataFrame
        Métricas de todos os modelos.

    predicoes_modelos : dict
        Probabilidades e previsões de cada modelo.
    """

    logger.info(
        "Iniciando comparação dos modelos"
    )

    modelos = {
        "Salinidade": [
            "LOG_SALINIDADE",
        ],

        "Bário": [
            "LOG_BARIO",
        ],

        "Estrôncio": [
            "LOG_ESTRONCIO",
        ],

        "Salinidade + Bário": [
            "LOG_SALINIDADE",
            "LOG_BARIO",
        ],

        "Salinidade + Estrôncio": [
            "LOG_SALINIDADE",
            "LOG_ESTRONCIO",
        ],

        "Salinidade + Bário + Estrôncio": [
            "LOG_SALINIDADE",
            "LOG_BARIO",
            "LOG_ESTRONCIO",
        ],
    }

    resultados = []

    predicoes_modelos = {}

    for nome_modelo, features in modelos.items():

        logger.info(
            "Avaliando modelo: %s",
            nome_modelo
        )

        resultado = avaliar_modelo(
            dados_modelo=dados_modelo,
            features=features,
            n_splits=n_splits,
            limite=limite,
        )

        resultados.append({
            "MODELO": nome_modelo,
            "N_VARIAVEIS": len(features),
            "AUC": resultado["AUC"],
            "ACURACIA": resultado["ACURACIA"],
            "PRECISAO": resultado["PRECISAO"],
            "SENSIBILIDADE": resultado["SENSIBILIDADE"],
            "ESPECIFICIDADE": resultado["ESPECIFICIDADE"],
            "F1_SCORE": resultado["F1_SCORE"],
            "VP": resultado["VP"],
            "FP": resultado["FP"],
            "VN": resultado["VN"],
            "FN": resultado["FN"],
        })

        predicoes_modelos[
            nome_modelo
        ] = resultado

    resultado_modelos = (
        pd.DataFrame(resultados)
        .sort_values(
            "AUC",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    logger.info(
        "Comparação dos modelos finalizada"
    )

    return (
        resultado_modelos,
        predicoes_modelos,
    )

def treinar_modelo_final(
    dados_modelo: pd.DataFrame
) -> Pipeline:
    """
    Treina o modelo químico final utilizando
    todas as observações disponíveis.
    """

    features = [
        "SALINIDADE",
        "BARIO",
        "ESTRONCIO",
    ]

    X = dados_modelo[
        features
    ]

    y = dados_modelo[
        COLUNA_ALVO
    ]

    modelo = criar_modelo_logistico()

    modelo.fit(
        X,
        y,
    )

    logger.info(
        "Modelo final treinado | %d plataformas",
        len(dados_modelo)
    )

    return modelo

def obter_coeficientes(
    modelo: Pipeline
) -> pd.DataFrame:
    """
    Retorna coeficientes padronizados e odds ratio
    da regressão logística.
    """

    regressao = (
        modelo
        .named_steps["modelo"]
    )

    variaveis = [
        "SALINIDADE",
        "BARIO",
        "ESTRONCIO",
    ]

    coeficientes = pd.DataFrame({
        "VARIAVEL": variaveis,

        "COEFICIENTE_PADRONIZADO":
            regressao.coef_[0],

        "ODDS_RATIO":
            np.exp(
                regressao.coef_[0]
            ),
    })

    return coeficientes