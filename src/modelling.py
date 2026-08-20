
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
    roc_curve,
)
from sklearn.model_selection import (
    StratifiedKFold,
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
    "RELACAO_BARIO_ESTRONCIO",
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
            "RELACAO_BARIO_ESTRONCIO": "RELACAO_BA_SR",
        }
    )

    variaveis = [
        "SALINIDADE",
        "BARIO",
        "ESTRONCIO",
        "RELACAO_BA_SR",
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
        & (dados["RELACAO_BA_SR"] > 0)
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

def calcular_limiar_youden(
    y_real,
    probabilidades,
) -> float:
    """Seleciona o limiar que maximiza sensibilidade + especificidade - 1."""

    fpr, tpr, limiares = roc_curve(y_real, probabilidades)
    validos = np.isfinite(limiares)
    if not validos.any():
        raise ValueError("Nao foi possivel calcular um limiar de Youden valido.")

    indice = np.argmax((tpr - fpr)[validos])
    return float(limiares[validos][indice])

def avaliar_modelo(
    dados_modelo: pd.DataFrame,
    features: list[str],
    n_splits: int = 4,
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

    Returns
    -------
    dict
        Métricas e previsões do modelo.
    """

    X = dados_modelo[features]

    y = dados_modelo[
        COLUNA_ALVO
    ]

    if dados_modelo.empty:
        raise ValueError(
            "Nenhuma observacao valida para modelagem. "
            "Verifique se as variaveis quimicas possuem "
            "valores numericos positivos."
        )

    contagem_classes = y.value_counts()

    if len(contagem_classes) < 2:
        raise ValueError(
            "A variavel alvo precisa conter as duas classes "
            "(0 e 1) para avaliar o modelo."
        )

    # Garante que não haja mais folds
    # que observações na menor classe
    menor_classe = int(
        contagem_classes.min()
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

    probabilidades = np.full(len(y), np.nan, dtype=float)
    particoes_folds = []

    for numero_fold, (indices_treino, indices_teste) in enumerate(
        cv.split(X, y),
        start=1,
    ):
        modelo_fold = criar_modelo_logistico()
        X_treino = X.iloc[indices_treino]
        X_teste = X.iloc[indices_teste]
        y_treino = y.iloc[indices_treino]
        y_teste = y.iloc[indices_teste]

        modelo_fold.fit(X_treino, y_treino)
        prob_fold = modelo_fold.predict_proba(X_teste)[:, 1]
        probabilidades[indices_teste] = prob_fold

        particoes_folds.append({
            "FOLD": numero_fold,
            "indices_treino": indices_treino,
            "indices_teste": indices_teste,
            "probabilidades": prob_fold,
        })

    limite = calcular_limiar_youden(y, probabilidades)
    metricas_folds = []

    for particao in particoes_folds:
        indices_treino = particao["indices_treino"]
        indices_teste = particao["indices_teste"]
        prob_fold = particao["probabilidades"]
        y_teste = y.iloc[indices_teste]
        prev_fold = (prob_fold >= limite).astype(int)

        tn_fold, fp_fold, fn_fold, tp_fold = confusion_matrix(
            y_teste,
            prev_fold,
            labels=[0, 1],
        ).ravel()

        metricas_folds.append({
            "FOLD": particao["FOLD"],
            "N_TREINO": len(indices_treino),
            "N_TESTE": len(indices_teste),
            "AUC": roc_auc_score(y_teste, prob_fold),
            "ACURACIA": accuracy_score(y_teste, prev_fold),
            "PRECISAO": precision_score(
                y_teste, prev_fold, zero_division=0
            ),
            "SENSIBILIDADE": recall_score(
                y_teste, prev_fold, zero_division=0
            ),
            "ESPECIFICIDADE": calcular_especificidade(
                y_teste, prev_fold
            ),
            "F1_SCORE": f1_score(
                y_teste, prev_fold, zero_division=0
            ),
            "VP": tp_fold,
            "FP": fp_fold,
            "VN": tn_fold,
            "FN": fn_fold,
        })

    previsoes = (
        probabilidades >= limite
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        previsoes,
        labels=[0, 1],
    ).ravel()

    tabela_folds = pd.DataFrame(metricas_folds)
    metricas_resumo = [
        "AUC",
        "ACURACIA",
        "PRECISAO",
        "SENSIBILIDADE",
        "ESPECIFICIDADE",
        "F1_SCORE",
    ]
    resumo_folds = {
        metrica: {
            "media": tabela_folds[metrica].mean(),
            "desvio_padrao": tabela_folds[metrica].std(ddof=1),
        }
        for metrica in metricas_resumo
    }

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
        "limite_youden": limite,
        "metricas_folds": tabela_folds,
        "resumo_folds": resumo_folds,
    }

    return resultado

def comparar_modelos(
    dados_modelo: pd.DataFrame,
    n_splits: int = 4,
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
        "Salinidade + Bário + Estrôncio": [
            "SALINIDADE",
            "BARIO",
            "ESTRONCIO",
        ],
        "Salinidade + Relação Ba/Sr": [
            "SALINIDADE",
            "RELACAO_BA_SR",
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
            "LIMIAR_YOUDEN": resultado["limite_youden"],
            **{
                f"{metrica}_MEDIA_FOLDS": resumo["media"]
                for metrica, resumo in resultado["resumo_folds"].items()
            },
            **{
                f"{metrica}_DP_FOLDS": resumo["desvio_padrao"]
                for metrica, resumo in resultado["resumo_folds"].items()
            },
            **{
                f"{metrica}_MEDIA_DP": (
                    f"{resumo['media']:.3f} ± "
                    f"{resumo['desvio_padrao']:.3f}"
                )
                for metrica, resumo in resultado["resumo_folds"].items()
            },
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
        "ESTRONCIO"
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
