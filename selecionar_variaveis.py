"""Compara subconjuntos pequenos de variaveis para classificar NORM."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
ARQUIVO_ENTRADA = BASE_DIR / "data" / "processed" / "base_analise.csv"
ARQUIVO_SAIDA = BASE_DIR / "outputs" / "selecao_variaveis.csv"

ALVO = "TEM_NORM"
CANDIDATAS = {
    "SALINIDADE": "MEDIANA_SALINIDADE_PLAT",
    "BARIO": "MEDIANA_BARIO_PLAT",
    "ESTRONCIO": "MEDIANA_ESTRONCIO_PLAT",
}


def criar_modelo() -> Pipeline:
    return Pipeline(
        steps=[
            ("log10", FunctionTransformer(np.log10)),
            ("padronizacao", StandardScaler()),
            (
                "classificador",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=42,
                ),
            ),
        ]
    )


def avaliar_combinacoes(
    dados: pd.DataFrame,
    repeticoes: int = 50,
    folds: int = 4,
) -> pd.DataFrame:
    """Avalia cada conjunto em repeticoes completas da validacao cruzada."""

    y = pd.to_numeric(dados[ALVO], errors="raise").astype(int)
    menor_classe = int(y.value_counts().min())
    folds = min(folds, menor_classe)
    if folds < 2 or y.nunique() != 2:
        raise ValueError("Sao necessarias duas classes e duas amostras por classe.")

    resultados = []
    nomes = tuple(CANDIDATAS)
    for tamanho in range(1, 4):
        for features in combinations(nomes, tamanho):
            colunas = [CANDIDATAS[nome] for nome in features]
            X = dados[colunas].apply(pd.to_numeric, errors="coerce")
            validos = X.notna().all(axis=1) & (X > 0).all(axis=1)
            X_avaliacao = X.loc[validos]
            y_avaliacao = y.loc[validos]

            aucs_roc = []
            aucs_pr = []
            for repeticao in range(repeticoes):
                cv = RepeatedStratifiedKFold(
                    n_splits=folds,
                    n_repeats=1,
                    random_state=42 + repeticao,
                )
                probabilidades = cross_val_predict(
                    criar_modelo(),
                    X_avaliacao,
                    y_avaliacao,
                    cv=cv,
                    method="predict_proba",
                )[:, 1]
                aucs_roc.append(roc_auc_score(y_avaliacao, probabilidades))
                aucs_pr.append(average_precision_score(y_avaliacao, probabilidades))

            resultados.append(
                {
                    "VARIAVEIS": " + ".join(features),
                    "N_VARIAVEIS": tamanho,
                    "N_OBSERVACOES": int(validos.sum()),
                    "AUC_ROC_MEDIA": np.mean(aucs_roc),
                    "AUC_ROC_DP": np.std(aucs_roc, ddof=1),
                    "AUC_PR_MEDIA": np.mean(aucs_pr),
                    "AUC_PR_DP": np.std(aucs_pr, ddof=1),
                }
            )

    return (
        pd.DataFrame(resultados)
        .sort_values(
            ["AUC_ROC_MEDIA", "AUC_PR_MEDIA", "N_VARIAVEIS"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )


def main() -> None:
    dados = pd.read_csv(ARQUIVO_ENTRADA)
    resultado = avaliar_combinacoes(dados)

    ARQUIVO_SAIDA.parent.mkdir(parents=True, exist_ok=True)
    resultado.to_csv(ARQUIVO_SAIDA, index=False)

    colunas = [
        "VARIAVEIS",
        "N_VARIAVEIS",
        "N_OBSERVACOES",
        "AUC_ROC_MEDIA",
        "AUC_ROC_DP",
        "AUC_PR_MEDIA",
        "AUC_PR_DP",
    ]
    print(resultado[colunas].head(15).round(3).to_string(index=False))
    print(f"\nResultado completo salvo em: {ARQUIVO_SAIDA}")


if __name__ == "__main__":
    main()
