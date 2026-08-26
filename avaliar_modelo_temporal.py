"""Avalia o modelo quimico em uma divisao cronologica 80/20."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.modelling import criar_modelo_logistico
from src.preparar_teste_temporal import criar_dataset_teste_temporal


BASE_DIR = Path(__file__).resolve().parent
PASTA_PROCESSED = BASE_DIR / "data" / "processed"
PASTA_OUTPUTS = BASE_DIR / "outputs"

FEATURES = ["SALINIDADE", "BARIO", "ESTRONCIO"]
ALVO = "TEM_NORM_MES"
LIMIAR = 0.5


def preparar_base_temporal() -> pd.DataFrame:
    fenix = pd.read_csv(PASTA_PROCESSED / "fenix_processado.csv")
    residuos = pd.read_csv(
        PASTA_PROCESSED / "base_integrada_residuos.csv"
    )
    dados = criar_dataset_teste_temporal(fenix, residuos)
    dados["Mes"] = pd.to_datetime(dados["Mes"], errors="coerce")

    for coluna in FEATURES:
        dados[coluna] = pd.to_numeric(dados[coluna], errors="coerce")

    validos = (
        dados["Mes"].notna()
        & dados[ALVO].notna()
        & dados[FEATURES].notna().all(axis=1)
        & (dados[FEATURES] > 0).all(axis=1)
    )
    return dados.loc[validos].sort_values("Mes").reset_index(drop=True)


def encontrar_data_corte(dados: pd.DataFrame, proporcao: float = 0.8):
    """Encontra o fim de mes que mais se aproxima da proporcao desejada."""

    acumulado = dados.groupby("Mes").size().sort_index().cumsum()
    alvo = proporcao * len(dados)
    return (acumulado - alvo).abs().idxmin()


def calcular_metricas(y_real, probabilidades: np.ndarray) -> dict:
    previstos = (probabilidades >= LIMIAR).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_real, previstos, labels=[0, 1]
    ).ravel()
    especificidade = tn / (tn + fp) if (tn + fp) else np.nan

    return {
        "AUC_ROC": roc_auc_score(y_real, probabilidades),
        "AUC_PR": average_precision_score(y_real, probabilidades),
        "AUC_PR_REFERENCIA": float(np.mean(y_real)),
        "ACURACIA": accuracy_score(y_real, previstos),
        "ACURACIA_BALANCEADA": balanced_accuracy_score(y_real, previstos),
        "PRECISAO": precision_score(y_real, previstos, zero_division=0),
        "SENSIBILIDADE": recall_score(y_real, previstos, zero_division=0),
        "ESPECIFICIDADE": especificidade,
        "F1_SCORE": f1_score(y_real, previstos, zero_division=0),
        "VN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "VP": int(tp),
    }


def main() -> None:
    dados = preparar_base_temporal()
    data_corte = encontrar_data_corte(dados)
    treino = dados[dados["Mes"] <= data_corte].copy()
    teste = dados[dados["Mes"] > data_corte].copy()

    modelo = criar_modelo_logistico()
    modelo.fit(treino[FEATURES], treino[ALVO].astype(int))
    probabilidades = modelo.predict_proba(teste[FEATURES])[:, 1]
    metricas = calcular_metricas(teste[ALVO].astype(int), probabilidades)

    previsoes = teste[
        ["LOCAL DA GERAÇÃO", "Mes", ALVO, *FEATURES]
    ].copy()
    previsoes["PROB_NORM"] = probabilidades
    previsoes["PREVISAO_NORM"] = (probabilidades >= LIMIAR).astype(int)

    resumo = pd.DataFrame(
        [
            {
                "DATA_CORTE": data_corte.date().isoformat(),
                "N_TREINO": len(treino),
                "N_TESTE": len(teste),
                "PERCENTUAL_TREINO": len(treino) / len(dados),
                "POSITIVOS_TREINO": int(treino[ALVO].sum()),
                "POSITIVOS_TESTE": int(teste[ALVO].sum()),
                "LIMIAR": LIMIAR,
                **metricas,
            }
        ]
    )

    PASTA_OUTPUTS.mkdir(parents=True, exist_ok=True)
    pasta_modelos = PASTA_OUTPUTS / "models"
    pasta_modelos.mkdir(parents=True, exist_ok=True)
    resumo.to_csv(PASTA_OUTPUTS / "avaliacao_temporal.csv", index=False)
    previsoes.to_csv(
        PASTA_OUTPUTS / "previsoes_teste_temporal.csv", index=False
    )
    joblib.dump(modelo, pasta_modelos / "modelo_norm_temporal_80.joblib")

    print("Features:", ", ".join(FEATURES))
    print(resumo.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
