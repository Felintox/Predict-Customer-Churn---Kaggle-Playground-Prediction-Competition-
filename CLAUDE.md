# Predict Customer Churn - Kaggle Playground

## Objetivo
Maximizar performance preditiva de churn de clientes de telecomunicações.
**Foco: performance acima de tudo.**

## Métrica Principal
- **ROC-AUC** (sempre usar `scoring="roc_auc"`)
- Validação obrigatória: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- Nunca reportar resultado sem cross-validation

## Dados
- `data/train.csv` — 594.194 linhas, 20 colunas
- `data/test.csv` — dados de teste para submissão
- `data/sample_submission.csv` — formato esperado
- **Target:** coluna `Churn` (Yes/No)
- **Desbalanceamento:** ~77.5% No / ~22.5% Yes → sempre usar `stratify`

### Variáveis Numéricas
`SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`

### Variáveis Categóricas
`gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`,
`InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`,
`TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`,
`PaperlessBilling`, `PaymentMethod`

> Atenção: categorias como `OnlineSecurity` têm valor "No internet service" — tratar adequadamente no encoding.

## Algoritmos a Testar (os 5 principais para classificação)
1. `LogisticRegression` — baseline linear
2. `RandomForestClassifier` — ensemble de árvores
3. `XGBClassifier` — gradient boosting (XGBoost)
4. `LGBMClassifier` — gradient boosting (LightGBM)
5. `CatBoostClassifier` — gradient boosting nativo para categorias

## Feature Engineering
Sempre testar múltiplas estratégias e comparar via cross-validation:

### Seleção de Features (feature_engine.selection)
- `DropConstantFeatures`
- `DropCorrelatedFeatures`
- `SmartCorrelatedSelection`

### Criação de Features (explorar)
- Interações entre variáveis (ex: `tenure * MonthlyCharges`)
- Agregações (ex: razão `MonthlyCharges / TotalCharges`)
- Binning de variáveis numéricas
- `feature_engine` para transformações automatizadas
- `featuretools` para deep feature synthesis se necessário

## Pipeline Padrão
```python
ColumnTransformer(
    num → StandardScaler(),
    cat → BinaryEncoder()  # category_encoders
)
```
Testar encoders alternativos: `OrdinalEncoder`, `TargetEncoder`, `OneHotEncoder`.

## Bibliotecas do Projeto
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from category_encoders import BinaryEncoder
from feature_engine.selection import DropConstantFeatures, DropCorrelatedFeatures, SmartCorrelatedSelection
```

## Regras Obrigatórias
- **Nunca** fazer fit de qualquer transformação com dados de teste (data leakage)
- **Sempre** usar `StratifiedKFold` dado o desbalanceamento
- **Sempre** comparar estratégias em tabela ordenada por `ROC-AUC Mean`
- Usar `n_jobs=-1` para paralelizar cross-validation
- `random_state=42` em todos os modelos e splits

## Notebook Principal
`churn-predict.ipynb` — todo o desenvolvimento ocorre aqui

## Próximos Passos (pipeline planejado)
1. [ ] Feature engineering avançado
2. [ ] Testar CatBoostClassifier
3. [ ] Tuning dos melhores modelos (Optuna ou GridSearch)
4. [ ] Ensemble / Stacking
5. [ ] Geração do arquivo de submissão
