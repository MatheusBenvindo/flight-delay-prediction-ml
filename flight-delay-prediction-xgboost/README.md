# 🛫 Modelo Preditivo de Atrasos de Voos — XGBoost

> **UC03 — Machine Learning | SENAI**  
> Modelo de classificação binária que prevê se um voo sofrerá atraso, desenvolvido com pipeline completo de pré-processamento, treinamento, serialização e inferência em produção.

<br>

## 🎯 Sobre o Projeto

Este projeto implementa um **pipeline completo de Machine Learning** para previsão de atrasos de voos. O fluxo cobre desde o pré-processamento dos dados até a implantação do modelo serializado, com suporte a inferência em tempo real e em lote.

O foco do projeto não é apenas na acurácia — mas em demonstrar um ciclo de desenvolvimento **robusto e pronto para produção**: codificação consistente de variáveis categóricas, serialização de objetos de transformação, benchmarking de inferência e rastreabilidade completa dos metadados do modelo.

<br>

## ⚙️ Por que XGBoost?

O **XGBoost (eXtreme Gradient Boosting)** foi escolhido por razões técnicas sólidas:

| Critério | Justificativa |
|---|---|
| **Gradient Boosting** | Constrói árvores sequencialmente corrigindo os erros anteriores — alta acurácia em dados tabulares |
| **Regularização L1/L2** | Penaliza complexidade excessiva, fundamental para datasets pequenos como este de 120 registros |
| **Recursos mistos** | Trabalha nativamente com features numéricas e categóricas (após encoding) sem necessidade de normalização |
| **Velocidade** | Construção paralela de árvores com `n_jobs=-1` — treinamento concluído em < 0,17 segundos |
| **Padrão de mercado** | Top performer consistente em competições de dados estruturados (Kaggle, benchmarks acadêmicos) |

**Hiperparâmetros utilizados:**
```python
XGBClassifier(
    objective        = 'binary:logistic',  # saída probabilística binária
    eval_metric      = 'logloss',
    max_depth        = 6,
    learning_rate    = 0.1,
    n_estimators     = 100,
    subsample        = 0.8,                # subamostragem para evitar overfitting
    colsample_bytree = 0.8,
    random_state     = 42,
    n_jobs           = -1                  # paralelismo total
)
```

<br>

## 🔤 Engenharia de Features — Label Encoding

As variáveis categóricas foram convertidas para formato numérico com **`sklearn.preprocessing.LabelEncoder`**. Os encoders foram **serializados junto ao modelo** (`models/label_encoders.pkl`), garantindo que a codificação na inferência seja idêntica à do treinamento — sem vazamento de dados.

| Feature | Tipo | Exemplo |
|---|---|---|
| `airline` | Categórica → Codificada | `LATAM`, `GOL`, `Azul` → `0, 1, 2` |
| `origin` | Categórica → Codificada | `GRU`, `CGH`, `SDU` → `0, 1, 2` |
| `destination` | Categórica → Codificada | `CWB`, `BSB`, `SSA` → `0, 1, 2` |
| `weather` | Categórica → Codificada | `Clear`, `Rain`, `Fog` → `0, 1, 2` |
| `departure_hour` | Numérica | `6`, `14`, `21` |
| `day_of_week` | Numérica | `0` (Seg) … `6` (Dom) |

**Variável alvo:** `delayed` → `0` (no prazo) / `1` (atrasado)

<br>

## 📊 Métricas de Performance

Avaliadas no conjunto de teste com **24 registros** (divisão 80/20 de 120 amostras totais).

| Métrica | Resultado |
|---|---|
| ✅ **Acurácia** | **91,67%** |
| 🎯 **Precisão** | **93,06%** |
| 🔁 **Recall** | **91,67%** |
| ⚖️ **F1-Score** | **91,72%** |
| 📈 **ROC-AUC** | **97,14%** |

### Desempenho de Inferência

| Modo | Latência Média | Throughput |
|---|---|---|
| Tempo real (5 registros) | 5,40 ms/registro | 185 pred/s |
| Lote (30 registros) | 0,27 ms/registro | 3.748 pred/s |

> O modelo em lote é **20× mais rápido** que a inferência individual, mantendo correlação de resultados de **99,99%**.

<br>

## 🗂️ Estrutura do Projeto

```
flight-delay-prediction-xgboost/
│
├── 📓 Atividade 4.ipynb              # Notebook principal — pipeline completo
│
├── 📁 data/
│   ├── flights_delays_120.csv        # Dataset fonte (120 registros)
│   ├── train_data.csv                # Split de treino (96 registros — 80%)
│   └── test_data.csv                 # Split de teste (24 registros — 20%)
│
├── 📁 models/
│   ├── xgboost_flight_delay_model.pkl   # Modelo XGBoost serializado
│   ├── label_encoders.pkl               # Encoders serializados (consistência de inferência)
│   └── model_metadata.json             # Snapshot de hiperparâmetros e métricas
│
└── 📁 results/
    ├── real_time_predictions.csv        # Saída de inferência em tempo real
    ├── batch_predictions_results.csv    # Saída de inferência em lote
    ├── comparison_analysis.json         # Comparativo de latência e throughput
    └── final_activity_report.json       # Relatório completo de performance
```

<br>

## 🛠️ Stack Tecnológica

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-189AB4?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?style=flat-square&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)

<br>

## 🌐 Portfólio Principal

Este projeto faz parte da minha jornada em Inteligência Artificial e Machine Learning.

👉 **[Acesse meu Portfólio → matheusbenvindo.github.io/MBR-Portfolio](https://matheusbenvindo.github.io/MBR-Portfolio/)**

---

<p align="center">
  <sub>Desenvolvido por <strong>Matheus Benvindo</strong> · SENAI — Unidade Curricular 03: Machine Learning</sub>
</p>
