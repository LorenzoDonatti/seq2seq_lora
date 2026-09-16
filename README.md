# LoRaWAN Multi-Node RSSI Forecasting Benchmark

Este repositório implementa um benchmark temporal e reprodutível para previsão de múltiplos nós em redes LoRaWAN (baseado no dataset de vinhedo `vineyard-2021`).

O objetivo central é investigar cientificamente o **trade-off entre modelos dedicados uninó (*Single-Node*) vs. modelos integrados multinó (*Multi-Node / Joint*)**.

---

## 📁 Estrutura do Projeto

```
seq2seq/
├── archive/
│   └── SEQ2SEQLoRa.ipynb        # Versão antiga/protótipo uninó arquivada
├── data/
│   └── combined_hourly_data.csv # Dataset de RSSI e variáveis meteorológicas
├── src/
│   ├── __init__.py               # Pacote principal
│   ├── cli/
│   │   └── run_benchmark.py      # Único ponto de entrada do projeto
│   ├── data_loader.py           # Split temporal, janelas completas/horárias, sem imputação
│   ├── metrics.py               # MAE (dBm), RMSE (dBm), latência de inferência e contagem de parâmetros
│   ├── plotting.py              # Gráficos comparativos (curvas temporais e barras de métricas)
│   ├── evaluation.py            # Orquestrador do benchmark comparativo
│   ├── hyperparameter_search.py # Busca neural reutilizável
│   ├── statistical_search.py    # Busca AR/ARIMA reutilizável
│   ├── node_topology.py         # Coordenadas e grafo físico dos nós
│   └── models/
│       ├── __init__.py
│       ├── baselines.py         # Persistência, AR e ARIMA
│       ├── single_node_models.py# Ensemble de 8 LSTMs independentes (1 por nó)
│       ├── multi_node_seq2seq.py# Seq2Seq Multi-Node com Atenção Temporal (direto sem teacher forcing)
│       ├── dlinear.py           # NLinear (AAAI 2023)
│       └── stgnn.py             # Grafo físico GPS + adjacência adaptativa aprendida
├── tests/                       # Testes metodológicos e de integridade
├── benchmark_results/           # Gráficos gerados (.png) e relatórios (.json)
├── seq2seqLoRa.ipynb            # Notebook interativo atualizado e corrigido
├── pyproject.toml               # Pacote, comandos instaláveis e configuração do pytest
├── requirements.txt             # Dependências
└── README.md
```

---

## 🎯 Horizontes de Previsão Definidos

Em redes LoRaWAN, diferentes horizontes de tempo atendem a requisitos operacionais específicos:

1. **Curto Prazo ($H = 1$ hora):**
   - **Caso de uso:** Decisões imediatas de *Adaptive Data Rate* (ADR), ajuste dinâmico de *Spreading Factor* (SF) e potência de transmissão ($TX_{power}$) para o próximo pacote.
2. **Médio Prazo ($H = 6$ horas):**
   - **Caso de uso:** Compensação de transições ambientais (entardecer/amanhecer, variações de umidade e formação de orvalho sobre folhagens do vinhedo).
3. **Longo Prazo / Diário ($H = 24$ horas):**
   - **Caso de uso:** Planejamento de janelas de recarga por energia solar/harvesting, predição de desvanecimento sazonal e prevenção de perda de pacotes.

---

## 🚀 Como executar

Existe somente um comando: `lora-benchmark`. A otimização de modelos é um parâmetro
desse mesmo comando, não uma etapa ou programa separado. Os vencedores da última
otimização ficam salvos por horizonte e são reutilizados automaticamente nas execuções
seguintes.

### 1. Preparar o ambiente

```bash
source .venv/bin/activate
pip install -e .
```

A instalação editável disponibiliza o comando `lora-benchmark`. Sem instalar o pacote,
o equivalente é `python -m src.cli.run_benchmark`.

Se a máquina estiver sem acesso à internet, mas as dependências já estiverem instaladas
na `.venv`, use o ambiente local também para construir o comando:

```bash
pip install -e . --no-deps --no-build-isolation
```

Se o shell ainda não localizar o comando depois da instalação, reative o ambiente com
`source .venv/bin/activate` ou execute diretamente `.venv/bin/lora-benchmark`.

### 2. Uso mais simples

```bash
lora-benchmark --horizon 1
```

Isso executa o benchmark H=1 com 128 épocas e reutiliza a última configuração otimizada
salva para H=1. Para selecionar novamente lags, ordem ARIMA, arquiteturas e batch sizes
somente na validação antes do teste final, basta acrescentar `--optimize`:

```bash
lora-benchmark --horizon 1 --optimize
```

Esse é o comando recomendado para o experimento final. A otimização e o benchmark são
executados na mesma chamada e todos os relatórios ficam no mesmo diretório.

### Exemplos

Executar todos os horizontes **sem refazer a otimização**, reutilizando automaticamente
os últimos vencedores salvos, com 128 épocas, seed fixa e um diretório de saída próprio:

```bash
lora-benchmark \
  --horizons 1 6 12 24 \
  --epochs 128 \
  --seed 42 \
  --output-dir benchmark_results/sem_otimizacao_seed42
```

Nesse modo, `--trials` e `--search-epochs` não são necessários. As configurações ficam
em `.lora_benchmark/last_optimized_configs.json`, separadas por horizonte. Se não houver
configuração compatível com o dataset e o histórico solicitados, o comando para com uma
mensagem clara em vez de trocar silenciosamente para os padrões.

Para ignorar deliberadamente a última otimização e usar os parâmetros internos padrão:

```bash
lora-benchmark \
  --horizons 1 6 12 24 \
  --use-defaults \
  --seed 42 \
  --output-dir benchmark_results/defaults_seed42
```

Alterar o horizonte e o número de épocas do treinamento final com otimização:

```bash
lora-benchmark --horizon 6 --epochs 30 --optimize
```

Controlar o custo da otimização — neste exemplo, oito configurações por modelo e cinco
épocas por configuração:

```bash
lora-benchmark --horizon 1 --optimize --trials 8 --search-epochs 5
```

Escolher o diretório de saída:

```bash
lora-benchmark --horizon 1 --optimize \
  --seed 42 \
  --output-dir benchmark_results/revised_H1_seed42
```

Executar vários horizontes com as últimas configurações otimizadas:

```bash
lora-benchmark --horizons 1 6 12 24
```

Otimizar e executar vários horizontes em uma única chamada:

```bash
lora-benchmark \
  --horizons 1 6 12 24 \
  --optimize \
  --trials 12 \
  --search-epochs 10 \
  --epochs 128 \
  --seed 42 \
  --output-dir benchmark_results/com_otimizacao_seed42
```

Nesse modo, cada horizonte recebe uma busca independente, pois possui seu próprio
objetivo de validação. Os relatórios são salvos como `statistical_search_H*.json` e
`neural_search_H*.json`, e cada benchmark usa somente a configuração selecionada para
seu respectivo horizonte. Durante a otimização são avaliados AR, ARIMA, VAR, VARX,
Seq2Seq, STGNN, LSTM uninó e NLinear. O teste permanece isolado até a avaliação final.

Principais parâmetros:

| Parâmetro | Padrão | Função |
| :--- | :---: | :--- |
| `--horizon` | `1` | Horizonte único: 1, 6, 12 ou 24 horas. |
| `--epochs` | `128` | Épocas do treinamento final. |
| `--optimize` | desligado | Ativa seleção de configurações na validação. |
| `--use-defaults` | desligado | Ignora explicitamente a última otimização e usa os padrões internos. |
| `--trials` | `12` | Configurações testadas por modelo neural. |
| `--search-epochs` | `10` | Épocas de cada configuração durante a otimização. |
| `--history` | `24` | Horas de histórico usadas como entrada. |
| `--seed` | `42` | Seed reprodutível. |
| `--output-dir` | `benchmark_results` | Diretório dos resultados. |

---

## 🔬 Modelos Avaliados

| Categoria | Modelo | Descrição |
| :--- | :--- | :--- |
| **Baseline Naive** | `Persistence` | Repete o último valor observado ($X_{t-1}$). Linha de base essencial para séries temporais. |
| **Baseline Estatístico** | `AR_Baseline` | Modelo AR ajustado por nó; o número de lags é selecionado somente na validação. |
| **Estatístico conjunto** | `Joint_VAR` | Um único VAR prevê os oito nós e aprende dependências cruzadas entre seus lags. |
| **Estatístico conjunto** | `Joint_VARX` | Regressão multivariada direta com lags de todos os RSSIs e do clima histórico. Não usa clima futuro. |
| **Uninó (Dedicado)** | `SingleNode_LSTM` | **8 modelos independentes**, um para cada nó LoRa, treinados apenas com seu histórico + clima. |
| **Multinó Integrado** | `MultiNode_Seq2Seq` | Encoder LSTM + Atenção Temporal + Link residual do último passo. |
| **Linear moderno** | `NLinear` | Projeção linear após subtrair a última observação (AAAI 2023). |
| **Espaço-temporal** | `PhysicalAdaptive_STGNN` | Grafo físico por coordenadas GPS combinado com adjacência aprendida e distância ao gateway. |

## Protocolo de dados e métricas

- Divisão cronológica: 70% treino, 15% validação e 15% teste.
- Os scalers são ajustados exclusivamente em valores observados no treino.
- Não há imputação: uma janela é aceita somente se todas as entradas e todos os alvos forem observados.
- Uma janela é aceita somente se todos os timestamps consecutivos estiverem exatamente a uma hora de distância; lacunas de aquisição nunca são atravessadas.
- `global` é a média sobre origens, nós e todos os passos do horizonte.
- `terminal_horizon` mede especificamente o passo final (por exemplo, `t+24`).
- `per_lead_time` reporta separadamente cada passo futuro.
- `per_node` é a visão principal: MAE e RMSE são reportados separadamente para cada nó.
- `metrics_per_node_H*.csv/png` contém a tabela e o mapa de calor por nó.
- `predictions_H*.npz` preserva previsões por origem, passo e nó para análises posteriores.
- Para medir o custo de atender todos os nós, o relatório inclui número de instâncias de
  modelo, parâmetros totais, memória numérica estimada dos pesos e tempo total de treino.
  Modelos integrados usam uma instância; AR, ARIMA e LSTM uninó usam oito.
- VAR e VARX são baselines integradas: produzem todos os nós com uma única matriz de
  coeficientes. A otimização seleciona lags e regularização Ridge somente na validação.
- O seed padrão é 42 e pode ser alterado com `--seed`.
- Cada `--optimize` atualiza `.lora_benchmark/last_optimized_configs.json`; sem essa opção,
  o benchmark reutiliza esses parâmetros. O protocolo registra `hyperparameter_source`
  como `fresh_optimization`, `last_saved_optimization` ou `built_in_defaults`.

## Desenvolvimento e validação

```bash
python -m pytest
python -m compileall -q src tests
```

Os módulos em `src/cli` contêm somente parsing e apresentação. A lógica deve permanecer
nos módulos diretamente sob `src`, permitindo uso tanto pela CLI quanto por notebooks e testes.
