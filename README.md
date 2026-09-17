# LoRaWAN: previsão por nó, compartilhada e conjunta

Estudo do compromisso entre precisão por nó e recursos para prever RSSI multinó.
O experimento 1 usa os oito nós do `vineyard-2021`; o experimento 2 é uma validação
externa com nove sensores do deployment urbano UVA recebidos pelo gateway A.

## Modelos e informação disponível

| Modelo | Estrutura | Entradas |
|---|---|---|
| ARX/ARIMAX | Um preditor por nó | RSSI próprio + clima histórico |
| VARX/VARIMAX | Uma regressão conjunta para todos os nós | RSSIs e clima históricos |
| SingleNode_Seq2Seq | Um encoder–decoder independente por nó | RSSI próprio + clima |
| MultiNode_Seq2Seq | Encoder LSTM, decoder LSTM recorrente e atenção aditiva | Todos os RSSIs + clima histórico |
| NLinear | Projeção temporal compartilhada, normalização pela última observação | RSSI próprio |
| DLinear | Decomposição por média móvel de 25 passos, duas projeções compartilhadas | RSSI próprio |
| PhysicalAdaptive_STGNN | Convoluções temporais e grafo físico/adaptativo | RSSIs + clima + geometria |

AR, persistência e VAR sem exógenas foram removidos por decisão do pesquisador. TCN
continua removido. Os modelos estatísticos usam CPU; os neurais podem usar CUDA.

NLinear e DLinear seguem as formulações originais com `individual=False`.
Eles compartilham parâmetros, mas **não misturam canais** e não usam clima.
Isso é diferente de um modelo conjunto que consome os outros nós na inferência.
A antiga projeção densa multinó chamada NLinear foi substituída.
Referência: [implementações dos autores](https://github.com/cure-lab/LTSF-Linear/tree/main/models).

O Seq2Seq mantém seu nome, agora com encoder–decoder recorrente: o estado final
do encoder inicializa o decoder; cada passo calcula atenção sobre o histórico,
consome a previsão anterior e produz os oito RSSIs seguintes. O primeiro passo
recebe o último RSSI observado. A cabeça prevê um residual relativo à última
observação. Treino e inferência usam o mesmo rollout, sem teacher forcing.
Não é uma reprodução literal de um modelo de tradução, mas uma adaptação
de [encoder–decoder com atenção aditiva](https://arxiv.org/abs/1409.0473).

## Execução

Preparar uma única vez o dataset UVA em formato horário causal:

```bash
lora-prepare-uva \
  --source-dir dataset \
  --output-file data/experiment_2_uva_gatewayA_hourly.csv
```

O comando preserva ausências, gera o sidecar de topologia e registra todas as decisões
em `data/experiment_2_uva_gatewayA_hourly.standardization.json`. A metodologia e a
auditoria estão em `dataset/FORECASTING_VALIDATION.md`.

Mantenha obrigatoriamente as saídas separadas:

```text
benchmark_results/experiment_1_vineyard/
benchmark_results/experiment_2_uva_gatewayA/
```

Experimento 1 (vinhedo):

```bash
source .venv/bin/activate
pip install -e . --no-deps --no-build-isolation
lora-benchmark --horizon 1 --optimize \
  --trials 12 --epochs 64 --search-epochs 64 --patience 10 \
  --history 24 --device auto --graph-ablations \
  --data-file data/combined_hourly_data.csv \
  --output-dir benchmark_results/experiment_1_vineyard/confirmatory_seed42
```

Experimento 2 (UVA, mesma política):

```bash
lora-benchmark --horizon 1 --optimize \
  --trials 12 --epochs 64 --search-epochs 64 --patience 10 \
  --history 24 --device auto --graph-ablations \
  --data-file data/experiment_2_uva_gatewayA_hourly.csv \
  --output-dir benchmark_results/experiment_2_uva_gatewayA/confirmatory_seed42
```

`--device auto` (padrão) seleciona CUDA quando disponível e CPU caso contrário; a opção
pode ser omitida. `--device cuda` força GPU e falha explicitamente se ela não estiver
disponível. Não há fallback silencioso após erros de treinamento. `--search-epochs`
assume o mesmo limite de `--epochs`
quando omitido; ambos usam early stopping na validação com `--patience 10`.

Sem `--optimize`, reutiliza-se o cache da última seleção por horizonte.
`--use-defaults` é a única forma explícita de usar configurações internas.
O protocolo v4 rejeita caches antigos e mantém vencedores separados por dataset e
histórico. Use um diretório novo por experimento; resultados existentes são protegidos.

```bash
lora-benchmark --horizon 1 --epochs 64 --device auto \
  --data-file data/combined_hourly_data.csv \
  --output-dir benchmark_results/experiment_1_vineyard/repeat_seed42
```

O comando equivalente sem instalar o pacote é `python -m src.cli.run_benchmark`.

## Protocolo metodológico

- Divisão cronológica 70/15/15, scalers ajustados apenas no treino.
- Sem imputação; entradas e alvos devem ser observados e horários consecutivos.
  Clima futuro não é entrada nem requisito de elegibilidade.
- Todos os modelos são comparados nas mesmas origens completas.
- ARIMA maximiza a soma de verossimilhanças de segmentos horários independentes,
  com parâmetros comuns por nó. Segmentos curtos excluídos são contabilizados.
  A inferência usa filtragem exata do statsmodels sobre cada histórico observado.
- Modelos neurais treinam e selecionam checkpoints por MAE na escala física,
  ponderando erros normalizados pela amplitude de treino de cada nó.
- Hiperparâmetros e checkpoints usam apenas validação; busca e treino final têm,
  por padrão, o mesmo limite de épocas e a mesma paciência.
- Métricas por nó são primárias. Média global, erro terminal e por lead são
  complementares. RSSI está em dBm; erros em dB. Chaves JSON `mae_dbm/rmse_dbm`
  foram mantidas por compatibilidade; as tabelas usam a unidade correta.
- Latências são medianas de 30 chamadas após 5 aquecimentos, lotes de 1 e 32
  origens, todos os nós. CUDA é sincronizada e transferências são incluídas.
  Compare também o dispositivo: métodos estatísticos permanecem na CPU.
- Armazenamento de parâmetros é estimado; não inclui runtime, buffers ou ativações.
  Tempo de fit não inclui busca; relatórios de busca registram os tempos dos trials.
- Protocolos preservam hardware, versões, seed, hash do dataset e dos fontes.
  NPZs incluem timestamps da primeira previsão de cada origem, verdade e previsões.

## Interpretação e limites

Resultados anteriores a este protocolo são históricos e não representam as
arquiteturas atuais. O run v3 do vinhedo foi preservado em
`benchmark_results/experiment_1_vineyard/v3_h1_gpu_seed42`.

O controle compartilhado ajuda a estudar compartilhamento sem informação cruzada.
A comparação com Seq2Seq/STGNN ainda muda a arquitetura: não atribuir causalmente
toda diferença apenas ao compartilhamento. Múltiplas sementes, blocos temporais,
ablação do grafo e outra implantação continuam necessários para uma conclusão forte.

Previsão da média horária não equivale à previsão do próximo pacote. ADR, energia,
PDR e prevenção de perdas são aplicações possíveis, ainda não validadas aqui.
Remover baselines simples limita o alcance de alegações de superioridade.

## Validação

```bash
python -m pytest
python -m compileall -q src tests
```

O CLI contém parsing/apresentação. Preparação, busca, treinamento e avaliação estão
em módulos separados em `src`. Consulte `PROJECT_CONTEXT.md` para o handoff.

## Protocolo v4 e famílias pareadas

O benchmark atual compara ARX, ARIMAX, VARX e VARIMAX; Seq2Seq dedicado e integrado;
NLinear e DLinear dedicados e integrados; e o PhysicalAdaptive_STGNN multinó. AR,
persistência, VAR sem exógenas e TCN foram removidos. As versões estatísticas usam
temperatura, umidade, pressão e chuva somente até a origem: para horizonte H, o
regressor meteorológico é defasado H horas. ARIMAX e VARIMAX acrescentam uma inovação
MA(1) diagonal à mesma regressão causal. Nenhum modelo consulta clima futuro.

O histórico padrão é 24 horas. O benchmark confirmatório usa somente previsão
one-step-ahead (H=1), mantendo horizontes maiores apenas para análises exploratórias.
Os pares
dedicados/integrados usam a mesma família e objetivo; a diferença é
o número de cópias e, nos modelos integrados, o acesso aos históricos dos demais nós.
NLinear e DLinear seguem as formulações originais sem mistura entre canais.

O grafo híbrido combina uma prior de distância geográfica com uma matriz adaptativa
aprendida, sem autoarestas. A convolução temporal é causal e dilatada. Use
`--graph-ablations` para refitar controles sem arestas, apenas físico e apenas adaptativo.
Essas matrizes são pesos preditivos estáticos, não conectividade física nem evidência causal.

Execução completa recomendada:

```bash
lora-benchmark --horizon 1 --optimize --trials 12 \
  --epochs 64 --search-epochs 64 --patience 10 --history 24 \
  --device auto --graph-ablations \
  --output-dir benchmark_results/experiment_1_vineyard/confirmatory_seed42
```
