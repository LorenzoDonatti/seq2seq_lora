# LoRaWAN: previsão por nó, compartilhada e conjunta

Estudo do compromisso entre precisão por nó e recursos para prever RSSI multinó.
O experimento 1 usa os oito nós do `vineyard-2021`; o experimento 2 é uma validação
externa com nove sensores do deployment urbano UVA recebidos pelo gateway A.

## Modelos e informação disponível

| Modelo | Estrutura | Entradas |
|---|---|---|
| ARIMAX Box–Jenkins | Um preditor por nó, ordem própria | RSSI próprio + clima histórico |
| VARX | Uma regressão conjunta para todos os nós | RSSIs e clima históricos |
| SingleNode_Seq2Seq | Um encoder–decoder independente por nó | RSSI próprio + clima |
| MultiNode_Seq2Seq | Encoder LSTM, decoder LSTM recorrente e atenção aditiva | Todos os RSSIs + clima histórico |
| NLinear | Projeção temporal compartilhada, normalização pela última observação | RSSI próprio |
| DLinear | Decomposição por média móvel de 25 passos, duas projeções compartilhadas | RSSI próprio |
| PhysicalAdaptive_STGNN | Convoluções temporais e grafo físico/adaptativo | RSSIs + clima + geometria |

Os modelos estatísticos usam CPU; os neurais podem usar CUDA.

NLinear e DLinear seguem as formulações originais com `individual=False`.
Eles compartilham parâmetros, mas **não misturam canais** e não usam clima.
Isso é diferente de um modelo conjunto que consome os outros nós na inferência.
Referência: [implementações dos autores](https://github.com/cure-lab/LTSF-Linear/tree/main/models).

O Seq2Seq mantém seu nome, agora com encoder–decoder recorrente: o estado final
do encoder inicializa o decoder; cada passo calcula atenção sobre o histórico,
consome a previsão anterior e produz os RSSIs de todos os nós. O primeiro passo
recebe o último RSSI observado. A cabeça prevê um residual relativo à última
observação. Treino e inferência usam o mesmo rollout, sem teacher forcing.
Não é uma reprodução literal de um modelo de tradução, mas uma adaptação
de [encoder–decoder com atenção aditiva](https://arxiv.org/abs/1409.0473).

Uma explicação visual e acessível da rede de grafos, das quatro ablações e do resultado
exploratório UVA está em [`docs/STGNN_EXPLICADA.md`](docs/STGNN_EXPLICADA.md).

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
  --history 24 --seed 42 --device auto --graph-ablations \
  --data-file data/combined_hourly_data.csv \
  --output-dir benchmark_results/experiment_1_vineyard/confirmatory_seed42
```

Experimento 2 (UVA, mesma política):

```bash
lora-benchmark --horizon 1 --optimize \
  --trials 12 --epochs 64 --search-epochs 64 --patience 10 \
  --history 24 --seed 42 --device auto --graph-ablations \
  --data-file data/experiment_2_uva_gatewayA_hourly.csv \
  --output-dir benchmark_results/experiment_2_uva_gatewayA/confirmatory_seed42
```

`--device auto` (padrão) seleciona CUDA quando disponível e CPU caso contrário; a opção
pode ser omitida. `--device cuda` força GPU e falha explicitamente se ela não estiver
disponível. Não há fallback silencioso após erros de treinamento. `--search-epochs`
assume o mesmo limite de `--epochs`
quando omitido; ambos usam early stopping na validação com `--patience 10`.

Sem `--optimize`, reutiliza-se o cache da última seleção para o mesmo dataset e histórico.
`--use-defaults` é a única forma explícita de usar configurações internas.
O cache só aceita exatamente o conjunto atual de modelos. Use um diretório novo por
experimento; resultados existentes são protegidos.

```bash
lora-benchmark --horizon 1 --epochs 64 --seed 42 --device auto \
  --data-file data/combined_hourly_data.csv \
  --output-dir benchmark_results/experiment_1_vineyard/repeat_seed42
```

O comando equivalente sem instalar o pacote é `python -m src.cli.run_benchmark`.

## Protocolo metodológico

- Divisão cronológica 70/15/15, scalers ajustados apenas no treino.
- Sem imputação; entradas e alvos devem ser observados e horários consecutivos.
  Clima futuro não é entrada nem requisito de elegibilidade.
- Todos os modelos são comparados nas mesmas origens completas.
- ARIMAX segue Box–Jenkins por nó: ADF/KPSS restringe `d`; quando ADF rejeita raiz
  unitária mas KPSS também rejeita estacionariedade, mantém-se `d=0` para evitar
  sobrediferenciação e o conflito é registrado como alerta de instabilidade estrutural.
  ACF/PACF da série já
  transformada propõe `p` e `q`. Somente o modelo sugerido, vizinhos imediatos e formas
  simples pré-definidas são estimados por máxima verossimilhança gaussiana em segmentos
  horários independentes. AICc seleciona `(p,d,q)` e Ljung–Box diagnostica os resíduos.
  Os correlogramas completos até 24 horas permanecem no relatório para auditoria.
- A adequação do VAR em níveis é verificada por diagnóstico de integração e posto de
  Johansen no treino, com sensibilidade a 1--3 defasagens das diferenças. Posto reduzido
  interrompe a execução e exige avaliar VECM/VECMX. VARX avalia estruturas
  densas `1...p`, com `p` até 24 horas, e estruturas sazonais esparsas
  `{1,6,12,24}` e `{1,2,3,6,12,24}`. O BIC de treino seleciona a estrutura apenas
  entre modelos na mesma escala em nível; dinâmicas AR instáveis são rejeitadas e
  segmentos separados nunca são concatenados artificialmente.
- Modelos neurais treinam e selecionam checkpoints por MAE na escala física,
  ponderando erros normalizados pela amplitude de treino de cada nó.
- Hiperparâmetros e checkpoints usam apenas validação; busca e treino final têm,
  por padrão, o mesmo limite de épocas e a mesma paciência.
- Métricas por nó são primárias. Média global, erro terminal e por lead são
  complementares. RSSI está em dBm; erros e chaves JSON `mae_db/rmse_db` estão em dB.
- Armazenamento de parâmetros é estimado; não inclui runtime, buffers ou ativações.
  Tempo de fit não inclui busca; relatórios de busca registram os tempos dos trials.
- Consolidação operacional é medida primariamente pelo número de instâncias que
  precisam ser treinadas, versionadas e servidas para cobrir todos os nós. Parâmetros
  e armazenamento complementam essa contagem; latência não é uma métrica do estudo.
- Protocolos preservam hardware, versões, seed, hash do dataset e dos fontes.
  NPZs incluem timestamps da primeira previsão de cada origem, verdade e previsões.

## Interpretação e limites

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

## Detalhes das famílias pareadas

O benchmark compara ARIMAX e VARX; Seq2Seq dedicado e integrado; NLinear e DLinear
dedicados e integrados; e o PhysicalAdaptive_STGNN multinó. Os modelos estatísticos usam
temperatura, umidade, pressão e chuva somente até a origem: para horizonte H, o
regressor meteorológico é defasado H horas. Cada ARIMAX tem ordem `(p,d,q)` própria;
o VARX estima conjuntamente as dependências defasadas entre todos os nós. Nenhum modelo
consulta clima futuro.

O histórico padrão é 24 horas e o benchmark usa previsão one-step-ahead (H=1). Os pares
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
  --seed 42 --device auto --graph-ablations \
  --output-dir benchmark_results/experiment_1_vineyard/confirmatory_seed42
```
