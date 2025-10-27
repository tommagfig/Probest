# EDA — NDT Subset

## Quantis escolhidos
- **q90** e **q99** são relevantes para desempenho de rede porque capturam *tail latency* (as maiores latências que impactam a experiência).
- **q95** dá um panorama intermediário entre mediana e cauda extrema.

## Arquivos gerados
- Tabelas: `eda_output/eda_summary.xlsx`
- Gráficos: client=client10, client=client13 (pasta `eda_output/`)

## Observações
- Compare os resumos por cliente e por servidor (planilhas `by_client` e `by_server`).
- Observe as diferenças em média, variância e quantis — especialmente q99 de RTT.
- Nos *scatter plots*, verifique a correlação entre RTT e Throughput (espera-se relação negativa em muitos cenários).