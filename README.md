# Basico

Pipeline em Python para previsão de gols e probabilidades de mercado com base em xG histórico.

## Como usar no Google Colab

### Opção 1 (recomendada): notebook pronto

Use o arquivo `colab_previsao_xg.ipynb` neste repositório.  
Ele já vem com células para:
- upload dos arquivos
- treino/execução do pipeline
- previsão da próxima rodada
- ranking de probabilidades (1X2, Over/Under e placar mais provável)
- export/download de `previsao_colab_analise.xlsx`

### Opção 2: execução manual por script

1. Faça upload dos arquivos:
   - `passadas.xlsx` (temporadas anteriores)
   - `atuais.xlsx` (temporada vigente)
   - `proxima.xlsx` (próxima rodada)
2. Faça upload também do script `modelo_v2.py`.
3. Rode no Colab:

```python
!pip install pandas numpy scipy scikit-learn openpyxl
!python modelo_v2.py
```

4. O resultado será salvo em `previsao_v2.xlsx`.

## Esquema mínimo esperado dos arquivos

- `passadas.xlsx` e `atuais.xlsx`:
  - `home`, `away`, `hg`, `ag`
  - `hxG`/`axG` (ou variantes como `hxg`, `axg`, `home_xg`, `away_xg`)
  - `ano` e `rodada` **opcionais**
  - `date` **opcional**

- `proxima.xlsx`:
  - `home`, `away`
  - `ano` e `rodada` **opcionais**
  - `date` **opcional**

## Regras de inferência temporal (quando não houver `ano`/`rodada`)

O script tenta preencher automaticamente:

1. Se houver `date`:
   - `ano` via ano ISO/data.
   - `rodada` pela semana ISO relativa ao início da temporada.
2. Se não houver `date`:
   - `rodada` por sequência de jogos de cada time (fallback).
   - `ano = 1` (temporada única, fallback).

> Recomendação: sempre que possível, incluir `date` para melhorar a ordenação temporal.
