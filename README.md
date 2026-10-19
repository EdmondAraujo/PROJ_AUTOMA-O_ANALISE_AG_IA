# Agente de Experimentação de Growth (A/B Testing)

Agente de linha de comando (CLI) em Python que lê um CSV de testes A/B em
nível de usuário, roda verificações de SRM (Sample Ratio Mismatch) e de
significância de taxa de conversão localmente com `scipy`, e alimenta o
Claude com os números exatos para gerar um *executive readout*.

## Stack

- **Python 3.9+**, `pandas`, `scipy`, `numpy`, `anthropic` (CLI pura, sem UI web).
- Toda a estatística é calculada localmente com `scipy`/`numpy` — o Claude
  recebe os números exatos e **não** refaz as contas.

## Início rápido

```bash
pip install -r requirements.txt

# Opcional: regenerar o dataset sintético incluído no repositório
python generate_mock_data.py --overwrite

# Analisar (requer ANTHROPIC_API_KEY)
set ANTHROPIC_API_KEY=...          # PowerShell: $env:ANTHROPIC_API_KEY="..."
python main.py --csv data/mock_ab_test.csv

# Analisar sem chamar a API (imprime o prompt exato enviado ao Claude)
python main.py --csv data/mock_ab_test.csv --skip-api
```

> **Nota de instalação:** as dependências de runtime vivem em
> `requirements.txt` (fonte única, para evitar divergência). Um eventual
> `pip install .` instala o pacote **sem** as dependências — use sempre
> `pip install -r requirements.txt`.

Setup de desenvolvimento e testes:

```bash
pip install -r requirements-dev.txt
python -m pytest test_main.py test_generate.py -v
```

São **81 testes** (64 em `test_main.py` + 17 em `test_generate.py`).

## Métodos estatísticos

### Verificação de SRM (Sample Ratio Mismatch)

- Compara o split observado control/tratamento com a razão esperada.
- **Método de decisão:** `chi-square` quando todas as células esperadas são
  ≥ 5; caso contrário `exact-binomial` (a aproximação qui-quadrado é
  não confiável para contagens esperadas pequenas). A decisão passa a usar o
  p-valor do teste exato, e o prompt reflete isso
  (`SRM DECISION method/p-value`).
- Alpha de SRM separado (`--srm-alpha`, padrão `0.05`) do alpha de conversão.

### Significância de conversão

- Lift absoluto/relativo, p-valor do teste z (bicaudal, com p pooling) e
  intervalo de confiança Newcombe.
- **IC:** Newcombe (1998) **método 10** (combinação "square-and-add" das
  distâncias das estimativas de Wilson; cobertura próxima do nominal),
  **não** o limite conservador `[L2 − U1, U2 − L1]`.
- **Método de decisão:** `z-test` quando todas as células esperadas são ≥ 5;
  caso contrário `fisher-exact` (p-valor de Fisher decide e é rotulado
  como tal no prompt).
- Casos degenerados (conversão 0%/100%, braços vazios) retornam ICs finitos
  e `N/A` para MDE/power em vez de quebrar.
- MDE a 80% de power e power observado (pós-hoc, enviesado — apenas
  contexto) usam o SE não agrupado (Wald) das taxas observadas.

### Nível de confiança

- O nível do IC no prompt é derivado do `alpha`
  (`--alpha 0.01` → `99% CI`; padrão `0.05` → `95% CI`).
- Power pós-hoc é rotulado como não decisivo; o IC carrega a evidência.

## Formato do CSV de entrada

Colunas obrigatórias:

| coluna    | significado |
|-----------|-------------|
| user_id   | único e não vazio, lido como texto (sobrevive a zeros à esquerda) |
| variant   | `control` ou `treatment` (case-insensitive; espaços externos removidos) |
| converted | `0` ou `1` (somente inteiros — `0.5` é erro, não truncamento) |

A validação levanta:

- `FileNotFoundError` — o caminho não existe.
- `ValueError` — arquivo vazio, colunas ausentes, `user_id` duplicado,
  valores nulos, variante inesperada ou `converted` fora de `{0, 1}`.
- Colunas extras são ignoradas.

## Dataset sintético

`data/mock_ab_test.csv` é **versionado** (o teste
`test_bundled_dataset_end_to_end` depende dele) e gerado por
`generate_mock_data.py` com seed `42`:

- 5.000 linhas control / 5.000 treatment; **probabilidades-alvo** de conversão
  `0.048` vs `0.061`.
- As taxas **realizadas** divergem das alvos (amostragem aleatória) — em
  torno de `4,32%` vs `5,66%` (~216 vs ~283 conversões).
- User IDs neutros `uNNNNNN` atribuídos **após** o shuffle (não vazam o braço).
- O gerador valida os parâmetros (inclusive tipos não numéricos → `ValueError`),
  usa RNG local (não altera o estado global de `random`) e grava de forma
  atômica (arquivo temporário + `os.replace`).

Para regenerar após um wipe do diretório:

```bash
python generate_mock_data.py --overwrite
```

## Referência da CLI

```
python main.py --csv <arquivo.csv> [--expected-ratio 0.5] [--alpha 0.05]
               [--srm-alpha 0.05] [--min-effect 0.01]
               [--model claude-sonnet-4-5] [--max-tokens 1500]
               [--temperature <SDK>] [--timeout 60] [--max-retries 3]
               [--skip-api]
```

- `--csv` é **obrigatório** (não há argumento posicional).
- Defaults reais: `--max-tokens 1500`, `--max-retries 3`, `--timeout 60`,
  `--temperature` sem default (usa o default do SDK da Anthropic).
- Todos os flags numéricos são validados (`--max-tokens > 0`,
  `--timeout > 0`, `--max-retries >= 0`, `0 < alpha < 1`,
  `0 < expected-ratio < 1`, `0 < min-effect < 1`, `0 <= temperature <= 1`).

## Estrutura do projeto

```
main.py                # motor estatístico (SRM, z-test, Fisher, Newcombe,
                       # MDE/power), system prompt, cliente Claude,
                       # orquestrador e CLI
generate_mock_data.py  # gerador do CSV sintético
test_main.py           # testes de estatística + loader + prompt + pipeline
test_generate.py       # testes do gerador
data/mock_ab_test.csv  # dataset sintético versionado (seed 42)
```
