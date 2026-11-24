# Spotify Sentiment Analysis — Documentação do Projeto

*Classificação de sentimentos em avaliações do Spotify com TF-IDF e Naive Bayes Multinomial · Documentação gerada em 22/11/2026 a partir do repositório (commit `0d8828e`)*

## 1. Visão geral

O **Spotify Sentiment Analysis** é uma aplicação completa de Machine Learning que recebe o texto de uma avaliação (*review*) do aplicativo Spotify e devolve:

- o **sentimento previsto**: `Positivo`, `Negativo` ou `Neutro`;
- a **probabilidade (%) de o texto pertencer a cada uma das três classes**.

O projeto cobre o ciclo completo: exploração do dataset → pré-processamento → treinamento → avaliação → persistência do modelo → API REST → interface web.

**Resultado final no conjunto de teste (12.269 avaliações):** acurácia de **74,79%**, F1 macro de **0,6378** e F1 ponderado de **0,7573**.

| Camada | Tecnologias |
|---|---|
| Dados | `dataset/reviews.csv`: 61.594 avaliações de usuários do Spotify |
| Machine Learning | Python 3.12, pandas, scikit-learn (`TfidfVectorizer` + `MultinomialNB`), joblib |
| Backend | FastAPI 0.165.0, Uvicorn 0.54.0, Pydantic 2.14.4 |
| Frontend | React 19, TypeScript 6, Vite 8, Tailwind CSS 4 |

> **Sobre a terminologia:** o enunciado fala em classes “positivo, negativo ou nulo”. No código do projeto a terceira classe se chama **`Neutro`** (avaliações de nota 3), e é assim que ela é tratada neste documento.

## 2. Arquitetura

```
Navegador (React/Vite, http://localhost:5173)
   |  POST /predict  {"review": "..."}
   v
API FastAPI (http://127.0.0.1:8000)
   |  model.predict / model.predict_proba
   v
Pipeline treinado (model/spotify_sentiment_model.pkl)
   TfidfVectorizer  ->  MultinomialNB
   ^
   |  joblib.dump
Notebooks de treinamento (notebooks/machine_learning_V2.ipynb)
```

**Fluxo de uma requisição**

1. O usuário digita a avaliação no frontend; o texto é sanitizado e enviado por `POST /predict`.
2. A API aplica o *pipeline* carregado do `.pkl`: o texto é vetorizado (TF-IDF) e o Naive Bayes calcula a probabilidade de cada classe.
3. A API devolve o sentimento (classe de maior probabilidade) e as probabilidades em percentual.
4. O frontend exibe o sentimento e uma barra de probabilidade para cada classe.

## 3. Estrutura do repositório

```
spotify-sentiment-analysis/
├── README.md                         # dataset e etapas do projeto
├── dataset/reviews.csv               # 61.594 avaliações (≈ 11,8 MB)
├── notebooks/
│   ├── analise_exploratoria.ipynb    # EDA e primeiros experimentos
│   ├── machine_learning_V1.ipynb     # primeiro modelo melhorado
│   └── machine_learning_V2.ipynb     # versão final (GridSearchCV)
├── model/spotify_sentiment_model.pkl # Pipeline serializado (≈ 943 KB)
├── backend/main.py                   # API FastAPI
└── frontend/spotify-sentiment-analysis/
    ├── src/ (App.tsx, main.tsx, index.css)
    └── vite.config.ts, tsconfig*.json, eslint.config.js, package.json
```

> **Nota:** no `.zip` analisado, `backend/main.py` aparecia apenas como bytecode (`__pycache__/*.pyc`), mas o arquivo está versionado no Git. Esta documentação usa o conteúdo dele no commit mais recente.

## 4. Dataset

O dataset **Spotify App Reviews** reúne avaliações reais de usuários do aplicativo, publicadas entre **01/01/2022 e 09/07/2022**.

| Atributo | Descrição | Tipo |
|---|---|---|
| `Time_submitted` | Data e hora da publicação | Data/hora |
| `Review` | Texto da avaliação (entrada do modelo) | Texto |
| `Rating` | Nota de 1 a 5 (origem da variável-alvo) | Inteiro |
| `Total_thumbsup` | Curtidas recebidas pela avaliação | Inteiro |
| `Reply` | Resposta da equipe do Spotify | Texto |

**Diagnóstico (verificado nos dados):**

- 61.594 linhas × 5 colunas; nenhuma linha totalmente duplicada.
- Único campo com valores ausentes: `Reply` (61.378 nulos; só 216 avaliações, 0,35%, têm resposta). Por isso a coluna não é usada no modelo.
- Tamanho dos textos: média de 163 caracteres, mediana de 130, mínimo de 10 e máximo de 3.753.
- `Total_thumbsup`: mediana 0, média 7,65 e máximo 8.195 (distribuição muito assimétrica).

| Nota | Quantidade | % | Sentimento |
|---|---:|---:|---|
| 1 | 17.653 | 28,66% | Negativo |
| 2 | 7.118 | 11,56% | Negativo |
| 3 | 6.886 | 11,18% | Neutro |
| 4 | 7.842 | 12,73% | Positivo |
| 5 | 22.095 | 35,87% | Positivo |

Resultado: **Positivo 29.937 (48,60%) · Negativo 24.771 (40,22%) · Neutro 6.886 (11,18%)**. O conjunto é **desbalanceado**, com Neutro como classe minoritária, o que orienta várias decisões descritas adiante.

## 5. Categorização do sentimento (Positivo, Negativo e Neutro)

Regra aplicada pela função `classificar_sentimento`:

| Nota (`Rating`) | `Sentiment` |
|---|---|
| 1 ou 2 | `Negativo` |
| 3 | `Neutro` |
| 4 ou 5 | `Positivo` |

### Justificativa

A categorização em **positivo, negativo ou neutro** existe para **gerar um resultado compreensível sobre a avaliação**: em vez de devolver apenas um texto cru ou uma nota de 1 a 5, o sistema responde com uma polaridade que qualquer pessoa entende de relance. Há também razões técnicas:

1. **Alvo supervisionado sem rotulagem manual.** O Naive Bayes precisa de classes discretas. Derivar a classe da nota já existente rotula automaticamente mais de 61 mil textos.
2. **Notas vizinhas não se distinguem pelo vocabulário.** No experimento exploratório com as 5 notas como classes, a acurácia foi de apenas 59,55% (F1 macro 0,32), com recall de 0,00 para a nota 2 e 0,02 para a nota 3, indicando que notas vizinhas compartilham vocabulário muito semelhante. Agrupando em 3 classes, o mesmo modelo-base subiu para 76,95% de acurácia.
3. **Neutro representa opiniões ambivalentes** (“é bom, mas…”), que seriam forçadas a um dos extremos se houvesse só duas classes.

**Limitação:** o rótulo vem da nota, não da leitura do texto, então há *ruído de rótulo* (por exemplo, quem elogia o app mas atribui nota 3). Neutro é a classe mais difícil de prever (seção 11).

## 6. Pré-processamento (versão final, notebook V2)

| Etapa | Operação | Linhas restantes |
|---|---|---:|
| 1 | Carregar o CSV | 61.594 |
| 2 | Remover `Rating`/`Review` nulos; converter `Rating` para inteiro | 61.594 |
| 3 | Normalizar espaços em branco, remover textos vazios e manter notas de 1 a 5 | 61.594 |
| 4 | Remover duplicatas exatas de (`Review`, `Rating`) | 61.393 |
| 5 | Remover os **11 textos conflitantes** (mesmo texto com sentimentos diferentes) e duplicatas restantes por texto | 61.345 |

Remover duplicatas e conflitos evita **vazamento de dados** (o mesmo texto no treino e no teste) e rótulos contraditórios. Após o preparo: Positivo 29.758 · Negativo 24.714 · Neutro 6.873.

**Divisão treino/teste:** 80% / 20%, **estratificada** por sentimento, com `random_state=42`, resultando em **49.076** textos de treino e **12.269** de teste. A estratificação preserva a proporção de cada classe nos dois conjuntos; a semente torna o experimento reprodutível.

## 7. Extração de atributos: TF-IDF

O texto é convertido em vetores numéricos pelo `TfidfVectorizer`. O TF-IDF pondera cada termo pela frequência no documento (TF) e pela raridade no corpus (IDF), reduzindo o peso de palavras comuns e destacando as discriminativas.

| Parâmetro | Valor | Motivo |
|---|---|---|
| `lowercase` | `True` | “Great” e “great” são o mesmo termo |
| `stop_words` | 308 palavras: lista inglesa do scikit-learn menos `not`, `no`, `nor`, `never`, `neither`, `but`, `however`, `although`, `though`, `yet` | Preserva **negações e conectivos adversativos**, que invertem ou matizam o sentimento |
| `ngram_range` | `(1, 2)` | Bigramas capturam expressões como “worst app” e “easy use” |
| `max_features` | 10.000 | Limita o vocabulário aos termos mais frequentes |
| `sublinear_tf` | `True` | Usa `1 + log(tf)`, evitando que repetições dominem |
| `min_df` / `max_df` | `2` / `0,95` | Ignora termos raros (ruído) e quase onipresentes; valores definidos por busca em grade |
| `token_pattern` | `(?u)\b\w[\w']+\b` | Mantém apóstrofos internos (`don't`) e descarta tokens de 1 caractere |
| `norm` | `l2` (padrão) | Normaliza o comprimento dos vetores |

O vocabulário final tem exatamente **10.000 termos** (unigramas e bigramas).

## 8. Algoritmo: Naive Bayes Multinomial

O classificador é o `MultinomialNB`. Ele aplica o **Teorema de Bayes** com a hipótese “ingênua” de que, dada a classe, os termos são condicionalmente independentes:

```
P(c | texto)      ∝  P(c) · ∏ P(termo_i | c) ^ x_i
log P(c | texto)  =  log P(c) + Σ x_i · log P(termo_i | c)
classe prevista   =  argmax_c  log P(c | texto)
```

em que `x_i` é o peso TF-IDF do termo `i` no texto. O cálculo é feito em logaritmos para evitar *underflow*, e as probabilidades devolvidas por `predict_proba` são o resultado normalizado (somam 100%).

### Justificativa do tipo Multinomial

O tipo **Multinomial** é utilizado para **estimar a probabilidade de um texto pertencer a uma determinada classe**. Ele modela o documento como uma “sacola de palavras” sorteada de uma distribuição multinomial própria de cada classe; assim, a probabilidade da classe é calculada a partir da **frequência (aqui, o peso TF-IDF) de cada termo em cada classe**. Essa saída, em forma de distribuição de probabilidades, é exatamente o que a API e a interface exibem. Comparação com as demais variantes:

| Variante | Premissa sobre os atributos | Adequação ao projeto |
|---|---|---|
| **Multinomial** | Contagens ou pesos de termos | **Escolhida**: padrão para classificação de texto baseada em frequências |
| Bernoulli | Apenas presença ou ausência | Descarta frequência e pesos TF-IDF |
| Gaussian | Valores contínuos com distribuição normal | Inadequada para vetores esparsos de palavras |
| Complement | Variante do Multinomial para dados desbalanceados | Testada na análise exploratória (sem saída registrada); o desbalanceamento foi tratado com `fit_prior=False` |

Além disso, o Multinomial treina e prediz em milissegundos, funciona bem em espaços esparsos de alta dimensão, gera um modelo pequeno (≈ 943 KB) e é interpretável. A distribuição multinomial pressupõe contagens inteiras, mas o scikit-learn documenta que pesos fracionários como o TF-IDF também funcionam na prática, e os resultados deste projeto são coerentes com isso.

**Termos mais característicos aprendidos** (maior diferença de log-probabilidade entre classes):

- **Negativo:** `worst app`, `app sucks`, `garbage`, `unusable`, `useless`
- **Positivo:** `best music`, `thank spotify`, `excellent`, `easy use`, `best app`
- **Neutro:** sem padrão semântico claro (ex.: `galaxy s20`, `drop menu`, `idk happened`), sinal de que a classe é heterogênea.

**Prior das classes:** o modelo usa `fit_prior=False`, ou seja, prior uniforme (1/3 para cada classe). Com priors proporcionais à frequência, o modelo praticamente ignorava Neutro (11% dos dados): no experimento exploratório o F1 de Neutro foi 0,00; com `fit_prior=False` subiu para 0,26 e o F1 macro foi de 0,54 para 0,62.

## 9. Suavização de Laplace

### O problema: probabilidade zero

Na estimativa por frequência, `P(termo | classe) = contagem(termo, classe) / total(classe)`. Se um termo do vocabulário **nunca apareceu** em textos de uma classe durante o treino, sua probabilidade é **zero**. Como o Naive Bayes **multiplica** as probabilidades (ou soma logaritmos, e `log 0 = −∞`), basta uma ocorrência desse termo para **anular a probabilidade da classe inteira**, mesmo que todos os outros termos apontem para ela.

### Justificativa

A suavização de Laplace é utilizada para **evitar casos de probabilidade zero**: soma-se `α` a toda contagem, garantindo que nenhum termo tenha probabilidade nula em nenhuma classe.

```
P(termo | c) = ( contagem(termo, c) + α ) / ( total(c) + α · |V| )
```

No projeto, `α = 1,0` (suavização de Laplace propriamente dita; valores diferentes de 1 são chamados de Lidstone), `|V| = 10.000` e `total(c)` é a soma dos pesos TF-IDF da classe (Negativo 86.982,3 · Neutro 23.775,3 · Positivo 81.769,5). O valor `α = 1,0` foi o escolhido pela busca em grade entre 0,1, 0,5, 1,0 e 2,0.

### Exemplo real calculado no modelo salvo

Termo `awesome music`, com a probabilidade do termo dada cada classe:

| Classe | Contagem TF-IDF | Sem suavização | Com Laplace (α = 1) |
|---|---:|---:|---:|
| Negativo | 0,0000 | **0** | 1,03 × 10⁻⁵ |
| Neutro | 0,1552 | 6,53 × 10⁻⁶ | 3,42 × 10⁻⁵ |
| Positivo | 29,1976 | 3,57 × 10⁻⁴ | 3,29 × 10⁻⁴ |

A suavização **resgata o zero** da classe Negativo e **quase não altera** a classe com muita evidência (Positivo). Esse caso é comum: **1.202 dos 10.000 termos (12%)** têm contagem zero em ao menos uma classe (167 em Negativo, 848 em Neutro e 309 em Positivo).

> **Atenção:** a suavização resolve termos *presentes no vocabulário, porém ausentes em uma classe*. Termos fora do vocabulário (por exemplo, palavras de outro idioma) são simplesmente ignorados pelo vetorizador. Se **nenhum** termo do texto for conhecido, as probabilidades ficam iguais ao prior (33,33% cada) e o desempate recai sobre a primeira classe, `Negativo`.

## 10. Otimização de hiperparâmetros

O notebook V2 usa `GridSearchCV` com validação cruzada de 5 dobras e métrica **F1 macro**:

| Hiperparâmetro | Valores testados | Melhor |
|---|---|---|
| `tfidf__min_df` | 1, 2, 3 | **2** |
| `tfidf__max_df` | 0,95; 1,0 | **0,95** |
| `naive_bayes__alpha` | 0,1; 0,5; 1,0; 2,0 | **1,0** |
| `naive_bayes__fit_prior` | True, False | **False** |

São 48 combinações × 5 dobras = **240 treinamentos**. O melhor F1 macro na validação cruzada foi **0,6386**, coerente com o 0,6378 obtido no teste (sem sinal de ajuste excessivo à busca).

**Por que F1 macro?** Com classes desbalanceadas, a acurácia pode ficar alta mesmo ignorando Neutro. O F1 macro dá o mesmo peso a cada classe e penaliza esse comportamento.

## 11. Avaliação e resultados

### Evolução dos experimentos

| Experimento | Configuração principal | Acurácia | F1 macro | F1 Neutro |
|---|---|---:|---:|---:|
| EDA, 5 classes (notas) | TF-IDF 5.000 termos, `MultinomialNB()` padrão | 0,5955 | 0,32 | n/a |
| EDA, 3 classes | Idem, priors pela frequência | 0,7695 | 0,54 | 0,00 |
| EDA, 3 classes | `fit_prior=False` | 0,72 | 0,62 | 0,26 |
| V1 | Bigramas, 10.000 termos, `sublinear_tf`, stopwords sem negações, α = 0,5, `fit_prior=False` | 0,7216 | 0,6265 | 0,29 |
| **V2 (final)** | V1 + deduplicação, remoção de conflitos, `token_pattern`, conectivos preservados e `GridSearchCV` | **0,7479** | **0,6378** | 0,29 |

A acurácia caiu de 0,77 para 0,72 ao equilibrar os priors: o modelo passou a errar mais nas classes grandes para acertar mais Neutro, troca deliberada medida pelo F1 macro. Os conjuntos de teste de V1 e V2 não são idênticos (V2 remove duplicatas), então essa comparação é indicativa.

### Modelo final (12.269 avaliações de teste)

| Classe | Precisão | Recall | F1 | Suporte |
|---|---:|---:|---:|---:|
| Negativo | 0,77 | 0,80 | 0,78 | 4.943 |
| Neutro | 0,25 | 0,33 | 0,29 | 1.375 |
| Positivo | 0,89 | 0,80 | 0,85 | 5.951 |
| **Acurácia** | | | **0,7479** | 12.269 |
| Média macro | 0,64 | 0,64 | 0,64 | 12.269 |
| Média ponderada | 0,77 | 0,75 | 0,76 | 12.269 |

**Matriz de confusão** (linhas: classe real; colunas: classe prevista):

| Real / Previsto | Negativo | Neutro | Positivo |
|---|---:|---:|---:|
| Negativo | 3.948 | 678 | 317 |
| Neutro | 664 | 448 | 263 |
| Positivo | 533 | 638 | 4.780 |

Total de **3.093 erros (25,2%)**.

**Interpretação:** Positivo e Negativo são bem separados (F1 de 0,85 e 0,78). Neutro é o ponto fraco (F1 0,29): dos 1.375 neutros reais, 927 (67%) foram classificados como polares. A análise de erros do notebook mostra o motivo: muitas avaliações têm **sentimento misto** (por exemplo, “Great app, but adding songs to playlists is a…” é Positivo, mas foi prevista como Neutro), e um modelo baseado em palavras e bigramas não captura contraste nem ironia.

### Testes qualitativos (modelo salvo; probabilidades em %)

| Avaliação | Previsto | Negativo | Neutro | Positivo |
|---|---|---:|---:|---:|
| Spotify is amazing! I love listening to music here. | Positivo | 0,46 | 1,70 | 97,84 |
| Spotify keeps crashing. This is very disappointing. | Negativo | 70,13 | 28,06 | 1,81 |
| The app is okay, nothing special. | Neutro | 24,39 | 43,65 | 31,96 |
| I don't like Spotify anymore. | Negativo | 46,34 | 32,51 | 21,14 |
| The music selection is good, but the interface could be better. | Positivo | 4,23 | 11,91 | 83,86 |
| I love the music but hate the ads. | Positivo | 7,53 | 24,50 | 67,97 |
| Spotify is not bad. | Neutro | 29,93 | 41,49 | 28,58 |
| This app is absolutely terrible. | Negativo | 62,28 | 20,36 | 17,37 |

## 12. API REST (FastAPI)

O arquivo `backend/main.py` (43 linhas) carrega o pipeline com `joblib.load` **uma única vez**, na inicialização, e o reutiliza em todas as requisições.

| Método | Rota | Descrição |
|---|---|---|
| GET | `/` | Verificação de saúde: retorna `{"message": "Spotify Sentiment Analysis API"}` |
| POST | `/predict` | Classifica uma avaliação |

**Requisição** (modelo `ReviewRequest`, validado pelo Pydantic): `{"review": "texto da avaliação"}`.

**Resposta 200:**

```json
{
  "review": "Spotify is amazing! I love listening to music here.",
  "sentiment": "Positivo",
  "probabilities": { "Negativo": 0.46, "Neutro": 1.7, "Positivo": 97.84 }
}
```

- `sentiment`: classe prevista por `model.predict`.
- `probabilities`: resultado de `model.predict_proba` multiplicado por 100 e arredondado a 2 casas (o JSON omite zeros à direita, por isso `1.7`). A ordem segue `model.classes_`: Negativo, Neutro, Positivo.

Exemplo de chamada:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"review": "Spotify is amazing! I love listening to music here."}'
```

**Comportamento verificado em testes:**

| Cenário | Resultado |
|---|---|
| Campo `review` ausente | `422` (erro `missing` em `body.review`) |
| `review` que não é texto (ex.: `123`) | `422` (erro `string_type`) |
| `review` vazio (`""`) | `200`, com 33,33% para cada classe e `sentiment` = `Negativo` (não há validação de conteúdo) |
| Origem `http://localhost:5173` | CORS permitido |
| Origem `http://127.0.0.1:5173` | Pré-voo CORS recusado (`400`) |

**CORS:** `allow_origins=["http://localhost:5173"]` (endereço padrão do Vite), com todos os métodos e cabeçalhos liberados. A documentação interativa (Swagger) é gerada automaticamente pelo FastAPI em `/docs`, e o esquema OpenAPI em `/openapi.json`.

**Execução:** `uvicorn main:app --reload`, a partir da pasta `backend/` (porta padrão 8000, a mesma usada pelo frontend).

## 13. Frontend (React + TypeScript)

Aplicação de página única em `frontend/spotify-sentiment-analysis/src/App.tsx` (componente `App`), renderizada por `main.tsx` e estilizada com Tailwind CSS 4 (`@import "tailwindcss"` em `index.css`, plugin `@tailwindcss/vite`).

**Comportamento**

- Estados: `review`, `result`, `loading` e `error`; a resposta é tipada pela interface `PredictionResponse`.
- Ao enviar o formulário, faz `POST http://127.0.0.1:8000/predict` com `{ review }`; uma resposta não-OK gera a mensagem “Failed to classify review.”.
- O botão fica desabilitado com o campo vazio ou durante o carregamento (“Analyzing...”), e digitar limpa o resultado e o erro anteriores.
- Exibe o sentimento previsto e uma barra de progresso por classe (largura igual à probabilidade). Cores: Positivo `#1DB954`, Negativo `#F87171` e Neutro `#9CA3AF`, em tema escuro inspirado no Spotify.
- Rodapé: “Powered by Machine Learning & Multinomial Naive Bayes”. A interface está em inglês, o idioma do dataset.

**Sanitização de entrada** (commit `0d8828e`): o `onChange` do campo aplica `replace(/[^a-zA-Z\s]/g, "")`, mantendo apenas letras de A a Z e espaços. Isso remove números, pontuação, emojis, **acentos e apóstrofos**. Verificado: `I don't like it! 10/10, 😀 ótimo` vira `I dont like it   timo`. O benefício é proteger o modelo de ruído; o custo é perder sinais úteis. O vetorizador foi configurado para preservar apóstrofos (`don't`), mas o frontend os remove (“dont” é outro termo). Exemplo real: `I don't like Spotify anymore.` resulta em Negativo com 46,34%, e `I dont like Spotify anymore` em Negativo com 49,50%.

**Scripts** (`package.json`): `npm run dev` (Vite), `npm run build` (`tsc -b && vite build`), `npm run lint` (ESLint com typescript-eslint, react-hooks e react-refresh) e `npm run preview`. O `tsconfig.app.json` ativa `noUnusedLocals` e `noUnusedParameters`.

## 14. Como executar

**Pré-requisitos:** Python 3.12, Node.js e npm. O repositório não tem `requirements.txt`; as dependências abaixo foram identificadas no ambiente virtual enviado.

**1) Backend**

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: .\venv\Scripts\activate
pip install fastapi "uvicorn[standard]" scikit-learn joblib
```

Em `main.py`, substitua o caminho absoluto do Windows por um caminho relativo ao arquivo:

```python
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "spotify_sentiment_model.pkl"
model = joblib.load(MODEL_PATH)
```

Depois, inicie a API com `uvicorn main:app --reload`.

**2) Frontend**

```bash
cd frontend/spotify-sentiment-analysis
npm install
npm run dev
```

Abra http://localhost:5173 (use `localhost`, e não `127.0.0.1`, por causa do CORS).

**3) Retreinar o modelo**

1. Instale `pandas matplotlib seaborn scikit-learn joblib jupyter`.
2. Abra `notebooks/machine_learning_V2.ipynb` e ajuste o caminho do CSV (hoje um caminho absoluto do Windows).
3. Execute todas as células (a busca em grade faz 240 treinamentos).
4. O notebook grava `spotify_sentiment_model.pkl` no diretório de trabalho; copie o arquivo para a pasta `model/`.

## 15. Pontos de atenção e recomendações

| # | Ponto | Impacto | Recomendação |
|---|---|---|---|
| 1 | Caminho absoluto do Windows (`C:\Users\nando\Desktop\...`) em `main.py` e nos notebooks | API e notebooks só rodam na máquina original | Usar `pathlib` relativo ou variável de ambiente |
| 2 | Modelo serializado com scikit-learn **1.7.2**, enquanto o ambiente tem **1.9.1** (aviso `InconsistentVersionWarning` ao carregar) | Risco de resultados inválidos ou falhas em atualizações | Fixar a versão no `requirements.txt` ou retreinar e salvar com a versão em uso |
| 3 | Sem `requirements.txt` e sem `.gitignore` na raiz; há um `.pyc` versionado e o `package-lock.json` da raiz está vazio | Ambiente não reproduzível e repositório com arquivos desnecessários | Criar `requirements.txt` e `.gitignore` (`venv/`, `__pycache__/`) e remover o lockfile da raiz |
| 4 | O modelo só entende inglês | Texto em português cai em 33,33% por classe e vira `Negativo` por desempate | Detectar o idioma, rejeitar baixa confiança ou treinar com dados em português |
| 5 | A sanitização do frontend remove apóstrofos e acentos | Perde negações como `don't` | Preservar apóstrofos (`/[^a-zA-Z\s']/g`) ou normalizar o texto no backend |
| 6 | Neutro tem F1 de 0,29 | Avaliações mistas são mal classificadas | Mais dados neutros e modelos que capturem contraste (regressão logística, transformers) |
| 7 | As probabilidades do Naive Bayes não são calibradas (hipótese de independência) | Podem ser extremas ou subestimadas | Calibrar (`CalibratedClassifierCV`) e apresentá-las como “confiança do modelo” |
| 8 | API sem validação de conteúdo (vazio ou muito longo), com duas inferências por requisição e sem testes ou logs | Resposta enganosa para entradas vazias e custo desnecessário | `min_length`/`max_length` no Pydantic, derivar a classe de `predict_proba` e adicionar testes (`pytest` + `TestClient`) |
| 9 | URL da API e origem do CORS fixas no código | Dificulta a implantação | Variáveis de ambiente (por exemplo, `VITE_API_URL` e lista de origens) |
| 10 | Rótulos derivados da nota | O ruído de rótulo limita o desempenho máximo | Revisão manual de uma amostra dos rótulos |

## 16. Glossário

| Termo | Definição |
|---|---|
| Naive Bayes | Classificador probabilístico baseado no Teorema de Bayes, com independência condicional entre atributos |
| Multinomial | Variante do Naive Bayes para atributos de contagem ou peso, típica de texto |
| Suavização de Laplace | Soma `α = 1` às contagens para evitar probabilidades zero |
| Lidstone | Generalização da suavização com `α` diferente de 1 |
| Prior | Probabilidade da classe antes de observar o texto |
| TF-IDF | Peso que combina a frequência do termo no documento e sua raridade no corpus |
| N-grama | Sequência de N palavras consecutivas (bigrama: 2) |
| Stop words | Palavras muito comuns removidas antes da vetorização |
| F1 macro | Média simples do F1 de cada classe, indiferente ao tamanho das classes |
| Matriz de confusão | Tabela de acertos e erros por classe real e classe prevista |
| Vazamento de dados | Informação do conjunto de teste que “vaza” para o treino e infla as métricas |
| Pipeline | Encadeamento de vetorizador e classificador, salvo como um único objeto |

## 17. Histórico de commits

| Commit | Data | Mensagem |
|---|---|---|
| `1b92b42` | 10/11/2026 | first commit (README e dataset) |
| `f9c5f3b` | 19/11/2026 | começo do machine learning (análise exploratória e V1) |
| `aebf4ab` | 20/11/2026 | machine-learning_V2 e inicialização do front |
| `3acf847` | 20/11/2026 | quase finalizado (API, interface e modelo movido para `model/`) |
| `0d8828e` | 22/11/2026 | barramento de caracteres especiais (sanitização no frontend) |

*Nota de verificação: as métricas, as probabilidades, o exemplo de Laplace e o comportamento da API descritos neste documento foram reproduzidos executando o modelo salvo e a API com os dados do repositório.*
