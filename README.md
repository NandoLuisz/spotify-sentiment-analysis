## Dataset

O dataset **Spotify App Reviews** contém avaliações realizadas por usuários do aplicativo Spotify, incluindo informações sobre suas experiências e opiniões a respeito da plataforma.

Cada registro representa uma avaliação e apresenta informações como a data de publicação, o conteúdo textual da avaliação, a nota atribuída pelo usuário, a quantidade de curtidas recebidas e uma possível resposta da empresa.

O conjunto de dados será utilizado para desenvolver um modelo de Machine Learning capaz de realizar a **classificação de sentimentos** em avaliações textuais, identificando opiniões positivas, negativas e neutras.

### Atributos do dataset

| Atributo         | Descrição                                                       | Tipo de dado |
| ---------------- | --------------------------------------------------------------- | ------------ |
| `Time_submitted` | Data e horário em que a avaliação foi publicada.                | Data/Hora    |
| `Review`         | Texto contendo a avaliação escrita pelo usuário.                | Texto        |
| `Rating`         | Nota atribuída pelo usuário ao aplicativo, geralmente de 1 a 5. | Numérico     |
| `Total_thumbsup` | Quantidade de curtidas recebidas pela avaliação.                | Numérico     |
| `Reply`          | Resposta publicada pela equipe do Spotify, quando disponível.   | Texto        |

### Variável de classificação

Para o treinamento supervisionado, será utilizada a coluna `Rating` para gerar uma nova variável denominada `Sentiment`, responsável por representar a classificação do sentimento:

- **Negativo:** avaliações com notas 1 e 2.
- **Neutro:** avaliações com nota 3.
- **Positivo:** avaliações com notas 4 e 5.

A coluna `Review` será utilizada como entrada (_feature_) do modelo, enquanto `Sentiment` será a variável-alvo (_target_) que o algoritmo Naive Bayes deverá aprender a prever.

## Etapas do desenvolvimento

| Etapa | Atividade                                |
| :---: | ---------------------------------------- |
|   1   | Exploração e análise do dataset          |
|   2   | Pré-processamento e preparação dos dados |
|   3   | Treinamento do Naive Bayes               |
|   4   | Avaliação do modelo                      |
|   5   | Salvamento do modelo                     |
|   6   | Desenvolvimento da API                   |
|   7   | Desenvolvimento da interface             |
|   8   | Integração e testes                      |
