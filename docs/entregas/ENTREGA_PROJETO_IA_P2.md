# Projeto P2 — Inteligência Artificial e Machine Learning
**Entrega Oficial das Aulas 2 e 3**  
**Professor:** Lucas Hermenegildo do Nascimento  
**Curso:** Engenharia de Software / Ciência da Computação • Campus Saquarema  

---

## 1. Identificação do Grupo e do Projeto

* **Integrantes do Grupo:**
  1. Maicon Fonseca Dutra
  2. Gabriel Caniza da Silva
  3. Gabriel Vidal de Souza
* **Nome do Projeto:** `Previsao_Partidas_Futebol_P2`
* **Versão:** `v1.0.0`
* **Caminho Escolhido:** `modo: tabular`
* **Tarefa:** Classificação Multiclasse (`classificacao`)
* **Classes do Alvo:** `vitoria_mandante`, `empate`, `vitoria_visitante`
* **Métrica Principal:** `f1_macro` (métrica adequada para classes com desbalanceamento moderado)

---

## 2. Definição do Problema e Origem dos Dados

* **Pergunta Central do Projeto:**  
  *É possível prever o desfecho de uma partida internacional de futebol (vitória do mandante, empate ou vitória do visitante) utilizando apenas as informações conhecidas antes do apito inicial (seleções envolvidas, relevância do torneio e fator de campo neutro)?*
* **Origem da Base de Dados:**  
  Kaggle (*International football results from 1872 to 2024*). Trata-se de uma base de dados pública sob licença aberta (CC0: Public Domain), sem restrições de direitos autorais ou dados sensíveis/pessoais (LGPD).
* **Unidade de Observação:**  
  Cada linha da tabela representa uma partida internacional oficial de futebol disputada entre duas seleções nacionais.
* **Volume de Dados:**  
  49.945 partidas internacionais válidas (sem ausências e sem dados nulos nas features essenciais).

---

## 3. Contrato de Entrada e Saída (Prevenção de Falhas Silenciosas)

### Schema de Entrada (`contrato.py` / `validar_payload`)

| Campo | Tipo | Obrigatório | Opções Permitidas / Regra | Descrição e Unidade |
|---|---|---|---|---|
| `mandante` | `texto` | Sim | Texto não vazio | Nome da seleção que joga com o mando de campo |
| `visitante` | `texto` | Sim | Texto não vazio | Nome da seleção visitante |
| `torneio` | `texto` | Sim | Texto não vazio | Competição (ex: `Friendly`, `FIFA World Cup`, `Copa América`) |
| `campo_neutro` | `texto` | Sim | `["Sim", "Nao"]` | Indica se o confronto é em campo neutro |

### Estrutura da Resposta
```json
{
  "resultado": "vitoria_mandante",
  "versao": "1.0.0"
}
```

### Prevenção de Data Leakage (Vazamento de Dados Futuros)
As colunas originais `home_score` e `away_score` foram utilizadas **exclusivamente** para rotular a coluna alvo (`resultado`). Elas foram **estritamente excluídas** das features de entrada, pois gols só acontecem durante/após a partida. Colocar placar como entrada causaria vazamento grave de dados futuros.

---

## 4. Pipeline de Pré-processamento e Divisão dos Dados

1. **Pipeline de Transformação (`ColumnTransformer`):**
   * **Variáveis Categóricas (`mandante`, `visitante`, `torneio`, `campo_neutro`):**  
     Imputação com valor mais frequente (`SimpleImputer(strategy='most_frequent')`) + codificação One-Hot (`OneHotEncoder(handle_unknown='ignore')`). O parâmetro `handle_unknown='ignore'` assegura que seleções ou torneios inéditos que surjam em tempo de inferência não causem erro no pipeline.
2. **Divisão de Experimento (`preparar_experimento`):**
   * **Estratégia:** Divisão Aleatória Estratificada (`stratify=resultado`).
   * **Proporção:** 80% treino (39.956 partidas) e 20% teste reservado (9.989 partidas).
   * **Semente Fixa:** `seed: 42` para total reprodutibilidade.
   * **Hash de Rastreabilidade (`manifesto.json`):** Garante a integridade da configuração, código e base.

---

## 5. Resultados Quantitativos da Validação Cruzada (Aula 3)

Avaliação em 5 folds estratificados no conjunto de desenvolvimento comparando o baseline ingênuo contra o modelo treinado:

| Modelo / Abordagem | Tipo de Algoritmo | Acurácia Média | F1-Score Macro Médio | Interpretação |
|---|---|---|---|---|
| **Referência (Baseline)** | `DummyClassifier` (Classe Majoritária) | 49,00% | 0,2192 | Prevê sempre `vitoria_mandante`. F1 muito baixo porque erra 100% dos empates e vitórias de visitantes. |
| **Primeira Versão (IA)** | `LogisticRegression` + `OneHotEncoder` | **57,66%** | **0,4457** | Aprende a força histórica e mando de campo das seleções. **Ganho expressivo de +0,2265 no F1 Macro**. |

---

## 6. Resultados dos Casos de Teste (`casos_do_grupo.json`)

| ID do Caso | Entrada | Comportamento Esperado | Resultado Observado | Status |
|---|---|---|---|---|
| `brasil_vs_argentina_copa_america` | Brasil x Argentina (Copa América, Mando Brasil) | Resposta não vazia | `vitoria_mandante` | **PASSOU** |
| `franca_vs_alemanha_amigavel` | França x Alemanha (Amistoso, Campo Neutro) | Resposta não vazia | `vitoria_visitante` | **PASSOU** |
| `inglaterra_vs_escocia_eliminatorias` | Inglaterra x Escócia (Eliminatórias, Mando Inglaterra) | Resposta não vazia | `vitoria_mandante` | **PASSOU** |
| `japao_vs_espanha_copa_do_mundo` | Japão x Espanha (Copa do Mundo, Campo Neutro) | Resposta não vazia | `vitoria_visitante` | **PASSOU** |
| `caso_recusa_falta_mandante` | Entrada sem o campo `mandante` | Rejeição por contrato (`ValueError`) | `mandante: campo obrigatório.` | **PASSOU** |
| `caso_recusa_campo_neutro_invalido` | Campo neutro com valor `"Talvez"` | Rejeição por contrato (`ValueError`) | `campo_neutro: valor não permitido.` | **PASSOU** |
| `argentina_vs_franca_final` (Fase Final) | Argentina x França (Copa do Mundo, Neutro) | Resposta não vazia | `vitoria_mandante` | **PASSOU** |
| `portugal_vs_espanha_nations_league` (Fase Final) | Portugal x Espanha (Nations League, Mando Portugal) | Resposta não vazia | `vitoria_visitante` | **PASSOU** |

**Taxa de Sucesso nos Testes:** 100% em desenvolvimento e 100% na fase final.

---

## 7. Investigação de Erros Relevantes e Limitações (Aula 3 — Passo 7)

A análise do arquivo `previsoes_inicial.csv` revelou as causas típicas de erros:
1. **Dificuldade na Previsão de Empates:** No futebol, empates ocorrem por dinâmicas táticas circunstanciais da partida que modelos lineares simples têm dificuldade de capturar apenas pelo histórico.
2. **Ausência de Contexto Pré-Jogo Dinâmico:** A base não contempla desfalques por lesão, escalação titular, condições meteorológicas ou momento recente dos últimos 5 jogos da equipe.
3. **Zebras Históricas:** Partidas onde equipes menores venceram potências internacionais contrariam o padrão histórico do modelo.
* **Hipótese de Melhoria para a P2:** Incluir engenharia de atributos temporais (ex: aproveitamento nas últimas 5 partidas e ranking FIFA das seleções no ano do jogo).

---

## 8. Resposta às Perguntas de Checagem do Professor

* **Bloco 1 (`configurar`):**
  * *Qual problema evita?* Impede que o sistema rode com configurações genéricas/incompletas ou modos incompatíveis, travando a execução com mensagem clara no início.
* **Bloco 2 (`validar_payload`):**
  * *Qual problema evita?* Evita falhas silenciosas e tipos errados que quebrariam o modelo durante a inferência no produto final. Provado com os casos de teste de recusa que foram barrados com sucesso.
* **Bloco 3 (`checar_config`):**
  * *Qual problema evita?* Garante integridade matemática entre a tarefa (classificação) e a métrica (F1 Macro), impedindo cálculos de métricas de regressão em alvos discretos.
* **Bloco 4 (`carregar_dados`):**
  * *Qual problema evita?* Audita ausências, inconsistências e dados inválidos, salvando o relatório em `auditoria.json` antes de qualquer manipulação.
* **Bloco 5 (`preparar_experimento`):**
  * *Qual problema evita?* Evita contaminação entre treino e teste através da separação estratificada e controle de versão criptográfica (`manifesto.json`).
* **Bloco 6 (`construir_pipeline`):**
  * *Qual problema evita?* Evita vazamento de dados (data leakage), calculando estatísticas e categorias exclusivamente dentro dos dados de treino de cada fold.
* **Bloco 7 (`treinar_inicial`):**
  * *Qual problema evita?* Evita falsas impressões de acurácia salvando a métrica como diagnóstico de treino e persistindo o modelo ajustado em `modelo_inicial.joblib`.

---

## 9. Divisão das Responsabilidades no Grupo

* **Maicon Fonseca Dutra:** Configuração do ambiente, arquitetura dos contratos (`config.projeto.json`, `contrato.py`) e preparação do dataset sem vazamento de dados.
* **Gabriel Caniza da Silva:** Implementação do pipeline de pré-processamento categórico, auditoria dos dados e treinamento do modelo baseline linear (`motor_tabular.py`, `aula_02.py`).
* **Gabriel Vidal de Souza:** Implementação da validação cruzada estratificada em 5 folds, comparação com modelo de referência dummy e elaboração da suíte de casos de teste (`aula_03.py`, `casos_do_grupo.json`).

---

## 10. Como Executar o Projeto

No terminal do projeto (com o ambiente virtual ativo):

```powershell
# 1. Executar a Aula 02 (Preparação, auditoria, manifesto e baseline inicial):
.venv\Scripts\python aula_02.py --config config.projeto.json

# 2. Executar a Aula 03 (Validação cruzada em 5 folds, comparativo com baseline e testes):
.venv\Scripts\python aula_03.py --config config.projeto.json
```
