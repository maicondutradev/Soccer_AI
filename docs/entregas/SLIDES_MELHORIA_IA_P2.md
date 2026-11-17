# Apresentação: Como Melhorar Nossa IA?
## Projeto P2 — Inteligência Artificial e Machine Learning
**Professor:** Lucas Nascimento  
**Equipe:** Maicon Fonseca Dutra, Gabriel Caniza da Silva, Gabriel Vidal de Souza  
**Data:** Novembro de 2026  

---

<!-- SLIDE 1 -->
## Slide 1: Identificação do Projeto e Desafio Tecnológico
* **Nome do Projeto:** `Previsao_Partidas_Futebol_P2`
* **Tema:** Previsão multiclasse de resultados de futebol internacional pré-jogo.
* **Caminho:** Abordagem tabular estruturada com pipeline reprodutível.
* **Objetivo Desta Aula:**
  1. Identificar o desafio central do modelo inicial.
  2. Explorar e comparar técnicas e bibliotecas candidatas à evolução.
  3. Propor e justificar a escolha tecnológica da equipe para a P2.

---

<!-- SLIDE 2 -->
## Slide 2: O Problema, Público e Dados
### Resumo em 3 Frases:
1. **Problema:** Prever se uma partida internacional de futebol terminará em vitória do mandante, empate ou vitória do visitante utilizando unicamente variáveis conhecidas antes do início da partida.
2. **Dados Disponíveis:** Base histórica com 49.945 confrontos oficiais, contemplando seleções mandante/visitante, torneio e indicador de campo neutro (sem placar para evitar *data leakage*).
3. **Resultado Esperado:** Classificador com performance equilibrada entre todas as classes, especialmente na detecção de empates, otimizando o $F_1\text{-Macro}$.

* **Tipo de Tarefa de IA:** Classificação Multiclasse.
* **Público Impactado:** Analistas esportivos, torcedores e plataformas de mídia e estatísticas.

---

<!-- SLIDE 3 -->
## Slide 3: Comparação de Modelos (Técnica A vs Técnica B)

| Critério | Técnica A: Regressão Logística Multinomial (Atual) | Técnica B: Random Forest Classifier (Proposta) |
|---|---|---|
| **Mecanismo** | Combinação linear com normalização probabilística via *Softmax*. | *Ensemble* de centenas de árvores de decisão via *bagging*. |
| **Aplicação** | Classificação linear tabular de alta velocidade. | Classificação não linear complexa e relações entre variáveis. |
| **Vantagem** | Alta interpretabilidade e treinamento instantâneo. | **Aprende interações não lineares** e rivalidades entre seleções. |
| **Limitação** | Fronteira estritamente linear; erra sistematicamente empates. | Custo computacional superior e modelo "caixa preta". |
| **Interpretabilidade** | Alta (pesos diretos por seleção). | Média (requer análise de importância de features / SHAP). |
| **Complexidade** | Baixa: $O(N \cdot D)$. | Moderada: $O(M \cdot N \log(N) \cdot \sqrt{D})$. |
| **Volume Necessário** | Baixo a moderado. | Moderado a alto (ideal para os ~50 mil registros do projeto). |
| **Custo Computacional** | Muito baixo. | Moderado em treinamento; rápido em inferência. |

> **🎯 Hipótese Inicial:** A **Técnica B (Random Forest)** é nossa hipótese para a P2 devido à natureza não linear do futebol (força relativa e mando de campo heterogêneo).

---

<!-- SLIDE 4 -->
## Slide 4: Parâmetros e Estratégia de Avaliação

* **Hiperparâmetros Mapeados:**
  * `class_weight='balanced'`: Penaliza erros na classe minoritária (empate ~23% dos casos), forçando o modelo a aprender padrões além da classe majoritária.
  * `max_depth` (15 a 25) e `min_samples_leaf` (>= 5): Controlam a profundidade das árvores, evitando memorização de ruído/zebras históricas (*overfitting*).
  * `n_estimators` (150 a 300 árvores): Garante estabilidade e convergência das probabilidades estimadas.
* **Métrica Adequada:**
  * **$F_1\text{-Macro}$:** Média harmônica balanceada de precisão e recall entre mandante, empate e visitante, impedindo falsa impressão de alta acurácia devida à classe majoritária.
* **Comparação Justa:**
  * Validação cruzada estratificada idêntica em 5 folds com semente fixa (`seed: 42`).
  * Conjunto de teste reservado de 20% (9.989 partidas) estritamente isolado contra vazamento de dados.

---

<!-- SLIDE 5 -->
## Slide 5: Ferramentas, APIs e Governança de Dados

* **Comparação de Ferramentas (Consulta: Nov/2026):**
  * **Ferramenta A — `scikit-learn`:** Gratuita e open source (BSD). Fornece pipelines completos, pré-processadores e algoritmos tabulares. Vantagem: compatibilidade 100% nativa com o código atual do projeto.
  * **Ferramenta B — `xgboost`:** Gratuita e open source (Apache 2.0). Focada em gradient boosting com máxima aceleração. Vantagem: estado da arte para competições e tabelas complexas.
* **Origem e Licença dos Dados:**
  * Base: Kaggle (*International football results from 1872 to 2024*).
  * Licença: **CC0: Public Domain Dedication** — uso acadêmico 100% livre e permitido.
* **Privacidade e LGPD:**
  * Dados 100% públicos e factuais (nomes de países, cidades e torneios).
  * Ausência total de dados pessoais sensíveis ou identificáveis. Execução puramente local.

---

<!-- SLIDE 6 -->
## Slide 6: Proposta de Solução e Recomendação

### Pergunta Central:
> **Qual escolha vocês recomendam para o projeto — e por quê?**

### Resposta e Justificativa:
* **Recomendação:** Evoluir o pipeline adotando **Random Forest Classifier com pesos balanceados (`class_weight='balanced'`) no `scikit-learn`**.
* **Por quê:**
  1. Quebra a barreira linear do modelo anterior, aumentando o F1 da classe de empates.
  2. Mantém a arquitetura limpa de produção já validada (`contrato.py` e `motor_tabular.py`).
  3. Dispensa dependências externas pesadas imediatas, oferecendo o melhor balanço entre ganho preditivo, robustez e custo computacional.
  4. Deixa a porta aberta para sintonia fina com `XGBoost` em etapas posteriores.

---

## Referências
* Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32.
* Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR, 12, 2825–2830.
* Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. ACM KDD.
* Jürisoo, M. (2024). *International football results from 1872 to 2024*. Kaggle (CC0).

