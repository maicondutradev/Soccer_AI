# Proposta Tecnológica de Melhoria de IA — Projeto P2
**Disciplina:** Inteligência Artificial e Machine Learning  
**Professor:** Lucas Nascimento  
**Curso:** Engenharia de Software / Ciência da Computação  
**Data da Consulta / Elaboração:** Novembro de 2026  

---

## 1. Identificação da Equipe e do Projeto

* **Integrantes do Grupo:**
  1. Maicon Fonseca Dutra
  2. Gabriel Caniza da Silva
  3. Gabriel Vidal de Souza
* **Nome do Projeto:** `Previsao_Partidas_Futebol_P2`
* **Repositório / Modo:** `modo: tabular`
* **Tipo de Tarefa de IA:** Classificação Multiclasse (`classificacao`)
* **Classes Alvo:** `vitoria_mandante`, `empate`, `vitoria_visitante`
* **Métrica Principal:** $F_1\text{-Macro}$

---

## 2. Descrição Clara do Problema e Dados (Slide 2)

### 2.1 Resumo Executivo do Projeto (Ação do Grupo: 3 Frases)
> **1. Problema:** O projeto visa prever o desfecho de partidas internacionais oficiais de futebol entre seleções (`vitoria_mandante`, `empate` ou `vitoria_visitante`) utilizando exclusivamente informações contextuais disponíveis previamente ao início do jogo.  
> **2. Dados Disponíveis:** A base histórica reúne 49.945 confrontos com variáveis pré-jogo consolidadas: seleção mandante, seleção visitante, tipo de torneio e indicador binário de campo neutro.  
> **3. Resultado Esperado:** Um modelo preditivo multiclasse capaz de superar significativamente o classificador de referência e a regressão logística linear inicial, atingindo melhor discriminação da classe empate e elevando o $F_1\text{-Macro}$ geral em validação cruzada estratificada.

### 2.2 Perguntas-Guia do Problema
* **Qual é o problema central do projeto?**  
  A incerteza intrínseca nos resultados esportivos e a dificuldade de prever desfechos que envolvem desbalanceamento de classes (vitória do mandante ocorre em ~49% das partidas, empate em ~23% e vitória do visitante em ~28%).
* **Quem é o público ou usuário afetado?**  
  Analistas táticos de futebol, federações e comissões técnicas (para planejamento de cenários), jornalistas esportivos e plataformas de estatísticas e entretenimento.
* **Quais dados de entrada estão disponíveis?**  
  Quatro variáveis categóricas padronizadas: `mandante` (texto), `visitante` (texto), `torneio` (texto) e `campo_neutro` (binário: "Sim" ou "Nao"). As variáveis de placar (`home_score` e `away_score`) foram estritamente excluídas do conjunto de entrada para prevenir vazamento de dados (*data leakage*).
* **Qual resultado ou saída é esperado?**  
  Um payload JSON estruturado contendo a predição categórica e a versão do modelo:
  ```json
  {
    "resultado": "vitoria_mandante",
    "versao": "2.0.0"
  }
  ```

---

## 3. Comparação de Modelos / Técnicas de IA (Slide 3)

Na etapa inicial (Aula 02 e 03), o grupo utilizou um modelo linear multinomial. Para evoluir a capacidade preditiva na P2, investigamos a transição para métodos baseados em árvores (*ensemble methods*).

| Critério de Comparação | Técnica A: Regressão Logística Multinomial | Técnica B: Random Forest Classifier (Ensemble) |
|---|---|---|
| **Como Funciona** | Modela a probabilidade condicional de cada classe aplicando uma transformação linear nas features codificadas via *One-Hot Encoding*, seguida da função *Softmax* para normalização em probabilidades. | Constrói uma floresta com centenas de árvores de decisão independentes geradas por amostragem com reposição (*bagging / bootstrap*), dividindo cada nó com subconjuntos aleatórios de features para reduzir a variância. |
| **Aplicação no Projeto** | Modelo de referência linear para classificação tabular rápida sobre matriz esparsa de alta cardinalidade. | Modelagem não linear de dados tabulares capaz de aprender interações complexas entre pares específicos de seleções (confrontos diretos, rivalidades) e fatores de campo/torneio. |
| **Vantagem Principal** | **Alta interpretabilidade e baixo custo computacional.** Permite inspecionar diretamente os pesos atribuídos a cada seleção e treina em menos de 10 segundos para 50 mil linhas. | **Captura relações não lineares e interações entre seleções.** Não assume linearidade, é insensível a escala e lida de forma superior com dependências conjuntas sem necessidade de criar termos cruzados manualmente. |
| **Limitação Principal** | **Fronteira de decisão estritamente linear.** Não consegue expressar que uma seleção A perde de B, mas vence C com facilidade, a menos que haja engenharia manual de atributos; baixa capacidade preditiva na classe "empate". | **Maior custo computacional de memória e treino**, além de comportamento "caixa preta", exigindo ferramentas adicionais (como *Feature Importance* ou *SHAP*) para explicar predições pontuais. |
| **Interpretabilidade** | **Alta:** coeficientes diretos por classe e por seleção. | **Média/Baixa:** requer análise agregada de importância de features ou valores SHAP. |
| **Complexidade Algorítmica** | **Baixa:** $O(N \cdot D)$, onde $N$ é o número de amostras e $D$ as dimensões. | **Moderada a Alta:** $O(M \cdot N \log(N) \cdot \sqrt{D})$, onde $M$ é o número de árvores. |
| **Volume de Dados Necessário** | Baixo a moderado; converge mesmo com poucas instâncias. | Moderado a alto; os 49.945 registros disponíveis são ideais para treinar a floresta sem *underfitting*. |
| **Custo Computacional** | Quase desprezível (poucos MB de RAM e CPU básica). | Moderado (requer paralelização de CPU nos folds de validação cruzada). |

### 🎯 Hipótese Inicial do Grupo para a P2:
A **Técnica B (Random Forest Classifier)** é definida como a **hipótese inicial prioritária** para o projeto P2.  
*Justificativa:* Os dados esportivos possuem forte não-linearidade tática (ex.: peso do mando de campo varia drasticamente conforme o país e o torneio). O Random Forest permite capturar essas nuances sem risco excessivo de overfitting graças à média de múltiplas árvores descorrelacionadas.

---

## 4. Parâmetros e Estratégia de Avaliação (Slide 4)

### 4.1 Mapeamento de Parâmetros Relevantes
1. **`class_weight='balanced'` (Tratamento de Desbalanceamento):**
   * *Configuração proposta:* Ativar pesos inversamente proporcionais à frequência das classes.
   * *Efeito esperado:* Na base, empates representam apenas ~23% do total. O modelo padrão tende a negligenciar empates em favor de vitórias mandantes. O parâmetro penaliza erros na classe minoritária, aumentando o *recall* e elevando o $F_1\text{-Macro}$.
2. **`max_depth` e `min_samples_leaf` (Controle de Overfitting):**
   * *Configuração proposta:* Estabelecer `max_depth` entre 15 e 25 e `min_samples_leaf >= 5`.
   * *Efeito esperado:* Evita que as árvores memorizem partidas históricas raras ou atípicas (zebras isoladas), garantindo que a floresta generalize bem para partidas futuras.
3. **`n_estimators` (Número de Estimadores):**
   * *Configuração proposta:* 150 a 300 árvores com `n_jobs=-1` para execução paralela em múltiplos núcleos de CPU.
   * *Efeito esperado:* Estabiliza as estimativas de probabilidade e reduz a variância global da predição.

### 4.2 Métrica de Avaliação Adequada: $F_1\text{-Macro}$
A acurácia tradicional é enganosa em cenários desbalanceados (o baseline dummy de chutar sempre vitória mandante alcança 49% de acurácia com F1 de apenas 0,2192). O **$F_1\text{-Macro}$** calcula a média não ponderada da média harmônica entre precisão e revocação de cada classe individual:
$$F_1\text{-Macro} = \frac{F_{1,\text{mandante}} + F_{1,\text{empate}} + F_{1,\text{visitante}}}{3}$$
Dessa forma, o modelo só obtém uma pontuação alta se for capaz de prever empates e vitórias de visitantes com precisão, e não apenas vitórias caseiras.

### 4.3 Estratégia de Comparação Justa
* **Isolamento de Teste (Sem Vazamento de Dados):** O conjunto de teste (20% dos dados, 9.989 partidas) permanece estritamente guardado e intocado.
* **Validação Cruzada Idêntica:** Ambos os modelos (Técnica A e Técnica B) são treinados e avaliados exatamente sob os mesmos 5 folds estratificados (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`).
* **Semente Aleatória Fixa:** `seed: 42` mantida em todos os experimentos para assegurar reprodutibilidade científica total.

---

## 5. Ferramentas, APIs e Governança dos Dados (Slide 5)

### 5.1 Comparativo de Ferramentas / Bibliotecas de IA

| Aspecto | Ferramenta A: `scikit-learn` | Ferramenta B: `xgboost` (ou `lightgbm`) |
|---|---|---|
| **Função** | Biblioteca abrangente de Machine Learning clássico em Python, oferecendo estimadores, transformadores, validação cruzada e métricas. | Biblioteca de alta performance focada exclusivamente em algoritmos de árvores com *Gradient Boosting*. |
| **Acesso e Limites** | Gratuita e open source (Licença BSD 3-Clause). Execução local sem limites de requisições ou custos de API. | Gratuita e open source (Licença Apache 2.0). Execução local sem limites de uso. |
| **Link Oficial** | [scikit-learn.org](https://scikit-learn.org) | [xgboost.readthedocs.io](https://xgboost.readthedocs.io) |
| **Aplicação no Projeto** | Implementação de pipelines completos (`ColumnTransformer`), pré-processamento One-Hot, `LogisticRegression` e `RandomForestClassifier`. | Treinamento sequencial de árvores impulsionadas por gradiente com regularização nativa $L_1$ e $L_2$. |
| **Vantagem Principal** | **Padronização e compatibilidade direta.** Já integrada ao ecossistema atual do projeto (`motor_tabular.py`, `contrato.py`), sem necessidade de novos compiladores ou dependências pesadas. | **Desempenho preditivo de ponta em dados tabulares.** Execução acelerada em C++ com suporte nativo a matrizes esparsas e poda dinâmica. |
| **Data da Consulta** | 03 de Novembro de 2026. | 03 de Novembro de 2026. |

### 5.2 Governança e Ética dos Dados do Projeto
* **Qual é a origem dos dados?**  
  Dataset público disponibilizado na plataforma Kaggle: *International football results from 1872 to 2024* (mantido por Mart Jürisoo).
* **O uso é permitido para fins acadêmicos?**  
  **Sim.** A base está publicada sob a licença internacional **CC0: Public Domain Dedication**, que autoriza livre cópia, modificação, distribuição e uso acadêmico e comercial sem exigência de royalties ou restrições.
* **Há dados pessoais? Como serão protegidos?**  
  **Não há nenhum dado pessoal ou sensível identificado.** O dataset contém unicamente dados factuais públicos de domínio histórico: datas, seleções nacionais, cidades, países e torneios. Não constam nomes de jogadores, comissões técnicas, árbitros, documentos ou dados de torcedores, garantindo plena conformidade com a LGPD (Lei Geral de Proteção de Dados - Lei nº 13.709/2018).
* **Solução Local vs Nuvem:**  
  O treinamento e a inferência são executados 100% em ambiente local via scripts Python, garantindo privacidade, soberania sobre os dados e custo computacional nulo com nuvem ou APIs pagas.

---

## 6. Proposta de Solução Justificada e Recomendação Final (Slide 6)

### 6.1 Resposta à Pergunta Central
> **Pergunta:** *Qual escolha vocês recomendam para o projeto — e por quê?*

**Recomendação do Grupo:**  
Recomendamos a adoção do **Random Forest Classifier com balanceamento de pesos (`class_weight='balanced'`) integrado via biblioteca `scikit-learn`** como a tecnologia principal de evolução para a P2.

**Por quê? (Justificativa Técnica):**
1. **Superação dos Limites Lineares:** Os experimentos da Aula 03 evidenciaram que a regressão logística atingiu um teto de $F_1\text{-Macro}$ (0,4457), falhando sistematicamente em prever empates e partidas com dinâmicas táticas não lineares. O Random Forest consegue criar partições hierárquicas no espaço amostral, mapeando forças relativas entre mandantes e visitantes sem assumir independência linear.
2. **Robustez ao Desbalanceamento:** O suporte nativo ao hiperparâmetro `class_weight='balanced'` no Scikit-Learn permite contrabalançar diretamente a menor frequência de empates sem a necessidade de técnicas artificiais arriscadas de sobreamostragem (como SMOTE) em dados categóricos de alta cardinalidade.
3. **Compatibilidade e Estabilidade de Engenharia:** O Scikit-Learn já está configurado no ambiente virtual do projeto (`.venv`), respeitando integralmente os contratos de validação (`contrato.py`), pipelines de transformação e geração de artefatos (`artefatos_v1/manifesto.json`). Isso viabiliza uma entrega ágil, robusta e cientificamente comparável.
4. **Caminho para Etapa Final:** Caso o Random Forest confirme a hipótese de ganho significativo de $F_1$, a equipe poderá testar pontualmente o `XGBoost` como segundo passo de otimização na P2 para refinamento marginal de hiperparâmetros.

---

## 7. Referências Bibliográficas

1. **BREIMAN, Leo.** *Random Forests*. Machine Learning, v. 45, n. 1, p. 5–32, 2001. DOI: [10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324).
2. **PEDREGOSA, Fabian et al.** *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research (JMLR), v. 12, p. 2825–2830, 2011. Disponível em: <https://scikit-learn.org>.
3. **CHEN, Tianqi; GUESTRIN, Carlos.** *XGBoost: A Scalable Tree Boosting System*. In: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. ACM, 2016. p. 785–794. Disponível em: <https://xgboost.readthedocs.io>.
4. **JÜRISOO, Mart.** *International football results from 1872 to 2024*. Kaggle Dataset (Public Domain CC0), 2024. Disponível em: <https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017>.
5. **BRASIL.** *Lei Geral de Proteção de Dados Pessoais (LGPD)*, Lei nº 13.709, de 14 de agosto de 2018. Presidência da República, Casa Civil, 2018.

