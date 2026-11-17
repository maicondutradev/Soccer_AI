# Oficina de Interfaces para Inteligência Artificial — P2

**Projeto:** Preditor de Resultados de Partidas Internacionais de Futebol  
**Disciplina:** Inteligência Artificial e Machine Learning — Engenharia de Software  
**Campus:** Saquarema — Universidade de Vassouras  
**Professor:** Lucas Hermenegildo do Nascimento  
**Integrantes do Grupo:**
* Maicon Fonseca Dutra
* Gabriel Caniza da Silva
* Gabriel Vidal de Souza

---

## 1. Definição do Usuário e Tarefa Principal

* **Quem utilizará:** Analistas esportivos, torcedores e entusiastas de dados que desejam avaliar o equilíbrio probabilístico de confrontos internacionais entre seleções.
* **Problema resolvido em uma única utilização:** Estimar as probabilidades relativas de vitória do mandante, empate ou vitória do visitante em uma partida específica, considerando o histórico de dados e o fator de mando de campo.
* **Declaração do fluxo principal:**
  > "Nossa aplicação ajuda **analistas e torcedores de futebol** a **estimar a probabilidade de desfecho de partidas internacionais**, utilizando **seleção mandante, seleção visitante, tipo de torneio e condição de campo neutro** para apresentar **a classe prevista e a distribuição percentual entre vitória mandante, empate e vitória visitante**. Depois, o usuário poderá **consultar o retrospecto histórico direto, simular inversão de mando de campo, alternar campo neutro ou testar novos confrontos**."

---

## 2. Decisões de Interface e Arquitetura

O sistema disponibiliza duas opções integradas ao mesmo modelo treinado (`artefatos_v1/modelo_inicial.joblib`):
1. **Interface Web Moderna (Navegador - `app_web.py`):** Desenvolvida em Flask com frontend responsivo em HTML5 e Tailwind CSS, permitindo execução rápida no navegador local (`http://127.0.0.1:5000`).
2. **Interface Desktop (`app_p2.py`):** Desenvolvida em Tkinter com tema escuro e animação incremental de diagnóstico.

### Estrutura das 4 Áreas da Tela:
1. **Identificação:** Título claro do projeto, identificação do modelo em operação (Random Forest treinado sobre 1872-2024) e escopo acadêmico da P2.
2. **Entrada:** Formulário estruturado com dropdowns alimentados pela base oficial de seleções e competições, botão de inversão rápida de mando e seleção de campo neutro.
3. **Ação:** Botão de ação direta (*"Prever Resultado com IA"*) com bloqueio preventivo de múltiplos envios durante o processamento.
4. **Resultado e Próxima Ação:** Cartão de resultado com barras de probabilidade proporcional, retrospecto histórico detalhado dos confrontos diretos, fatores avaliados e botões de reanálise imediata.

---

## 3. Os 6 Estados de Interface Implementados

| Estado | O que é apresentado | Ação / Comportamento do Sistema |
|---|---|---|
| **1. Inicial** | Cards com instruções explicativas e orientações de uso passo a passo. | Apresenta botão de carregamento rápido de partidas clássicas para demonstração. |
| **2. Entrada Inválida** | Alerta visual em destaque com lista detalhada dos campos incorretos. | Impede envio ao backend caso as seleções sejam iguais (ex.: Brasil x Brasil) ou vazias, preservando o restante dos campos. |
| **3. Processando** | Indicador de carregamento animado com mensagem informativa. | Desabilita o botão de inferência para impedir requisições duplicadas. |
| **4. Sucesso** | Classe prevista (Vitória Mandante / Empate / Vitória Visitante), probabilidades relativas em barra visual e histórico de confrontos diretos. | Permite reverter mando de campo ou testar novo confronto com 1 clique. |
| **5. Inconclusivo** | Alerta destacado de incerteza estatística quando a diferença entre as maiores classes for inferior a 4% e não houver histórico de confrontos diretos. | Informa a limitação ao usuário com honestidade em vez de forçar um veredito artificial. |
| **6. Falha Técnica** | Alerta de erro de comunicação ou de inferência com mensagem compreensível. | Disponibiliza botão de "Tentar Novamente" mantendo os dados preenchidos. |

---

## 4. Matriz de Casos de Teste Executados

| ID do Teste | Cenário / Entrada | Comportamento Esperado | Resultado Observado | Status |
|---|---|---|---|---|
| `TC-01` | Brasil (M) vs Argentina (V), Copa América, Campo Não Neutro | Prever vitória do mandante ou distribuição consistente com histórico | Status 200: Vitória de Brazil (56.9%), Empate (23.2%), Argentina (20.0%) | Aprovado |
| `TC-02` | Brasil (M) vs Brasil (V) | Recusa imediata com mensagem informativa ao usuário | Status 422: "A seleção mandante e visitante não podem ser a mesma" | Aprovado |
| `TC-03` | França (M) vs Alemanha (V), Amistoso, Campo Neutro Sim | Fator neutro considerado e probabilidades calculadas | Status 200: Distribuição equilibrada com remoção do fator casa | Aprovado |
| `TC-04` | Clique duplo ou repetido no botão de prever | Botão desabilitado durante o cálculo assíncrono | Desabilitação imediata com cursor bloqueado, sem duplicidade | Aprovado |
| `TC-05` | Seleções sem histórico prévio e probabilidades empatadas | Notificação de limitação / Inconclusivo | Badge de alerta de alta incerteza e ausência de histórico direto | Aprovado |
| `TC-06` | Inversão de mando pós-resultado | Recálculo imediato atualizando mandante e visitante | Mando invertido e atualização visual das probabilidades | Aprovado |

---

## 5. Interpretação Científica, Modelo v2.0 e Limitações

### Evolução Técnica da IA (Modelo v2.0):
O grupo implementou a proposta tecnológica com engenharia de atributos:
1. **Rating Elo Dinâmico (1872–2024):** Calculado cronologicamente partida a partida (sem vazamento de dados), gerando a força relativa atualizada de cada seleção (ex.: Espanha 2.135, Argentina 2.099, França 2.031, Brasil 1.974).
2. **Momentum / Forma Recente:** Aproveitamento e sequência de resultados dos últimos 5 jogos anteriores à partida.
3. **Random Forest Classifier (Ensemble de 200 árvores com `class_weight='balanced'`):**
   * **$F_1$-Macro Teste:** Saltou de **$0,4201$** (Baseline) para **$0,5039$** (**$+20\%$ de ganho relativo**).
   * **Acurácia Teste:** **$55,14\%$** em problema multiclasse desbalanceado com 3 desfechos.

Em conformidade com a Seção 6 e os Slides 4 e 8 do material da disciplina:
* **Não inventar confiança:** O percentual exibido representa a distribuição de probabilidade a priori aprendida pelo modelo multiclasse sob o padrão histórico de seleções, e não uma "certeza matemática" de acerto futuro.
* **Separar fatos históricos de predição:** O retrospecto de confrontos diretos (ex.: vitórias e empates reais entre 1872 e 2024) é apresentado em bloco separado do diagnóstico do modelo de IA.
* **Limitações declaradas:** O futebol internacional possui alta variabilidade intrínseca (cartões vermelhos, condições climáticas, ausência de jogadores titulares e imprevisibilidade esportiva não capturadas em tabelas simples). A acurácia global de 55,14% e F1 de 0,5039 superam com ampla folga uma escolha aleatória uniforme (33,3%), mantendo total honestidade científica perante a banca avaliadora.
