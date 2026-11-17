# Projeto P2 — Inteligência Artificial e Machine Learning
**Previsão de Resultados de Partidas Internacionais de Futebol**  
**Professor:** Lucas Nascimento  
**Integrantes:**
- Maicon Fonseca Dutra
- Gabriel Caniza da Silva
- Gabriel Vidal de Souza

---

## 📁 Estrutura do Projeto

O repositório está organizado de forma modular para separar claramente código de execução, dados brutos, artefatos gerados e documentação:

```text
IA_P2/
├── artefatos_v1/          # Modelos serializados, métricas e manifestos gerados
├── data/                  # Base de dados tabular (partidas_futebol.csv)
├── docs/                  # Documentação completa, relatórios e apresentações
│   ├── entregas/          # Relatórios e apresentações oficiais das etapas
│   │   ├── ENTREGA_PROJETO_IA_P2.md    # Relatório das Aulas 02 e 03
│   │   ├── PROPOSTA_MELHORIA_IA_P2.md  # Documento da Proposta de Melhoria da IA
│   │   └── SLIDES_MELHORIA_IA_P2.md    # Formato de apresentação/slides
│   ├── guias/             # Guias de laboratório e passos das aulas anteriores
│   │   ├── AULA_02_GUIA.md
│   │   └── AULA_03_GUIA.md
│   └── historico/         # Arquivos legados, notebooks e entregas anteriores
├── app_web.py             # Interface Web moderna para uso no navegador (Flask + Tailwind CSS)
├── app_p2.py              # Interface gráfica interativa desktop (Tkinter Dark Mode)
├── aula_02.py             # Script de execução da Aula 02 (baseline, auditoria e manifesto)
├── aula_03.py             # Script de execução da Aula 03 (validação cruzada e testes)
├── contrato.py            # Definição e validação do schema/payload de entrada
├── etapas.py              # Orquestrador das rotinas e CLI das aulas
├── motor_tabular.py       # Pipelines de pré-processamento, treino e avaliação
├── config.projeto.json    # Configurações centrais do experimento
├── casos_do_grupo.json    # Casos de teste funcionais e de recusa
└── requirements.txt       # Dependências Python do ambiente
```

---

## 🚀 Como Executar o Projeto

Certifique-se de estar com o ambiente virtual `.venv` ativo no terminal:

```powershell
# 1. Executar a Interface Web Moderna no Navegador (Recomendada):
.venv\Scripts\python app_web.py
# Abra no navegador: http://127.0.0.1:5000

# 2. Executar a Interface Desktop em Tkinter:
.venv\Scripts\python app_p2.py

# 3. Executar a Aula 02 (Auditoria, manifesto e treino do baseline inicial):
.venv\Scripts\python aula_02.py --config config.projeto.json

# 4. Executar a Aula 03 (Validação cruzada em 5 folds, comparação com baseline e testes):
.venv\Scripts\python aula_03.py --config config.projeto.json
```

---

## 📄 Documentos de Entrega

* [Relatório da Oficina de Interfaces](docs/entregas/ENTREGA_OFICINA_INTERFACES.md): Registro das decisões de interface, os 6 estados, matriz de testes e limitações metodológicas.
* [Relatório Completo — Proposta de Melhoria de IA](docs/entregas/PROPOSTA_MELHORIA_IA_P2.md): Análise detalhada, comparação técnica de modelos, parâmetros, governança de dados e recomendação justificada para a P2.
* [Apresentação em Slides — Como Melhorar Nossa IA?](docs/entregas/SLIDES_MELHORIA_IA_P2.md): Roteiro de slides estruturado para apresentação oral ou exportação para PDF.
* [Relatório das Aulas 02 e 03](docs/entregas/ENTREGA_PROJETO_IA_P2.md): Registro da entrega inicial e dos resultados quantitativos de validação cruzada.
