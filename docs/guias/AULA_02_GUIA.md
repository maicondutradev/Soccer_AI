# Aula 2 — Preparar a IA do próprio projeto

**Professor Lucas Hermenegildo do Nascimento • IA e Machine Learning • Projeto P2**

## Objetivo
Implementar a primeira versão executável da IA de cada grupo, mantendo seu problema, dados e tecnologia. O conteúdo é neutro: nenhuma área, dataset, rótulo ou quantidade de entradas é imposta.

## Escolher o caminho do seu projeto
| Situação | Caminho neste pacote | O que o grupo configura/implementa |
|---|---|---|
| Entradas tabulares e resposta conhecida para aprender | `modo: tabular` | CSV, entradas numéricas/categóricas, alvo, tarefa, métrica e divisão |
| IA já existente, geração, busca, recomendação, visão ou outra arquitetura | `modo: adaptador` | schema, módulo real de inferência, preparação própria e avaliação adequada |
Não troque seu problema para caber no exemplo. O adaptador padroniza a ligação, não implementa automaticamente todos esses algoritmos. Projetos não tabulares mantêm sua IA e completam `motor_do_grupo.py` com a chamada real. Os placeholders não são dados nem resultados prontos.

## Preparação do ambiente
1. Extraia o pacote inteiro e abra a pasta no VS Code. Python 3.10+.
2. Crie ambiente: `python -m venv .venv`. No Windows, pode usar `py`.
3. Instale: `.venv\Scripts\python -m pip install -r requirements.txt` (Windows), ou `.venv/bin/python -m pip install -r requirements.txt` (Linux/macOS).
4. Com ambiente ativo, use os comandos abaixo; sem ativar, substitua `python` pelo executável completo da `.venv`.
5. Tabular: preencha config.projeto.json e coloque seu CSV, com suas próprias colunas. Adaptador: preencha config.adaptador.json, schema e motor_do_grupo.py. Não coloque senhas/chaves no material; use variáveis de ambiente no projeto.
6. Preencha os casos com entradas e expectativas reais. O arquivo casos_do_grupo.json é apenas estrutura para substituir.

## Numeração e tempo
A Aula 1 foi a introdução/conexão usando um dataset como exemplo. A sequência neutra começa na Aula 2 (13/10/2026). Etapas seguintes seguem o calendário confirmado pelo professor; datas/pesos da prova não são redefinidos.
Dinâmica sugerida para 100 min: 10 retomada, 15 explicação, 45 implementação, 20 avaliação/discussão, 10 registro/entrega.


## Executar esta aula
```bash
python aula_02.py --config config.projeto.json
```
Para a arquitetura própria: `python aula_02.py --config config.adaptador.json`.

## Roteiro de implementação no projeto
### Passo 1 — Continuidade com projetos diferentes
Implementar a primeira versão executável da IA de cada grupo, mantendo seu problema, dados e tecnologia.

**No código:** `configurar`.

**Ação do grupo:** Definir o que a IA recebe, o que devolve e em qual parte do produto ela será usada.

**Evidência:** registre entrada/configuração, versão, resultado observado e uma interpretação. Não basta mostrar “executou”.

### Passo 2 — Problema, tarefa e critério de sucesso
Distinguir classificar, prever valor, recomendar, buscar informação ou gerar conteúdo. O critério deve refletir a tarefa.

**No código:** `config.projeto.json / config.adaptador.json`.

**Ação do grupo:** Escolher modo tabular para ML supervisionado ou adaptador para integrar a IA já existente; não trocar o tema para caber no exemplo.

**Evidência:** registre entrada/configuração, versão, resultado observado e uma interpretação. Não basta mostrar “executou”.

### Passo 3 — Contrato de entrada e saída
Definir campos, tipos, unidades, obrigatoriedade e estrutura de resposta.

**No código:** `validar_payload`.

**Ação do grupo:** Documentar o contrato com nomes reais do projeto; nenhuma quantidade de campos é imposta.

**Evidência:** registre entrada/configuração, versão, resultado observado e uma interpretação. Não basta mostrar “executou”.

### Passo 4 — Dados ou fonte de conhecimento
Auditar origem, disponibilidade, ausência, repetição e autorização. Projetos sem alvo podem preparar base de conhecimento, índice ou prompt.

**No código:** `carregar_dados ou motor_do_grupo.treinar`.

**Ação do grupo:** Mapear a preparação específica; chamar uma API pronta não é treinar os pesos do modelo.

**Evidência:** registre entrada/configuração, versão, resultado observado e uma interpretação. Não basta mostrar “executou”.

### Passo 5 — Configurar a preparação
Numéricos: imputação/escala; categorias: imputação/codificação. Para outras IAs, preparar a entrada na implementação do grupo.

**No código:** `construir_pipeline ou motor_do_grupo.treinar`.

**Ação do grupo:** Evitar usar informação do futuro, do alvo ou de casos finais na preparação.

**Evidência:** registre entrada/configuração, versão, resultado observado e uma interpretação. Não basta mostrar “executou”.

### Passo 6 — Separar desenvolvimento e avaliação final
Separar por amostra, grupo ou tempo quando cabível; em outras IAs, reservar casos inéditos.

**No código:** `preparar_experimento / casos_do_grupo.json`.

**Ação do grupo:** Justificar estratégia pela unidade de observação. Não sortear mensagens do mesmo usuário entre treino/teste sem análise.

**Evidência:** registre entrada/configuração, versão, resultado observado e uma interpretação. Não basta mostrar “executou”.

### Passo 7 — Executar a primeira versão
Treinar baseline tabular ou conectar a preparação/inferência da IA do grupo.

**No código:** `treinar_inicial / chamar_adaptador`.

**Ação do grupo:** Executar e registrar resultado real, sem prometer que funciona para qualquer tarefa sem adaptar.

**Evidência:** registre entrada/configuração, versão, resultado observado e uma interpretação. Não basta mostrar “executou”.

### Passo 8 — Entrega da Aula 2
Contrato, configuração, origem dos dados, primeira execução e versão identificada.

**No código:** `aula_02.py`.

**Ação do grupo:** Cada integrante explica uma parte. Próxima etapa: avaliar qualidade com evidência.

**Evidência:** registre entrada/configuração, versão, resultado observado e uma interpretação. Não basta mostrar “executou”.

## Código explicado parte por parte
Leia os blocos do caminho escolhido. As funções tabulares não precisam ser usadas em um projeto de outra arquitetura.

### Bloco 1 — `contrato.py` → `configurar`
**O que implementa:** Ler identidade e modo do projeto.

**Entrada e saída:** Recebe caminho JSON e devolve config validada.

```python
def configurar(caminho='config.projeto.json'):
    config = ler_json(caminho)
    if config.get('modo') not in ('tabular', 'adaptador'):
        raise ValueError('modo deve ser tabular ou adaptador.')
    if not config.get('nome') or 'PREENCHER' in config['nome']:
        raise ValueError('Preencha o nome do seu projeto no JSON.')
    config['_arquivo_config'] = str(Path(caminho).resolve())
    return config
```

1. ler_json lê a configuração UTF-8; nome deve ser preenchido e modo deve ser tabular/adaptador.
2. _arquivo_config guarda o caminho para identificação; não é um campo do dataset.
3. A configuração orienta a implementação; não transforma qualquer tarefa em classificação.

**Pergunta de checagem:** qual problema esse bloco evita no seu projeto e que evidência prova que funcionou?

### Bloco 2 — `contrato.py` → `validar_payload`
**O que implementa:** Validar o contrato de entrada.

**Entrada e saída:** Recebe um dicionário de valores e o schema definido pelo grupo.

```python
def validar_payload(entrada, schema):
    """Valida o contrato sem impor um tema ou uma quantidade fixa de campos."""
    if not isinstance(entrada, dict):
        raise ValueError('A entrada deve ser um objeto/dicionário.')
    inesperados = set(entrada) - set(schema)
    if inesperados:
        raise ValueError(f'Campos não previstos: {sorted(inesperados)}')
    for nome, regra in schema.items():
        valor = entrada.get(nome)
        if valor is None:
            if regra.get('obrigatorio', True):
                raise ValueError(f'{nome}: campo obrigatório.')
            continue
        tipo = regra['tipo']
        if tipo == 'numero':
            if isinstance(valor, bool) or not isinstance(valor, (int, float)) or not math.isfinite(valor):
                raise ValueError(f'{nome}: número finito esperado.')
            if 'min' in regra and valor < regra['min']:
                raise ValueError(f'{nome}: abaixo do mínimo.')
            if 'max' in regra and valor > regra['max']:
                raise ValueError(f'{nome}: acima do máximo.')
        elif tipo == 'texto':
            if not isinstance(valor, str) or not valor.strip():
                raise ValueError(f'{nome}: texto não vazio esperado.')
        elif tipo == 'lista':
            if not isinstance(valor, list):
                raise ValueError(f'{nome}: lista esperada.')
        else:
            raise ValueError(f'Tipo de campo não implementado: {tipo}. Adapte o contrato.')
        if 'opcoes' in regra and valor not in regra['opcoes']:
            raise ValueError(f'{nome}: valor não permitido.')
    return entrada
```

1. Campos inesperados são rejeitados; ausentes são permitidos apenas se obrigatorio=false.
2. Número deve ser finito e não booleano; limites min/max são opcionais.
3. Texto precisa ser não vazio; lista aceita uma lista Python, cuja estrutura interna deve ser validada pelo grupo se necessário.
4. opcoes restringe valores quando declarado. Tipo não implementado exige adaptar contrato, não ignorar a validação.

**Pergunta de checagem:** qual problema esse bloco evita no seu projeto e que evidência prova que funcionou?

### Bloco 3 — `motor_tabular.py` → `checar_config`
**O que implementa:** Conferir tarefa e configuração tabular.

**Entrada e saída:** Recebe config; falha cedo se configuração é incompatível.

```python
def checar_config(config):
    if config['tarefa'] not in ('classificacao', 'regressao'):
        raise ValueError('Defina tarefa como classificacao ou regressao.')
    campos = config['numericas'] + config['categoricas']
    if not campos or len(campos) != len(set(campos)) or config['alvo'] in campos:
        raise ValueError('Defina entradas únicas e mantenha o alvo fora delas.')
    if config['divisao'] not in ('aleatoria', 'grupo', 'tempo'):
        raise ValueError('divisao deve ser aleatoria, grupo ou tempo.')
    permitidas = {'classificacao': ['f1_macro', 'balanced_accuracy'],
                 'regressao': ['neg_mean_absolute_error', 'neg_root_mean_squared_error']}
    if config['metrica'] not in permitidas[config['tarefa']]:
        raise ValueError('Escolha métrica compatível com a tarefa.')
    if not 0.05 <= config['fracao_teste'] <= 0.4:
        raise ValueError('fracao_teste deve ficar entre 0.05 e 0.4 nesta base.')
```

1. Entradas devem ser únicas e não conter o alvo; campos de identificação não devem ser entradas por padrão.
2. Escolher classificação/regressão e métrica adequada. Scores de erro do scikit-learn são negativos na busca.
3. Divisão aleatória/grupo/tempo depende da origem dos registros, não da preferência por uma métrica maior.

**Pergunta de checagem:** qual problema esse bloco evita no seu projeto e que evidência prova que funcionou?

### Bloco 4 — `motor_tabular.py` → `carregar_dados`
**O que implementa:** Ler dados e produzir auditoria.

**Entrada e saída:** Recebe config; devolve X, y e auxiliar de grupo/tempo quando existe.

```python
def carregar_dados(config):
    checar_config(config)
    df = pd.read_csv(config['csv'], sep=config['separador'], decimal=config['decimal'],
                     encoding='utf-8-sig')
    campos = config['numericas'] + config['categoricas']
    auxiliar = config.get('coluna_grupo') if config['divisao'] == 'grupo' else (
        config.get('coluna_tempo') if config['divisao'] == 'tempo' else None)
    if config['divisao'] != 'aleatoria' and not auxiliar:
        raise ValueError('Informe coluna_grupo ou coluna_tempo para a divisão escolhida.')
    obrigatorias = campos + [config['alvo']] + ([auxiliar] if auxiliar else [])
    faltam = set(obrigatorias) - set(df.columns)
    if faltam:
        raise ValueError(f'Colunas ausentes: {sorted(faltam)}')
    auditoria = {'originais': len(df), 'duplicatas_completas': int(df.duplicated().sum()),
                 'ausencias_originais': df[obrigatorias].isna().sum().astype(int).to_dict(),
                 'observacao': 'Duplicatas não removidas automaticamente; investigar unidade de observação.'}
    for coluna in config['numericas']:
        df[coluna] = pd.to_numeric(df[coluna], errors='coerce').replace([np.inf, -np.inf], np.nan)
        limites = config.get('limites', {}).get(coluna, {})
        if 'min' in limites:
            df.loc[df[coluna] < limites['min'], coluna] = np.nan
        if 'max' in limites:
            df.loc[df[coluna] > limites['max'], coluna] = np.nan
    for coluna in config['categoricas']:
        df[coluna] = df[coluna].map(lambda x: np.nan if pd.isna(x) or not str(x).strip() else str(x).strip())
    if config['tarefa'] == 'regressao':
        df[config['alvo']] = pd.to_numeric(df[config['alvo']], errors='coerce').replace([np.inf, -np.inf], np.nan)
    else:
        df[config['alvo']] = df[config['alvo']].map(lambda x: np.nan if pd.isna(x) or not str(x).strip() else str(x).strip())
    validos = df[config['alvo']].notna()
    if auxiliar:
        if config['divisao'] == 'tempo':
            df[auxiliar] = pd.to_datetime(df[auxiliar], errors='coerce', utc=True)
        validos &= df[auxiliar].notna()
    auditoria['removidas_sem_alvo_ou_auxiliar'] = int((~validos).sum())
    df = df.loc[validos].copy()
    if len(df) < 30 or df[campos].isna().all().any():
        raise ValueError('Necessário ao menos 30 registros válidos e nenhuma entrada totalmente ausente.')
    if config['tarefa'] == 'classificacao' and df[config['alvo']].nunique() < 2:
        raise ValueError('Classificação exige pelo menos duas classes conhecidas.')
    if config['divisao'] == 'tempo':
        df = df.sort_values(auxiliar, kind='stable')
    df = df.reset_index(drop=True)
    auditoria['validas'] = len(df)
    auditoria['ausencias_entradas_apos_conversao'] = df[campos].isna().sum().astype(int).to_dict()
    salvar_json(Path(config['saida']) / 'auditoria.json', auditoria)
    return df[campos], df[config['alvo']], df[auxiliar] if auxiliar else None
```

1. read_csv respeita separador/decimal; a verificação de colunas identifica cabeçalhos incorretos.
2. Numerais inválidos/infinito/fora de domínio viram ausência. Campos categóricos são convertidos para texto consistente.
3. Alvo de regressão precisa ser numérico; alvo de classificação preserva os rótulos do projeto, sem forçar Sim/Nao.
4. Remove apenas registros sem alvo/auxiliar válido. Duplicatas são contadas, não eliminadas automaticamente: duas transações iguais podem ser legítimas.
5. Ordena por tempo quando necessário e devolve somente entradas selecionadas. Uma entrada totalmente vazia ou amostra muito pequena exige revisão.

**Pergunta de checagem:** qual problema esse bloco evita no seu projeto e que evidência prova que funcionou?

### Bloco 5 — `motor_tabular.py` → `preparar_experimento`
**O que implementa:** Preservar versão e separar dados.

**Entrada e saída:** Recebe config; devolve X/y/auxiliar/índices de treino e teste.

```python
def preparar_experimento(config):
    saida = Path(config['saida']); saida.mkdir(parents=True, exist_ok=True)
    arquivo = saida / 'manifesto.json'
    atual = assinatura(config)
    if arquivo.exists() and ler_json(arquivo)['assinatura'] != atual:
        raise ValueError('Base, configuração ou código mudou. Preserve resultados e use outra pasta de versão.')
    if not arquivo.exists():
        salvar_json(arquivo, {'assinatura': atual, 'python': platform.python_version(),
                             'sklearn': sklearn.__version__, 'pandas': pd.__version__})
    X, y, auxiliar = carregar_dados(config)
    indices = np.arange(len(X))
    if config['divisao'] == 'grupo':
        treino, teste = next(GroupShuffleSplit(n_splits=1, test_size=config['fracao_teste'],
                                               random_state=config['seed']).split(X, y, groups=auxiliar))
    elif config['divisao'] == 'tempo':
        corte = int(len(X) * (1 - config['fracao_teste']))
        # Não permite o mesmo instante temporal nos dois conjuntos.
        ponto = auxiliar.iloc[corte]
        treino = indices[(auxiliar < ponto).to_numpy()]
        teste = indices[(auxiliar >= ponto).to_numpy()]
    else:
        treino, teste = train_test_split(indices, test_size=config['fracao_teste'],
                                         random_state=config['seed'],
                                         stratify=y if config['tarefa'] == 'classificacao' else None)
    if len(treino) < 15 or len(teste) < 2:
        raise ValueError('A divisão gerou amostras insuficientes; reveja tamanho, tempo ou grupos.')
    if X.iloc[treino].isna().all().any():
        raise ValueError('Uma entrada ficou totalmente vazia no treino.')
    if config['tarefa'] == 'classificacao' and y.iloc[treino].nunique() < 2:
        raise ValueError('Treino precisa de pelo menos duas classes.')
    salvar_json(saida / 'divisao.json', {'treino': treino.tolist(), 'teste': teste.tolist(),
                                       'estrategia': config['divisao']})
    return X, y, auxiliar, treino, teste
```

1. Hash verifica base/config/código; mudança pede outra pasta de resultados.
2. Divisão aleatória estratifica classificação. Por grupo, entidades não cruzam treino/teste.
3. Por tempo, treino usa instantes anteriores ao primeiro instante do teste; mesmo timestamp não cruza a fronteira.
4. Registra os índices e rejeita conjuntos mínimos insuficientes. A separação não elimina vazamento de campos que revelam a resposta.

**Pergunta de checagem:** qual problema esse bloco evita no seu projeto e que evidência prova que funcionou?

### Bloco 6 — `motor_tabular.py` → `construir_pipeline`
**O que implementa:** Pré-processar tipos diferentes sem vazamento.

**Entrada e saída:** Recebe config e estimador; devolve Pipeline ainda não ajustada.

```python
def construir_pipeline(config, estimador):
    transformadores = []
    if config['numericas']:
        numerico = Pipeline([('ausencias', SimpleImputer(strategy='median')),
                              ('escala', StandardScaler())])
        transformadores.append(('numerico', numerico, config['numericas']))
    if config['categoricas']:
        categorico = Pipeline([('ausencias', SimpleImputer(strategy='most_frequent')),
                                ('codificacao', OneHotEncoder(handle_unknown='ignore'))])
        transformadores.append(('categorico', categorico, config['categoricas']))
    preparo = ColumnTransformer(transformadores, remainder='drop')
    return Pipeline([('preparo', preparo), ('modelo', estimador)])
```

1. Pipeline numérica: mediana e StandardScaler. Pipeline categórica: moda e OneHotEncoder.
2. handle_unknown=ignore mantém formato quando surgir categoria nova; isso não garante boa previsão para categoria nunca aprendida.
3. ColumnTransformer aplica transformação apenas às colunas selecionadas e descarta demais.
4. fit dentro do fold aprende estatísticas só no treino; predict repete essas regras nas novas entradas.

**Pergunta de checagem:** qual problema esse bloco evita no seu projeto e que evidência prova que funcionou?

### Bloco 7 — `motor_tabular.py` → `treinar_inicial`
**O que implementa:** Construir a primeira versão supervisionada.

**Entrada e saída:** Recebe config; salva modelo_inicial e diagnóstico de treino.

```python
def treinar_inicial(config):
    X, y, _, tr, _ = preparar_experimento(config)
    modelo = construir_pipeline(config, candidatos(config)[0][1])
    modelo.fit(X.iloc[tr], y.iloc[tr])
    joblib.dump(modelo, Path(config['saida']) / 'modelo_inicial.joblib')
    salvar_json(Path(config['saida']) / 'treino_inicial.json', {
        'metricas_treino': medir(config, y.iloc[tr], modelo.predict(X.iloc[tr])),
        'limite': 'Treino é diagnóstico; avaliação de generalização vem nas próximas aulas.'})
```

1. preparar_experimento cria a separação; a etapa não calcula previsão no teste final.
2. Escolhe candidato linear compatível com a tarefa e chama fit no treino.
3. Grava a pipeline completa e métricas de treino; esses números não comprovam generalização.

**Pergunta de checagem:** qual problema esse bloco evita no seu projeto e que evidência prova que funcionou?

### Preparação da arquitetura própria
```python
"""Ponto de ligação à IA existente do grupo. Preencha sem trocar seu tema.

Pode chamar seu classificador, recomendador, busca, chatbot, modelo de imagem
ou serviço de IA. O material não supõe que todas essas soluções sejam iguais.
Não coloque chaves de API no código nem no JSON; use variáveis de ambiente.
"""

def inferir(entrada):
    """Receba dict, invoque sua IA e devolva dict com 'resultado'."""
    # Substitua pela chamada real já usada no projeto:
    # resposta = minha_ia.executar(entrada)
    # return {'resultado': resposta, 'versao': 'identificador_da_versao'}
    raise NotImplementedError('Conecte aqui a função real de inferência da IA do grupo.')


def treinar(config):
    """Opcional: implemente só se seu projeto realmente tem treinamento local.

Para IA que usa modelo pronto/API, esta etapa pode preparar conhecimento,
índice, prompts e parâmetros. Documente a ação como preparação, não finja
que ela treinou os pesos do modelo.
"""
    raise NotImplementedError('Implemente a preparação específica ou execute-a no projeto e documente.')

```
As exceções indicam tarefas pendentes. Substitua pela implementação real, sem esconder a ausência de ligação. Em projetos que usam modelo pronto, documente preparação/contexto/índice como tal; não afirme que treinou seus pesos.

## Atividade de aprofundamento
Escolha uma função da aula e explique: responsabilidade, entrada, processamento, saída, falha esperada e onde ela se liga ao software. Execute um caso comum e um caso difícil do seu projeto. Compare o observado com o esperado antes de afirmar que funcionou.

## Registro e defesa individual
No relatório, preencher objetivo, versão, mudança, comando, artefatos, métrica/oráculo, interpretação e próxima ação. Cada integrante registra contribuição e explica uma decisão. Avalie adequação ao problema, não “quem teve maior acurácia” entre tarefas diferentes.

## Diagnóstico de problemas
| Sintoma | Verificação | Ação |
|---|---|---|
| PREENCHER na configuração | campos ainda genéricos | substituir por nomes reais do projeto |
| NotImplementedError | IA do grupo não conectada | implementar preparação/inferir no módulo real |
| CSV/coluna ausente | caminho, cabeçalho e separador | conferir JSON e origem dos dados |
| Tarefa/métrica incompatível | classificação versus regressão | escolher métrica da tarefa |
| Um fold tem só uma classe | tempo/grupo/classe rara | rever protocolo ou obter dados, sem vazamento futuro |
| Configuração/base mudou | assinatura e versão | preservar resultados e criar pasta de versão nova |
| Caso “passa” com saída errada | oráculo fraco | fortalecer expectativa e complementar revisão humana |
| Modelo/API indisponível | versão, arquivos, credencial e serviço | tratar indisponibilidade na arquitetura real |
| Caso final já executado | conjunto usado na escolha | preservar avaliação; nova melhoria precisa de nova validação final |

## Limites e adaptação
A implementação tabular cobre classificação e regressão com entradas numéricas/categóricas. Não implementa classificação de imagem, geração de texto, recomendação ou aprendizado não supervisionado como se fossem tarefas tabulares idênticas. Nessas arquiteturas, usar adaptador para a IA real e avaliador apropriado. Em saídas abertas, os testes automáticos simples não provam qualidade semântica. Dados pessoais, fontes autorizadas e segredos devem ser tratados conforme o projeto e sua instituição.

## Referências técnicas
https://scikit-learn.org/stable/common_pitfalls.html
https://scikit-learn.org/stable/modules/cross_validation.html
https://scikit-learn.org/stable/modules/compose.html
https://docs.streamlit.io/develop/api-reference
