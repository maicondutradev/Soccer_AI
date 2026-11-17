# Aula 3 — Avaliar a qualidade da IA no projeto

**Professor Lucas Hermenegildo do Nascimento • IA e Machine Learning • Projeto P2**

## Objetivo
Implementar avaliação reproduzível da qualidade e dos erros da primeira versão. O conteúdo é neutro: nenhuma área, dataset, rótulo ou quantidade de entradas é imposta.

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
python aula_03.py --config config.projeto.json
```
Para a arquitetura própria: `python aula_03.py --config config.adaptador.json`.

## Roteiro de implementação no projeto
### Passo 1 — Avaliar além de funcionar
Implementar avaliação reproduzível da qualidade e dos erros da primeira versão.

**No código:** `avaliar_desenvolvimento / executar_casos`.

**Ação do grupo:** Uma resposta produzida não comprova que ela resolve o problema.

**Evidência:** registre entrada/configuração, versão, resultado observado e uma interpretação. Não basta mostrar “executou”.

### Passo 2 — Métrica adequada à tarefa
Classificação: precisão/recall/F1; regressão: MAE/RMSE; recomendação: precisão no top-k; texto: rubrica humana e evidências.

**No código:** `medir ou avaliador do projeto`.

**Ação do grupo:** O código pronto calcula métricas tabulares e regras de casos; top-k/rubrica específica precisam ser ligados pelo grupo.

**Evidência:** registre entrada/configuração, versão, resultado observado e uma interpretação. Não basta mostrar “executou”.

### Passo 3 — Referência para comparar
Criar um baseline, uma regra simples ou a versão anterior.

**No código:** `DummyClassifier / DummyRegressor / versão anterior`.

**Ação do grupo:** Comparar com algo plausível; não chamar qualquer resultado de melhoria.

**Evidência:** registre entrada/configuração, versão, resultado observado e uma interpretação. Não basta mostrar “executou”.

### Passo 4 — Casos e critérios esperados
Declarar casos comuns, limites, inválidos e comportamento esperado antes de executar.

**No código:** `avaliar_regra / casos_do_grupo.json`.

**Ação do grupo:** Oráculo igual/aproximado/contem serve a critérios específicos; nao_vazio só verifica contrato.

**Evidência:** registre entrada/configuração, versão, resultado observado e uma interpretação. Não basta mostrar “executou”.

### Passo 5 — Avaliação sem vazamento
Usar folds adequados ou conjunto de desenvolvimento; casos finais ficam reservados.

**No código:** `definir_folds`.

**Ação do grupo:** Em tempo, treinar com passado e avaliar futuro; por grupo, separar entidades.

**Evidência:** registre entrada/configuração, versão, resultado observado e uma interpretação. Não basta mostrar “executou”.

### Passo 6 — Automatizar e registrar
Guardar previsões/casos, tempo, erro e versão para repetir o teste.

**No código:** `executar_casos / avaliar_desenvolvimento`.

**Ação do grupo:** Distinguir erro do programa, resposta incorreta e resposta incompleta.

**Evidência:** registre entrada/configuração, versão, resultado observado e uma interpretação. Não basta mostrar “executou”.

### Passo 7 — Investigar falhas
Analisar cinco erros relevantes e propor hipótese concreta.

**No código:** `previsoes_inicial.csv / casos_desenvolvimento.json`.

**Ação do grupo:** Relacionar causa provável e correção possível; respostas de texto exigem inspeção humana.

**Evidência:** registre entrada/configuração, versão, resultado observado e uma interpretação. Não basta mostrar “executou”.

### Passo 8 — Entrega da Aula 3
Métricas ou rubrica, comparação com referência e análise de erros.

**No código:** `aula_03.py`.

**Ação do grupo:** Defender por que os critérios avaliam a tarefa. Próxima: experimentar melhorias.

**Evidência:** registre entrada/configuração, versão, resultado observado e uma interpretação. Não basta mostrar “executou”.

## Código explicado parte por parte
Leia os blocos do caminho escolhido. As funções tabulares não precisam ser usadas em um projeto de outra arquitetura.

### Bloco 1 — `motor_tabular.py` → `definir_folds`
**O que implementa:** Criar rodadas compatíveis com a origem.

**Entrada e saída:** Recebe treino X/y/auxiliar e devolve pares de índices.

```python
def definir_folds(config, X, y, auxiliar):
    if config['divisao'] == 'grupo':
        n = min(5, auxiliar.nunique())
        if n < 2: raise ValueError('Treino precisa de ao menos dois grupos para validar.')
        return list(GroupKFold(n).split(X, y, groups=auxiliar))
    if config['divisao'] == 'tempo':
        n = min(5, len(X) - 1)
        pares = []
        for tr, va in TimeSeriesSplit(n).split(X):
            # Retira do treino registros no mesmo instante que o início da validação.
            tr = tr[(auxiliar.iloc[tr] < auxiliar.iloc[va].min()).to_numpy()]
            if len(tr): pares.append((tr, va))
        if len(pares) < 2: raise ValueError('Poucos períodos distintos para validar no tempo.')
        return pares
    if config['tarefa'] == 'classificacao':
        n = min(5, int(y.value_counts().min()))
        if n < 2: raise ValueError('Classe com poucos exemplos no treino.')
        return list(StratifiedKFold(n, shuffle=True, random_state=config['seed']).split(X, y))
    return list(KFold(5, shuffle=True, random_state=config['seed']).split(X))
```

1. Grupo: GroupKFold mantém entidade junta. Tempo: TimeSeriesSplit usa passado para prever períodos posteriores.
2. Remove do treino valores no instante inicial da validação temporal. Primeiros exemplos temporais podem não ter previsão fora de fold.
3. Aleatório: StratifiedKFold para classes ou KFold para regressão. Classe rara/grupos/períodos insuficientes podem impedir treinamento.
4. Em classificação temporal ou por grupo, um fold pode ter só uma classe; não force mistura futura para esconder isso: revisar dados/períodos e protocolar a limitação.

**Pergunta de checagem:** qual problema esse bloco evita no seu projeto e que evidência prova que funcionou?

### Bloco 2 — `motor_tabular.py` → `medir`
**O que implementa:** Medir de acordo com a tarefa.

**Entrada e saída:** Recebe valores reais e previstos; devolve métricas.

```python
def medir(config, real, previsto):
    if config['tarefa'] == 'classificacao':
        classes = sorted(set(real) | set(previsto))
        return {'acuracia': float(accuracy_score(real, previsto)),
                'f1_macro': float(f1_score(real, previsto, average='macro', zero_division=0)),
                'relatorio': classification_report(real, previsto, output_dict=True, zero_division=0),
                'classes': classes, 'matriz': confusion_matrix(real, previsto, labels=classes).tolist()}
    r2 = float(r2_score(real, previsto))
    return {'mae': float(mean_absolute_error(real, previsto)),
            'rmse': float(np.sqrt(mean_squared_error(real, previsto))),
            'r2': r2 if np.isfinite(r2) else None}
```

1. Classificação: acurácia, F1 macro, relatório por classe e matriz com ordem explicitada.
2. Regressão: MAE na unidade do alvo; RMSE penaliza mais erros grandes; R² compara com referência de variabilidade e pode ser negativo.
3. Nenhuma dessas métricas avalia texto livre, recomendação ou imagem de forma universal.

**Pergunta de checagem:** qual problema esse bloco evita no seu projeto e que evidência prova que funcionou?

### Bloco 3 — `motor_tabular.py` → `avaliar_desenvolvimento`
**O que implementa:** Comparar baseline e versão inicial.

**Entrada e saída:** Recebe config; executa folds só nos dados de desenvolvimento.

```python
def avaliar_desenvolvimento(config):
    from sklearn.base import clone
    X, y, auxiliar, tr, _ = preparar_experimento(config)
    Xtr, ytr = X.iloc[tr], y.iloc[tr]
    aux = auxiliar.iloc[tr] if auxiliar is not None else None
    pares = definir_folds(config, Xtr, ytr, aux)
    referencia = DummyClassifier(strategy='most_frequent') if config['tarefa'] == 'classificacao' else DummyRegressor(strategy='mean')
    for nome, estimador in [('referencia', referencia), ('inicial', candidatos(config)[0][1])]:
        linhas, valores = [], []
        for numero, (a, b) in enumerate(pares):
            modelo = construir_pipeline(config, clone(estimador))
            modelo.fit(Xtr.iloc[a], ytr.iloc[a])
            previsto = modelo.predict(Xtr.iloc[b])
            valores.append(medir(config, ytr.iloc[b], previsto))
            for posicao, pred in zip(b, previsto):
                linhas.append({'indice': int(Xtr.index[posicao]), 'fold': numero,
                               'real': ytr.iloc[posicao], 'previsto': pred})
        pd.DataFrame(linhas).to_csv(Path(config['saida']) / f'previsoes_{nome}.csv', index=False)
        salvar_json(Path(config['saida']) / f'avaliacao_{nome}.json', {
            'por_fold': valores, 'aviso': 'Em divisão temporal, primeiros exemplos podem não ter previsão de validação.'})
```

1. DummyClassifier escolhe a classe majoritária; DummyRegressor prevê média do treino.
2. Cada fold cria uma nova pipeline e aprende somente com sua parte de treino.
3. Salva previsões com índice/fold e métricas por rodada. Inspecionar CSV permite ligar erro ao caso real.
4. Este código calcula evidência quantitativa; o grupo ainda interpreta erros e requisitos do seu projeto.

**Pergunta de checagem:** qual problema esse bloco evita no seu projeto e que evidência prova que funcionou?

### Bloco 4 — `contrato.py` → `avaliar_regra`
**O que implementa:** Declarar um oráculo verificável.

**Entrada e saída:** Recebe resultado da IA e regra do caso; devolve verdadeiro/falso.

```python
def avaliar_regra(resultado, regra):
    """Oráculos simples e explícitos; não fingem avaliar significado de texto."""
    tipo = regra['tipo']
    if tipo == 'igual':
        return resultado == regra['valor']
    if tipo == 'aproximado':
        if isinstance(resultado, bool) or not isinstance(resultado, (float, int)):
            return False
        return math.isfinite(resultado) and abs(resultado - regra['valor']) <= regra['tolerancia']
    if tipo == 'contem':
        return isinstance(resultado, str) and all(t in resultado for t in regra['trechos'])
    if tipo == 'nao_vazio':
        return resultado is not None and resultado != '' and resultado != [] and resultado != {}
    raise ValueError(f'Oráculo não implementado: {tipo}')
```

1. igual confere resposta exata; aproximado compara valor numérico com tolerância explícita.
2. contem confere trechos de texto; nao_vazio confere somente presença de saída.
3. Oráculos simples não conferem significado: um texto errado pode ter as palavras esperadas. Para saídas abertas, complementar com rubrica humana e fontes.

**Pergunta de checagem:** qual problema esse bloco evita no seu projeto e que evidência prova que funcionou?

### Bloco 5 — `contrato.py` → `executar_casos`
**O que implementa:** Executar cenários e guardar evidências.

**Entrada e saída:** Recebe config, função de inferência e fase.

```python
def executar_casos(config, funcao, fase='desenvolvimento'):
    casos = ler_json(config['casos'])
    selecionados = [c for c in casos if c['fase'] == fase]
    if not selecionados:
        raise ValueError(f'Nenhum caso definido para a fase {fase}.')
    registros = []
    for caso in selecionados:
        inicio = perf_counter()
        try:
            resposta = funcao(caso['entrada'])
            passou = (not caso.get('espera_erro', False)
                      and avaliar_regra(resposta['resultado'], caso['regra']))
            registro = {'id': caso['id'], 'passou': bool(passou), 'resposta': resposta}
        except Exception as erro:
            # Captura ampla só no executor de testes: registra o erro, não o oculta.
            # Um caso de recusa deve declarar a classe esperada, não aceitar qualquer crash.
            esperado = caso.get('erro_tipo', 'ValueError')
            passou = caso.get('espera_erro', False) and type(erro).__name__ == esperado
            registro = {'id': caso['id'], 'passou': bool(passou),
                        'erro_tipo': type(erro).__name__, 'erro': str(erro)}
        registro['segundos'] = perf_counter() - inicio
        registros.append(registro)
    relatorio = {'fase': fase, 'quantidade': len(registros),
                 'taxa_sucesso': sum(r['passou'] for r in registros) / len(registros),
                 'casos': registros,
                 'limite': 'Taxa relativa aos oráculos definidos; não comprova correção semântica.'}
    salvar_json(Path(config['saida']) / f'casos_{fase}.json', relatorio)
    return relatorio
```

1. Seleciona somente desenvolvimento ou final; as fases devem ter casos reais distintos.
2. Mede tempo e registra resposta/erro, id e status. Espera a classe declarada de exceção em um caso de recusa.
3. A captura ampla é exclusiva do executor de testes: um crash não conta como sucesso salvo se for o erro explicitamente esperado.
4. Taxa de sucesso resume os oráculos escolhidos, não a qualidade universal da IA. Registrar versão e rubrica junto ao relatório.

**Pergunta de checagem:** qual problema esse bloco evita no seu projeto e que evidência prova que funcionou?

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
