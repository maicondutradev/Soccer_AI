import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GroupShuffleSplit, KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
import joblib

from contrato import ler_json, salvar_json

def candidatos(config):
    if config['tarefa'] == 'classificacao':
        from sklearn.linear_model import LogisticRegression
        return [('linear', LogisticRegression(max_iter=1000, random_state=config['seed']))]
    from sklearn.linear_model import Ridge
    return [('linear', Ridge(random_state=config['seed']))]

def assinatura(config):
    import hashlib
    limpo = {k: v for k, v in config.items() if not k.startswith('_')}
    payload = json.dumps(limpo, sort_keys=True)
    h = hashlib.sha256(payload.encode('utf-8'))
    csv_path = Path(config['csv'])
    if csv_path.exists():
        h.update(str(csv_path.stat().st_mtime).encode('utf-8'))
        h.update(str(csv_path.stat().st_size).encode('utf-8'))
    return h.hexdigest()[:16]

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

def preparar_experimento(config):
    saida = Path(config['saida'])
    saida.mkdir(parents=True, exist_ok=True)
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

def medir(config, real, previsto):
    if config['tarefa'] == 'classificacao':
        classes = sorted(set(real) | set(previsto))
        return {'acuracia': float(accuracy_score(real, previsto)),
                'f1_macro': float(f1_score(real, previsto, average='macro', zero_division=0)),
                'relatorio': classification_report(real, previsto, output_dict=True, zero_division=0),
                'classes': classes,
                'matriz': confusion_matrix(real, previsto, labels=classes).tolist()}
    r2 = float(r2_score(real, previsto))
    return {'mae': float(mean_absolute_error(real, previsto)),
            'rmse': float(np.sqrt(mean_squared_error(real, previsto))),
            'r2': r2 if np.isfinite(r2) else None}

def treinar_inicial(config):
    X, y, _, tr, _ = preparar_experimento(config)
    modelo = construir_pipeline(config, candidatos(config)[0][1])
    modelo.fit(X.iloc[tr], y.iloc[tr])
    joblib.dump(modelo, Path(config['saida']) / 'modelo_inicial.joblib')
    metricas = medir(config, y.iloc[tr], modelo.predict(X.iloc[tr]))
    salvar_json(Path(config['saida']) / 'treino_inicial.json', {
        'metricas_treino': metricas,
        'limite': 'Treino é diagnóstico; avaliação de generalização vem nas próximas aulas.'})
    return modelo, metricas

def definir_folds(config, X, y, auxiliar):
    if config['divisao'] == 'grupo':
        n = min(5, auxiliar.nunique())
        if n < 2:
            raise ValueError('Treino precisa de ao menos dois grupos para validar.')
        return list(GroupKFold(n).split(X, y, groups=auxiliar))
    if config['divisao'] == 'tempo':
        n = min(5, len(X) - 1)
        pares = []
        for tr, va in TimeSeriesSplit(n).split(X):
            tr = tr[(auxiliar.iloc[tr] < auxiliar.iloc[va].min()).to_numpy()]
            if len(tr):
                pares.append((tr, va))
        if len(pares) < 2:
            raise ValueError('Poucos períodos distintos para validar no tempo.')
        return pares
    if config['tarefa'] == 'classificacao':
        n = min(5, int(y.value_counts().min()))
        if n < 2:
            raise ValueError('Classe com poucos exemplos no treino.')
        return list(StratifiedKFold(n, shuffle=True, random_state=config['seed']).split(X, y))
    return list(KFold(5, shuffle=True, random_state=config['seed']).split(X))

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
            'por_fold': valores,
            'aviso': 'Em divisão temporal, primeiros exemplos podem não ter previsão de validação.'})
