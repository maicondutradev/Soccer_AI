import json
import math
from pathlib import Path
from time import perf_counter

def ler_json(caminho):
    with open(caminho, 'r', encoding='utf-8') as f:
        return json.load(f)

def salvar_json(caminho, dados):
    p = Path(caminho)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)

def configurar(caminho='config.projeto.json'):
    config = ler_json(caminho)
    if config.get('modo') not in ('tabular', 'adaptador'):
        raise ValueError('modo deve ser tabular ou adaptador.')
    if not config.get('nome') or 'PREENCHER' in config['nome']:
        raise ValueError('Preencha o nome do seu projeto no JSON.')
    config['_arquivo_config'] = str(Path(caminho).resolve())
    return config

def validar_payload(entrada, schema):
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

def avaliar_regra(resultado, regra):
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
