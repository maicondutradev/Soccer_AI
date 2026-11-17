import argparse
import sys
from pathlib import Path
import pandas as pd
import joblib

from contrato import configurar, validar_payload, executar_casos, ler_json
from motor_tabular import carregar_dados, preparar_experimento, treinar_inicial, avaliar_desenvolvimento

def criar_funcao_inferencia(config):
    modelo_path = Path(config["saida"]) / "modelo_inicial.joblib"
    if not modelo_path.exists():
        raise FileNotFoundError("Modelo inicial nao encontrado. Execute aula_02.py primeiro.")
    modelo = joblib.load(modelo_path)
    schema = config.get("schema", {})

    def inferir(entrada):
        validar_payload(entrada, schema)
        df_in = pd.DataFrame([entrada])
        colunas = config["numericas"] + config["categoricas"]
        df_in = df_in[colunas]
        pred = modelo.predict(df_in)[0]
        return {"resultado": str(pred), "versao": config.get("versao", "1.0.0")}

    return inferir

def executar_etapa_2(config):
    print("=" * 60)
    print(f"AULA 02 - PROJETO: {config['nome']} (Versao: {config.get('versao', '1.0.0')})")
    print(f"Modo: {config['modo']} | Tarefa: {config['tarefa']} | Metrica: {config['metrica']}")
    print("=" * 60)

    if "schema" in config:
        print("\n[Passo 3] Validando contrato de entrada (schema)...")
        campos_obrigatorios = [k for k, v in config["schema"].items() if v.get("obrigatorio", True)]
        print(f"Campos no contrato: {list(config['schema'].keys())}")
        print(f"Campos obrigatorios: {campos_obrigatorios}")

    print("\n[Passo 4] Carregando e auditando dados...")
    X, y, auxiliar = carregar_dados(config)
    print(f"Registros validos: {len(X)}")
    print(f"Entradas categoricas: {config['categoricas']}")
    print(f"Entradas numericas: {config['numericas']}")
    print(f"Coluna alvo: {config['alvo']} (Classes: {sorted(y.unique().tolist())})")

    print("\n[Passo 5 e 6] Preparando experimento e separacao de dados...")
    X, y, auxiliar, treino, teste = preparar_experimento(config)
    print(f"Estrategia de divisao: {config['divisao']}")
    print(f"Tamanho do conjunto de treino: {len(treino)} ({(len(treino)/len(X))*100:.1f}%)")
    print(f"Tamanho do conjunto de teste: {len(teste)} ({(len(teste)/len(X))*100:.1f}%)")

    print("\n[Passo 7] Treinando modelo inicial (baseline)...")
    modelo, metricas = treinar_inicial(config)
    print("Modelo treinado e salvo com sucesso!")
    print(f"Acuracia no treino: {metricas.get('acuracia', 0):.4f}")
    print(f"F1 Macro no treino: {metricas.get('f1_macro', 0):.4f}")

    saida = Path(config["saida"])
    print("\n[Passo 8] Artefatos gerados:")
    for arq in sorted(saida.glob("*")):
        print(f" - {arq.name} ({arq.stat().st_size} bytes)")

    print("\nExecucao da Aula 02 concluida com sucesso!")

def executar_etapa_3(config, fase="desenvolvimento"):
    print("=" * 60)
    print(f"AULA 03 - AVALIACAO DE QUALIDADE: {config['nome']}")
    print(f"Tarefa: {config['tarefa']} | Metrica principal: {config['metrica']}")
    print("=" * 60)

    print("\n[Passo 1, 3 e 5] Executando validacao cruzada (folds de desenvolvimento)...")
    avaliar_desenvolvimento(config)
    print("Validacao cruzada concluida com sucesso!")

    saida = Path(config["saida"])
    aval_ref = ler_json(saida / "avaliacao_referencia.json")
    aval_ini = ler_json(saida / "avaliacao_inicial.json")

    f1_ref = [x["f1_macro"] for x in aval_ref["por_fold"]]
    acc_ref = [x["acuracia"] for x in aval_ref["por_fold"]]
    f1_ini = [x["f1_macro"] for x in aval_ini["por_fold"]]
    acc_ini = [x["acuracia"] for x in aval_ini["por_fold"]]

    media_f1_ref = sum(f1_ref) / len(f1_ref)
    media_acc_ref = sum(acc_ref) / len(acc_ref)
    media_f1_ini = sum(f1_ini) / len(f1_ini)
    media_acc_ini = sum(acc_ini) / len(acc_ini)

    print("\n[Passo 2 e 8] Comparativo com Modelo de Referencia (Baseline Dummy):")
    print("-" * 60)
    print(f"Referencia (Dummy Most Frequent):")
    print(f"  - Acuracia media: {media_acc_ref:.4f}")
    print(f"  - F1 Macro medio: {media_f1_ref:.4f}")
    print(f"Modelo Inicial (Regressao Logistica):")
    print(f"  - Acuracia media: {media_acc_ini:.4f}")
    print(f"  - F1 Macro medio: {media_f1_ini:.4f}")
    print(f"Ganho de F1 Macro sobre baseline: {(media_f1_ini - media_f1_ref):+.4f}")
    print("-" * 60)

    print(f"\n[Passo 4 e 6] Executando casos de teste declarados (fase: {fase})...")
    inferir = criar_funcao_inferencia(config)
    rel_desenv = executar_casos(config, inferir, fase=fase)
    print(f"Casos ({fase}):")
    print(f"  - Total testado: {rel_desenv['quantidade']}")
    print(f"  - Taxa de sucesso: {rel_desenv['taxa_sucesso'] * 100:.1f}%")
    for c in rel_desenv["casos"]:
        status = "PASSOU" if c["passou"] else "FALHOU"
        detalhe = c.get("resposta", {}).get("resultado") or c.get("erro")
        print(f"    * [{status}] {c['id']}: {detalhe}")

    if fase == "desenvolvimento":
        try:
            rel_final = executar_casos(config, inferir, fase="final")
            print(f"\nCasos da Fase Final (Ineditos):")
            print(f"  - Total testado: {rel_final['quantidade']}")
            print(f"  - Taxa de sucesso: {rel_final['taxa_sucesso'] * 100:.1f}%")
            for c in rel_final["casos"]:
                status = "PASSOU" if c["passou"] else "FALHOU"
                detalhe = c.get("resposta", {}).get("resultado") or c.get("erro")
                print(f"    * [{status}] {c['id']}: {detalhe}")
        except ValueError:
            pass

    print("\n[Passo 7] Investigacao de previsoes e erros:")
    prev_csv = saida / "previsoes_inicial.csv"
    if prev_csv.exists():
        df_prev = pd.read_csv(prev_csv)
        erros = df_prev[df_prev["real"] != df_prev["previsto"]]
        print(f"Total de previsoes na validacao cruzada: {len(df_prev)}")
        print(f"Total de erros de classificacao: {len(erros)} ({(len(erros)/len(df_prev))*100:.1f}%)")
        print(f"Amostra de 5 casos com discrepancia para investigacao:")
        for idx, row in erros.head(5).iterrows():
            print(f"  - Linha {row['indice']}: Real='{row['real']}' vs Previsto='{row['previsto']}'")

    print("\nExecucao da Aula 03 concluida com sucesso!")

def cli(etapa):
    parser = argparse.ArgumentParser(description=f"Execucao da Etapa {etapa}")
    parser.add_argument("--config", default="config.projeto.json", help="Arquivo de configuracao")
    parser.add_argument("--fase", default="desenvolvimento", help="Fase dos casos de teste")
    args = parser.parse_args()

    config = configurar(args.config)
    if etapa == 2:
        executar_etapa_2(config)
    elif etapa == 3:
        executar_etapa_3(config, fase=args.fase)
    else:
        raise ValueError(f"Etapa {etapa} nao suportada.")
