import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import joblib

BASE_DIR = Path(__file__).resolve().parent
CSV_ORIGINAL = BASE_DIR / "data" / "partidas_futebol.csv"
ARTEFATOS_V2 = BASE_DIR / "artefatos_v2"
ARTEFATOS_V2.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_ORIGINAL)

elos = {}
historico_forma = {}
jogos_recentes = {}

elo_mandante = []
elo_visitante = []
diff_elo = []
forma_mandante = []
forma_visitante = []
diff_forma = []

for _, row in df.iterrows():
    m = row['mandante']
    v = row['visitante']
    res = row['resultado']
    neutro = row['campo_neutro']

    e_m = elos.get(m, 1500.0)
    e_v = elos.get(v, 1500.0)
    bonus_casa = 70.0 if neutro == 'Nao' else 0.0
    d_e = (e_m - e_v) + bonus_casa

    f_m_lista = historico_forma.get(m, [])
    f_v_lista = historico_forma.get(v, [])
    f_m = sum(f_m_lista[-5:])
    f_v = sum(f_v_lista[-5:])
    d_f = f_m - f_v

    elo_mandante.append(round(e_m, 1))
    elo_visitante.append(round(e_v, 1))
    diff_elo.append(round(d_e, 1))
    forma_mandante.append(f_m)
    forma_visitante.append(f_v)
    diff_forma.append(d_f)

    e_score_m = 1.0 / (1.0 + 10.0 ** ((e_v - (e_m + bonus_casa)) / 400.0))
    if res == 'vitoria_mandante':
        s_m = 1.0
        pts_m = 3
        pts_v = 0
        sigla_m = 'V'
        sigla_v = 'D'
    elif res == 'empate':
        s_m = 0.5
        pts_m = 1
        pts_v = 1
        sigla_m = 'E'
        sigla_v = 'E'
    else:
        s_m = 0.0
        pts_m = 0
        pts_v = 3
        sigla_m = 'D'
        sigla_v = 'V'

    k_factor = 32.0
    elos[m] = e_m + k_factor * (s_m - e_score_m)
    elos[v] = e_v + k_factor * ((1.0 - s_m) - (1.0 - e_score_m))

    if m not in historico_forma:
        historico_forma[m] = []
        jogos_recentes[m] = []
    if v not in historico_forma:
        historico_forma[v] = []
        jogos_recentes[v] = []

    historico_forma[m].append(pts_m)
    historico_forma[v].append(pts_v)
    jogos_recentes[m].append(sigla_m)
    jogos_recentes[v].append(sigla_v)

df['elo_mandante'] = elo_mandante
df['elo_visitante'] = elo_visitante
df['diff_elo'] = diff_elo
df['forma_mandante'] = forma_mandante
df['forma_visitante'] = forma_visitante
df['diff_forma'] = diff_forma

df.to_csv(BASE_DIR / "data" / "partidas_futebol_v2.csv", index=False)

estado_selecoes = {}
for selecao in sorted(list(set(df['mandante'].unique().tolist() + df['visitante'].unique().tolist()))):
    e = round(elos.get(selecao, 1500.0), 1)
    pts = sum(historico_forma.get(selecao, [])[-5:])
    ultimos = jogos_recentes.get(selecao, [])[-5:]
    estado_selecoes[selecao] = {
        "elo": e,
        "pontos_ultimos_5": pts,
        "aproveitamento_ultimos_5_pct": round((pts / 15.0) * 100.0, 1) if pts > 0 else 0.0,
        "ultimos_5_jogos": ultimos
    }

with open(ARTEFATOS_V2 / "estado_atual_selecoes.json", "w", encoding="utf-8") as f:
    json.dump(estado_selecoes, f, ensure_ascii=False, indent=2)

ranking_ordenado = sorted(estado_selecoes.items(), key=lambda item: item[1]['elo'], reverse=True)
top_ranking = [{"posicao": i+1, "selecao": item[0], "elo": item[1]['elo'], "pontos_ultimos_5": item[1]['pontos_ultimos_5'], "sequencia": "".join(item[1]['ultimos_5_jogos'])} for i, item in enumerate(ranking_ordenado)]

with open(ARTEFATOS_V2 / "ranking_elo_2024.json", "w", encoding="utf-8") as f:
    json.dump(top_ranking, f, ensure_ascii=False, indent=2)

features_numericas = ['diff_elo', 'diff_forma', 'elo_mandante', 'elo_visitante']
features_categoricas = ['torneio', 'campo_neutro']
alvo = 'resultado'

X = df[features_numericas + features_categoricas]
y = df[alvo]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

prep_num = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

prep_cat = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore'))
])

preparador = ColumnTransformer([
    ('num', prep_num, features_numericas),
    ('cat', prep_cat, features_categoricas)
])

estimador = RandomForestClassifier(
    n_estimators=200,
    max_depth=16,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

modelo_v2 = Pipeline([
    ('preparo', preparador),
    ('modelo', estimador)
])

modelo_v2.fit(X_tr, y_tr)

y_pred = modelo_v2.predict(X_te)
acuracia = float(accuracy_score(y_te, y_pred))
f1_macro = float(f1_score(y_te, y_pred, average='macro'))
relatorio = classification_report(y_te, y_pred, output_dict=True)
classes = sorted(list(y.unique()))
matriz = confusion_matrix(y_te, y_pred, labels=classes).tolist()

metricas_v2 = {
    "versao": "2.0.0",
    "modelo": "RandomForestClassifier (Ensemble com Elo e Forma Recente)",
    "acuracia_teste": round(acuracia, 4),
    "f1_macro_teste": round(f1_macro, 4),
    "classes": classes,
    "matriz_confusao": matriz,
    "relatorio_completo": relatorio,
    "features_utilizadas": {
        "numericas": features_numericas,
        "categoricas": features_categoricas
    },
    "comparacao_baseline": {
        "f1_macro_inicial_aula02": 0.4201,
        "f1_macro_v2": round(f1_macro, 4),
        "ganho_absoluto": round(f1_macro - 0.4201, 4),
        "ganho_percentual": f"+{round(((f1_macro - 0.4201) / 0.4201) * 100, 1)}%"
    }
}

joblib.dump(modelo_v2, ARTEFATOS_V2 / "modelo_v2.joblib")

with open(ARTEFATOS_V2 / "metricas_v2.json", "w", encoding="utf-8") as f:
    json.dump(metricas_v2, f, ensure_ascii=False, indent=2)

print("Treinamento V2 concluido com sucesso!")
print(f"Acuracia Teste: {acuracia:.4f}")
print(f"F1-Macro Teste: {f1_macro:.4f}")
print(f"Ganho frente ao inicial: {metricas_v2['comparacao_baseline']['ganho_percentual']}")
