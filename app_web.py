import os
import json
import math
from pathlib import Path
from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

BASE_DIR = Path(__file__).resolve().parent

with open(BASE_DIR / "config.projeto.json", "r", encoding="utf-8") as f:
    config = json.load(f)

v2_modelo_path = BASE_DIR / "artefatos_v2" / "modelo_v2.joblib"
v1_modelo_path = BASE_DIR / config["saida"] / "modelo_inicial.joblib"

if v2_modelo_path.exists():
    modelo = joblib.load(v2_modelo_path)
    usando_v2 = True
elif v1_modelo_path.exists():
    modelo = joblib.load(v1_modelo_path)
    usando_v2 = False
else:
    raise FileNotFoundError("Nenhum modelo serializado encontrado em artefatos_v2 ou artefatos_v1")

estado_selecoes_path = BASE_DIR / "artefatos_v2" / "estado_atual_selecoes.json"
if estado_selecoes_path.exists():
    with open(estado_selecoes_path, "r", encoding="utf-8") as f:
        estado_selecoes = json.load(f)
else:
    estado_selecoes = {}

csv_path = BASE_DIR / config["csv"]
if csv_path.exists():
    df_historico = pd.read_csv(csv_path)
    todas_selecoes = sorted(list(set(df_historico["mandante"].dropna().unique().tolist() + 
                                    df_historico["visitante"].dropna().unique().tolist())))
    todos_torneios = df_historico["torneio"].value_counts().index.tolist()
else:
    df_historico = None
    todas_selecoes = ["Brazil", "Argentina", "France", "Germany", "England", "Spain", "Italy", "Uruguay"]
    todos_torneios = ["FIFA World Cup", "Copa América", "UEFA Euro", "Friendly", "FIFA World Cup qualification"]

CONFEDERACOES = {
    "CONMEBOL": {
        "Brazil", "Argentina", "Uruguay", "Colombia", "Chile", "Paraguay", 
        "Peru", "Ecuador", "Venezuela", "Bolivia"
    },
    "UEFA": {
        "Germany", "France", "England", "Spain", "Italy", "Portugal", "Netherlands",
        "Belgium", "Croatia", "Denmark", "Sweden", "Switzerland", "Poland", "Austria",
        "Scotland", "Wales", "Northern Ireland", "Czech Republic", "Czechia", "Norway",
        "Greece", "Turkey", "Ukraine", "Hungary", "Serbia", "Romania", "Republic of Ireland",
        "Ireland", "Finland", "Slovakia", "Slovenia", "Albania", "Georgia", "Bosnia and Herzegovina",
        "North Macedonia", "Bulgaria", "Iceland", "Israel", "Russia", "Cyprus", "Estonia",
        "Latvia", "Lithuania", "Luxembourg", "Malta", "Moldova", "Montenegro", "Armenia",
        "Azerbaijan", "Belarus", "Kazakhstan", "Kosovo", "Andorra", "San Marino", "Gibraltar",
        "Liechtenstein", "Faroe Islands"
    },
    "CONCACAF": {
        "United States", "USA", "Mexico", "Canada", "Costa Rica", "Panama", "Jamaica",
        "Honduras", "El Salvador", "Haiti", "Trinidad and Tobago", "Guatemala", "Curacao",
        "Martinique", "Suriname", "Cuba", "Nicaragua", "Bermuda", "Antigua and Barbuda",
        "Dominican Republic", "Grenada", "Barbados", "Saint Lucia", "Belize", "Puerto Rico"
    },
    "CAF": {
        "Morocco", "Senegal", "Nigeria", "Egypt", "Algeria", "Ghana", "Cameroon",
        "Ivory Coast", "South Africa", "Tunisia", "Mali", "DR Congo", "Burkina Faso",
        "Guinea", "Cape Verde", "Angola", "Zambia", "Uganda", "Gabon", "Benin",
        "Equatorial Guinea", "Congo", "Kenya", "Mauritania", "Mozambique", "Togo", "Sudan"
    },
    "AFC": {
        "Japan", "South Korea", "Korea Republic", "Australia", "Saudi Arabia", "Iran",
        "Qatar", "China", "China PR", "Iraq", "United Arab Emirates", "Jordan", "Uzbekistan",
        "Oman", "Bahrain", "Syria", "Palestine", "Vietnam", "Thailand", "India", "Tajikistan",
        "Lebanon", "Kyrgyzstan", "Malaysia", "Kuwait", "Philippines", "Indonesia", "North Korea"
    }
}

TORNEIOS_RESTRITOS = {
    "UEFA Euro": "UEFA",
    "UEFA Nations League": "UEFA",
    "UEFA Euro qualification": "UEFA",
    "African Cup of Nations": "CAF",
    "African Cup of Nations qualification": "CAF",
    "AFC Asian Cup": "AFC",
    "AFC Asian Cup qualification": "AFC",
    "Gold Cup": "CONCACAF"
}

def construir_mapa_confederacoes():
    mapa = {}
    for conf, times in CONFEDERACOES.items():
        for t in times:
            mapa[t] = conf
    if df_historico is not None:
        regras_inferencia = {
            **TORNEIOS_RESTRITOS,
            "CONCACAF Nations League": "CONCACAF",
            "CONCACAF Gold Cup": "CONCACAF",
        }
        for torneio_nome, conf in regras_inferencia.items():
            jogos_t = df_historico[df_historico["torneio"] == torneio_nome]
            if jogos_t.empty:
                continue
            times_t = set(jogos_t["mandante"].dropna()).union(set(jogos_t["visitante"].dropna()))
            for t in times_t:
                if t not in mapa:
                    mapa[t] = conf
    return mapa

MAPA_CONFEDERACOES = construir_mapa_confederacoes()

def obter_confederacao(selecao):
    return MAPA_CONFEDERACOES.get(selecao, "Outra")

def validar_torneio_geografico(mandante, visitante, torneio):
    if torneio == "Copa América":
        conf_m = obter_confederacao(mandante)
        conf_v = obter_confederacao(visitante)
        permitidas = {"CONMEBOL", "CONCACAF"}
        incomp = []
        if conf_m not in permitidas:
            incomp.append(f"{mandante} ({conf_m})")
        if conf_v not in permitidas:
            incomp.append(f"{visitante} ({conf_v})")
        if incomp:
            return f"Torneio Incompatível: {' e '.join(incomp)} não disputa a Copa América (torneio restrito às Américas). Escolha FIFA World Cup ou Amistoso."
        return None

    if torneio in TORNEIOS_RESTRITOS:
        conf_exigida = TORNEIOS_RESTRITOS[torneio]
        conf_m = obter_confederacao(mandante)
        conf_v = obter_confederacao(visitante)
        incomp = []
        if conf_m != conf_exigida:
            incomp.append(f"{mandante} ({conf_m})")
        if conf_v != conf_exigida:
            incomp.append(f"{visitante} ({conf_v})")
        if incomp:
            return f"Torneio Incompatível: {' e '.join(incomp)} não pertence à {conf_exigida} e não disputa o torneio {torneio}. Escolha FIFA World Cup ou Amistoso."

    return None

def obter_estatisticas_historico(mandante, visitante):
    if df_historico is None:
        return {"total": 0, "mensagem": "Base histórica não disponível."}

    mask_direto = (df_historico["mandante"] == mandante) & (df_historico["visitante"] == visitante)
    mask_inverso = (df_historico["mandante"] == visitante) & (df_historico["visitante"] == mandante)
    jogos = df_historico[mask_direto | mask_inverso]
    total = len(jogos)

    if total == 0:
        return {
            "total": 0,
            "vitorias_mandante": 0,
            "vitorias_visitante": 0,
            "empates": 0,
            "mensagem": "Nenhum confronto histórico registrado entre essas duas seleções na base oficial (1872-2024)."
        }

    vm1 = len(df_historico[(df_historico["mandante"] == mandante) & (df_historico["visitante"] == visitante) & (df_historico["resultado"] == "vitoria_mandante")])
    vm2 = len(df_historico[(df_historico["mandante"] == visitante) & (df_historico["visitante"] == mandante) & (df_historico["resultado"] == "vitoria_visitante")])
    vitorias_m = vm1 + vm2

    vv1 = len(df_historico[(df_historico["mandante"] == mandante) & (df_historico["visitante"] == visitante) & (df_historico["resultado"] == "vitoria_visitante")])
    vv2 = len(df_historico[(df_historico["mandante"] == visitante) & (df_historico["visitante"] == mandante) & (df_historico["resultado"] == "vitoria_mandante")])
    vitorias_v = vv1 + vv2

    empates = len(jogos[jogos["resultado"] == "empate"])

    return {
        "total": total,
        "vitorias_mandante": vitorias_m,
        "vitorias_visitante": vitorias_v,
        "empates": empates,
        "pct_mandante": round((vitorias_m / total) * 100, 1),
        "pct_visitante": round((vitorias_v / total) * 100, 1),
        "pct_empate": round((empates / total) * 100, 1),
        "mensagem": f"{total} jogos históricos registrados."
    }

def obter_ultimo_confronto(mandante, visitante):
    if df_historico is None:
        return None
    mask_direto = (df_historico["mandante"] == mandante) & (df_historico["visitante"] == visitante)
    mask_inverso = (df_historico["mandante"] == visitante) & (df_historico["visitante"] == mandante)
    jogos = df_historico[mask_direto | mask_inverso]
    if len(jogos) == 0:
        return None
    ultimo = jogos.iloc[-1]
    res_map = {
        "vitoria_mandante": f"Vitória de {ultimo['mandante']}",
        "empate": "Empate",
        "vitoria_visitante": f"Vitória de {ultimo['visitante']}"
    }
    return {
        "data": str(ultimo["data"]),
        "mandante": str(ultimo["mandante"]),
        "visitante": str(ultimo["visitante"]),
        "torneio": str(ultimo["torneio"]),
        "desfecho": res_map.get(ultimo["resultado"], str(ultimo["resultado"]))
    }

def calcular_placar_e_gols(diff_elo, prob_m_modelo, prob_v_modelo):
    exp_factor = 1.0 / (1.0 + 10.0 ** (-diff_elo / 400.0))
    lambda_m = max(0.4, 1.35 * exp_factor * 2.0)
    lambda_v = max(0.4, 1.15 * (1.0 - exp_factor) * 2.0)

    best_p = 0.0
    best_score = (1, 1)
    prob_under_2_5 = 0.0

    for i in range(6):
        p_i = (lambda_m ** i) * math.exp(-lambda_m) / math.factorial(i)
        for j in range(6):
            p_j = (lambda_v ** j) * math.exp(-lambda_v) / math.factorial(j)
            p_joint = p_i * p_j
            if i + j <= 2:
                prob_under_2_5 += p_joint
            if p_joint > best_p:
                best_p = p_joint
                best_score = (i, j)

    pct_under = round(prob_under_2_5 * 100, 1)
    faixa_txt = f"Menos de 2.5 gols ({pct_under}%)" if pct_under >= 50.0 else f"Mais de 2.5 gols ({round(100 - pct_under, 1)}%)"

    return {
        "placar": f"{best_score[0]} x {best_score[1]}",
        "prob_placar_pct": round(best_p * 100, 1),
        "faixa_gols": faixa_txt,
        "prob_under_2_5_pct": pct_under
    }

def calcular_termometro_zebra(diff_elo, prob_m, prob_v):
    saldo = abs(diff_elo)
    if saldo < 45:
        return {
            "nivel": "Confronto de Alto Equilíbrio",
            "detalhe": "Diferencial de força inferior a 45 pontos Elo. Cenário de jogo truncado com elevada chance de empate.",
            "cor": "amber"
        }
    if saldo > 130:
        favorito = "Mandante" if diff_elo > 0 else "Visitante"
        return {
            "nivel": "Favoritismo Consolidado",
            "detalhe": f"Vantagem estatística ampla superior a 130 pontos Elo a favor do {favorito}. Baixo risco de zebra esportiva.",
            "cor": "emerald"
        }
    return {
        "nivel": "Alerta de Zebra Possível",
        "detalhe": "Há um favorito pontual, porém a equipe adversária possui fatores de mando ou momento suficientes para surpreender.",
        "cor": "sky"
    }

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/opcoes", methods=["GET"])
def get_opcoes():
    return jsonify({
        "selecoes": todas_selecoes,
        "torneios": todos_torneios,
        "confederacoes": MAPA_CONFEDERACOES,
        "torneios_restritos": TORNEIOS_RESTRITOS,
        "versao_modelo": "2.0.0" if usando_v2 else "1.0.0"
    })

@app.route("/api/prever", methods=["POST"])
def prever():
    try:
        dados = request.get_json(silent=True)
        if not dados:
            return jsonify({
                "status": "invalido",
                "mensagem": "Formato de requisição inválido. Envie um JSON com os campos da partida."
            }), 400

        mandante = str(dados.get("mandante", "")).strip()
        visitante = str(dados.get("visitante", "")).strip()
        torneio = str(dados.get("torneio", "")).strip()
        campo_neutro = str(dados.get("campo_neutro", "")).strip()

        erros = []
        if not mandante:
            erros.append("Selecione a seleção mandante.")
        if not visitante:
            erros.append("Selecione a seleção visitante.")
        if mandante and visitante and mandante.lower() == visitante.lower():
            erros.append("A seleção mandante e visitante não podem ser a mesma.")
        if not torneio:
            erros.append("Informe o torneio ou competição.")
        if campo_neutro not in ("Sim", "Nao"):
            erros.append("Informe se a partida é em campo neutro ('Sim' ou 'Nao').")

        if erros:
            return jsonify({
                "status": "invalido",
                "erros": erros,
                "mensagem": "Verifique os dados informados para prosseguir."
            }), 422

        erro_geo = validar_torneio_geografico(mandante, visitante, torneio)
        if erro_geo:
            return jsonify({
                "status": "invalido",
                "erros": [erro_geo],
                "mensagem": erro_geo
            }), 422

        info_m = estado_selecoes.get(mandante, {"elo": 1500.0, "pontos_ultimos_5": 7, "ultimos_5_jogos": ["E", "V", "D", "V", "E"]})
        info_v = estado_selecoes.get(visitante, {"elo": 1500.0, "pontos_ultimos_5": 7, "ultimos_5_jogos": ["E", "V", "D", "V", "E"]})

        elo_m = float(info_m.get("elo", 1500.0))
        elo_v = float(info_v.get("elo", 1500.0))
        bonus_casa = 70.0 if campo_neutro == "Nao" else 0.0
        diff_elo = (elo_m - elo_v) + bonus_casa

        forma_m = int(info_m.get("pontos_ultimos_5", 7))
        forma_v = int(info_v.get("pontos_ultimos_5", 7))
        diff_forma = forma_m - forma_v

        if usando_v2:
            df_in = pd.DataFrame([{
                "diff_elo": diff_elo,
                "diff_forma": diff_forma,
                "elo_mandante": elo_m,
                "elo_visitante": elo_v,
                "torneio": torneio,
                "campo_neutro": campo_neutro
            }])
        else:
            df_in = pd.DataFrame([{
                "mandante": mandante,
                "visitante": visitante,
                "torneio": torneio,
                "campo_neutro": campo_neutro
            }])

        classes = list(modelo.classes_)
        pred = modelo.predict(df_in)[0]
        probas = modelo.predict_proba(df_in)[0]

        mapa_probas = {classe: float(prob) for classe, prob in zip(classes, probas)}
        prob_m = mapa_probas.get("vitoria_mandante", 0.0)
        prob_e = mapa_probas.get("empate", 0.0)
        prob_v = mapa_probas.get("vitoria_visitante", 0.0)

        hist = obter_estatisticas_historico(mandante, visitante)
        ultimo_jogo = obter_ultimo_confronto(mandante, visitante)
        projeção_gols = calcular_placar_e_gols(diff_elo, prob_m, prob_v)
        termometro = calcular_termometro_zebra(diff_elo, prob_m, prob_v)

        titulos_resultado = {
            "vitoria_mandante": f"Vitória de {mandante}",
            "empate": "Previsão de Empate",
            "vitoria_visitante": f"Vitória de {visitante}"
        }

        valores_ordenados = sorted([prob_m, prob_e, prob_v], reverse=True)
        diferenca_topo = valores_ordenados[0] - valores_ordenados[1]
        eh_inconclusivo = diferenca_topo < 0.035 and hist["total"] == 0

        fatores = []
        if campo_neutro == "Sim":
            fatores.append("Condição de Campo Neutro: O bônus estatístico de mando de campo (+70 Elo) foi neutralizado.")
        else:
            fatores.append(f"Mando de Campo: {mandante} conta com a vantagem empírica do fator casa (+70 Elo computados).")

        if abs(elo_m - elo_v) < 50:
            fatores.append(f"Força técnica nivelada: {mandante} ({elo_m:.0f} Elo) vs {visitante} ({elo_v:.0f} Elo).")
        elif elo_m > elo_v:
            fatores.append(f"Superioridade no Ranking Elo: {mandante} ({elo_m:.0f} Elo) supera {visitante} ({elo_v:.0f} Elo) em {abs(elo_m - elo_v):.0f} pontos.")
        else:
            fatores.append(f"Superioridade no Ranking Elo: {visitante} ({elo_v:.0f} Elo) supera {mandante} ({elo_m:.0f} Elo) em {abs(elo_m - elo_v):.0f} pontos.")

        if diff_forma > 3:
            fatores.append(f"Momento favorável para {mandante}: somou {forma_m} pontos nos últimos 5 jogos contra {forma_v} de {visitante}.")
        elif diff_forma < -3:
            fatores.append(f"Momento favorável para {visitante}: somou {forma_v} pontos nos últimos 5 jogos contra {forma_m} de {mandante}.")
        else:
            fatores.append(f"Momento recente similar: {mandante} ({forma_m} pts) e {visitante} ({forma_v} pts) nos últimos 5 confrontos.")

        if hist["total"] > 0:
            fatores.append(f"Retrospecto direto: {hist['total']} partidas oficiais registradas entre ambas as seleções.")
        else:
            fatores.append("Inédito na base: Sem confrontos diretos prévios catalogados entre as seleções.")

        desfecho_txt = titulos_resultado.get(pred, str(pred))
        p_top_val = round(mapa_probas.get(pred, 0.0) * 100, 1)

        if abs(elo_m - elo_v) >= 50:
            time_maior_elo = mandante if elo_m > elo_v else visitante
            gap_elo = abs(elo_m - elo_v)
            frase_elo = f"O modelo detectou vantagem de {gap_elo:.0f} pontos de Rating Elo a favor de {time_maior_elo}."
        else:
            frase_elo = "O confronto apresenta alto equilíbrio de força técnica no ranking Elo."

        if campo_neutro == "Nao":
            frase_campo = f"O fator casa confere bônus empírico de +70 pontos a favor de {mandante}."
        else:
            frase_campo = "A condição de campo neutro anula o bônus tradicional de mando de campo."

        if hist["total"] > 0:
            frase_hist = f"No retrospecto direto de {hist['total']} confrontos, constam {hist['vitorias_mandante']} vitórias de {mandante}, {hist['vitorias_visitante']} de {visitante} e {hist['empates']} empates."
        else:
            frase_hist = "Não há registro de confrontos diretos prévios entre ambas na base oficial."

        sintese_ia = (
            f"O modelo projeta como desfecho mais provável a {desfecho_txt} ({p_top_val}%). "
            f"{frase_elo} {frase_campo} No momento recente, {mandante} acumulou {forma_m} pontos nos últimos 5 jogos contra {forma_v} pontos de {visitante}. "
            f"{frase_hist} Com base na combinação desses fatores, o placar estatístico mais provável é {projeção_gols['placar']}, com tendência de {projeção_gols['faixa_gols']}."
        )

        resposta = {
            "status": "inconclusivo" if eh_inconclusivo else "sucesso",
            "versao_modelo": "2.0.0 (RandomForest com Elo e Momento)" if usando_v2 else "1.0.0 (Baseline)",
            "resultado_classe": str(pred),
            "resultado_formatado": titulos_resultado.get(pred, str(pred)),
            "sintese_ia": sintese_ia,
            "probabilidades": {
                "vitoria_mandante": round(prob_m * 100, 1),
                "empate": round(prob_e * 100, 1),
                "vitoria_visitante": round(prob_v * 100, 1)
            },
            "prob_maior": round(mapa_probas.get(pred, 0.0) * 100, 1),
            "diferenca_topo_pct": round(diferenca_topo * 100, 1),
            "eh_inconclusivo": eh_inconclusivo,
            "motivo_inconclusivo": "A margem estatística entre os dois desfechos mais prováveis é inferior a 3.5% e não há histórico de confronto direto.",
            "partida": {
                "mandante": mandante,
                "visitante": visitante,
                "torneio": torneio,
                "campo_neutro": campo_neutro
            },
            "metricas_confronto": {
                "elo_mandante": elo_m,
                "elo_visitante": elo_v,
                "diff_elo": round(diff_elo, 1),
                "forma_mandante": {
                    "pontos": forma_m,
                    "sequencia": info_m.get("ultimos_5_jogos", [])
                },
                "forma_visitante": {
                    "pontos": forma_v,
                    "sequencia": info_v.get("ultimos_5_jogos", [])
                }
            },
            "historico": hist,
            "ultimo_confronto": ultimo_jogo,
            "projecao_gols": projeção_gols,
            "termometro_zebra": termometro,
            "fatores": fatores,
            "nota_metodologica": "Modelo RandomForest v2.0 treinado com Rating Elo acumulado e Momentum recente (F1-Macro de 0.5039 em teste, superando em +20% o baseline inicial)."
        }
        return jsonify(resposta), 200

    except Exception as e:
        return jsonify({
            "status": "falha",
            "mensagem": f"Ocorreu um erro no processamento do modelo: {str(e)}",
            "orientacao": "Tente novamente ou selecione outro confronto para análise."
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
