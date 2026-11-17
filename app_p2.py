import tkinter as tk
from tkinter import ttk
from pathlib import Path
import json
import threading
import time
import pandas as pd
import joblib

class AppPreditorFutebol(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Preditor de Futebol Internacional — P2 Machine Learning")
        self.geometry("1140x750")
        self.minsize(1050, 700)

        self.COLOR_BG = "#1e1e2e"
        self.COLOR_CARD = "#2a2a3c"
        self.COLOR_CARD_BORDER = "#3b3b54"
        self.COLOR_TEXT = "#cdd6f4"
        self.COLOR_TEXT_MUTED = "#9399b2"
        self.COLOR_ACCENT = "#00f5d4"
        self.COLOR_PURPLE = "#b4befe"
        self.COLOR_SUCCESS = "#a6e3a1"
        self.COLOR_WARNING = "#f9e2af"
        self.COLOR_BLUE = "#89b4fa"

        self.configure(bg=self.COLOR_BG)
        self._stream_id = 0

        self._carregar_recursos()
        self._configurar_estilos()
        self._construir_interface()

    def _carregar_recursos(self):
        with open("config.projeto.json", "r", encoding="utf-8") as f:
            self.config = json.load(f)

        modelo_path = Path(self.config["saida"]) / "modelo_inicial.joblib"
        if not modelo_path.exists():
            raise FileNotFoundError("Modelo nao encontrado em artefatos_v1/modelo_inicial.joblib")
        self.modelo = joblib.load(modelo_path)

        csv_path = Path(self.config["csv"])
        if csv_path.exists():
            self.df_historico = pd.read_csv(csv_path)
            todas_selecoes = sorted(list(set(self.df_historico["mandante"].dropna().unique().tolist() + 
                                            self.df_historico["visitante"].dropna().unique().tolist())))
            self.lista_selecoes = todas_selecoes
            contagem_torneios = self.df_historico["torneio"].value_counts().index.tolist()
            self.lista_torneios = contagem_torneios
        else:
            self.df_historico = None
            self.lista_selecoes = ["Brazil", "Argentina", "France", "Germany", "England", "Spain", "Italy", "Uruguay"]
            self.lista_torneios = ["FIFA World Cup", "Copa América", "UEFA Euro", "Friendly", "FIFA World Cup qualification"]

    def _configurar_estilos(self):
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure(".", background=self.COLOR_BG, foreground=self.COLOR_TEXT)
        self.style.configure("Card.TFrame", background=self.COLOR_CARD, relief="flat")
        self.style.configure("Header.TLabel", font=("Segoe UI", 18, "bold"), foreground=self.COLOR_ACCENT, background=self.COLOR_BG)
        self.style.configure("SubHeader.TLabel", font=("Segoe UI", 10), foreground=self.COLOR_TEXT_MUTED, background=self.COLOR_BG)
        self.style.configure("CardTitle.TLabel", font=("Segoe UI", 12, "bold"), foreground=self.COLOR_PURPLE, background=self.COLOR_CARD)
        self.style.configure("FieldLabel.TLabel", font=("Segoe UI", 10, "bold"), foreground=self.COLOR_TEXT, background=self.COLOR_CARD)
        self.style.configure("TCombobox", fieldbackground="#313244", background="#45475a", foreground="#ffffff", borderwidth=0)
        self.style.map("TCombobox", fieldbackground=[("readonly", "#313244")], foreground=[("readonly", "#ffffff")])
        self.style.configure("Primary.TButton", font=("Segoe UI", 11, "bold"), background="#00b4d8", foreground="#000000", padding=10, borderwidth=0)
        self.style.map("Primary.TButton", background=[("active", "#90e0ef")])
        self.style.configure("Secondary.TButton", font=("Segoe UI", 9), background="#45475a", foreground="#ffffff", padding=6, borderwidth=0)
        self.style.map("Secondary.TButton", background=[("active", "#585b70")])
        self.style.configure("TProgressbar", thickness=8, troughcolor="#313244", background=self.COLOR_ACCENT)

    def _criar_campo_combo(self, parent, label_text, values, default_val):
        ttk.Label(parent, text=label_text, style="FieldLabel.TLabel").pack(anchor="w", pady=(5, 2))
        cb = ttk.Combobox(parent, state="normal", font=("Segoe UI", 10), values=values)
        if default_val in values:
            cb.set(default_val)
        elif len(values) > 0:
            cb.set(values[0])
        cb.pack(fill="x", pady=(0, 12))
        return cb

    def _construir_interface(self):
        top_frame = tk.Frame(self, bg=self.COLOR_BG, pady=15, padx=25)
        top_frame.pack(fill="x")
        ttk.Label(top_frame, text="⚽ Preditor Inteligente de Partidas de Futebol", style="Header.TLabel").pack(anchor="w")
        ttk.Label(top_frame, text="Classificação probabilística pré-jogo baseada no modelo de Machine Learning (P2).", style="SubHeader.TLabel").pack(anchor="w")

        main_container = tk.Frame(self, bg=self.COLOR_BG, padx=25, pady=10)
        main_container.pack(fill="both", expand=True)
        main_container.columnconfigure(0, weight=1, uniform="group1")
        main_container.columnconfigure(1, weight=1, uniform="group1")
        main_container.rowconfigure(0, weight=1)

        left_card = tk.Frame(main_container, bg=self.COLOR_CARD, highlightbackground=self.COLOR_CARD_BORDER, highlightthickness=1, padx=20, pady=20)
        left_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        ttk.Label(left_card, text="⚙️ Dados do Confronto", style="CardTitle.TLabel").pack(anchor="w", pady=(0, 15))

        default_mandante = "Brazil" if "Brazil" in self.lista_selecoes else self.lista_selecoes[0]
        default_visitante = "Argentina" if "Argentina" in self.lista_selecoes else self.lista_selecoes[1]
        default_torneio = "Copa América" if "Copa América" in self.lista_torneios else self.lista_torneios[0]

        self.cb_mandante = self._criar_campo_combo(left_card, "Seleção Mandante (Casa):", self.lista_selecoes, default_mandante)
        self.cb_visitante = self._criar_campo_combo(left_card, "Seleção Visitante (Fora):", self.lista_selecoes, default_visitante)
        self.cb_torneio = self._criar_campo_combo(left_card, "Torneio / Competição:", self.lista_torneios, default_torneio)
        self.cb_neutro = self._criar_campo_combo(left_card, "Partida em Campo Neutro?", ["Nao", "Sim"], "Nao")

        btn_box = tk.Frame(left_card, bg=self.COLOR_CARD)
        btn_box.pack(fill="x", side="bottom", pady=(15, 0))

        self.btn_prever = ttk.Button(btn_box, text="⚡ Prever Resultado (Machine Learning)", style="Primary.TButton", command=self._iniciar_predicao)
        self.btn_prever.pack(fill="x", pady=(0, 6))

        btn_grid = tk.Frame(btn_box, bg=self.COLOR_CARD)
        btn_grid.pack(fill="x")
        btn_grid.columnconfigure(0, weight=1)
        btn_grid.columnconfigure(1, weight=1)

        self.btn_inverter = ttk.Button(btn_grid, text="⇄ Inverter Mando", style="Secondary.TButton", command=self._inverter_mando)
        self.btn_inverter.grid(row=0, column=0, sticky="ew", padx=(0, 4))

        self.btn_resetar = ttk.Button(btn_grid, text="🔄 Resetar", style="Secondary.TButton", command=self._resetar_campos)
        self.btn_resetar.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        self.right_card = tk.Frame(main_container, bg=self.COLOR_CARD, highlightbackground=self.COLOR_CARD_BORDER, highlightthickness=1, padx=20, pady=20)
        self.right_card.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        ttk.Label(self.right_card, text="🎯 Diagnóstico e Probabilidades da IA", style="CardTitle.TLabel").pack(anchor="w", pady=(0, 10))

        self.progress = ttk.Progressbar(self.right_card, style="TProgressbar", mode="indeterminate")
        self.res_container = tk.Frame(self.right_card, bg=self.COLOR_CARD)
        self.res_container.pack(fill="both", expand=True)
        self._exibir_estado_inicial()

    def _exibir_estado_inicial(self):
        for widget in self.res_container.winfo_children():
            widget.destroy()
        tk.Label(
            self.res_container,
            text="Selecione as seleções e o torneio à esquerda e clique em\n'Prever Resultado' para consultar o modelo de Machine Learning.",
            font=("Segoe UI", 11, "italic"), bg=self.COLOR_CARD, fg=self.COLOR_TEXT_MUTED, justify="center"
        ).pack(expand=True)

    def _inverter_mando(self):
        mandante = self.cb_mandante.get()
        visitante = self.cb_visitante.get()
        self.cb_mandante.set(visitante)
        self.cb_visitante.set(mandante)

    def _resetar_campos(self):
        self._stream_id += 1
        self.progress.stop()
        self.progress.pack_forget()
        self.btn_prever.config(state="normal")
        if "Brazil" in self.lista_selecoes:
            self.cb_mandante.set("Brazil")
        if "Argentina" in self.lista_selecoes:
            self.cb_visitante.set("Argentina")
        if "Copa América" in self.lista_torneios:
            self.cb_torneio.set("Copa América")
        self.cb_neutro.set("Nao")
        self._exibir_estado_inicial()

    def _exibir_erro_validacao(self, mensagem):
        self.progress.stop()
        self.progress.pack_forget()
        self.btn_prever.config(state="normal")
        for widget in self.res_container.winfo_children():
            widget.destroy()
        box_erro = tk.Frame(self.res_container, bg="#3b1d28", padx=15, pady=12, highlightbackground="#f38ba8", highlightthickness=1)
        box_erro.pack(fill="x", pady=10)
        tk.Label(box_erro, text="⚠️ Entrada Inválida", font=("Segoe UI", 12, "bold"), bg="#3b1d28", fg="#f38ba8").pack(anchor="w")
        tk.Label(box_erro, text=mensagem, font=("Segoe UI", 10), bg="#3b1d28", fg="#cdd6f4", justify="left", wraplength=420).pack(anchor="w", pady=(4, 0))

    def _iniciar_predicao(self):
        mandante = self.cb_mandante.get().strip()
        visitante = self.cb_visitante.get().strip()
        torneio = self.cb_torneio.get().strip()
        neutro = self.cb_neutro.get().strip()

        if not mandante or not visitante:
            self._exibir_erro_validacao("Selecione ambas as seleções para continuar.")
            return

        if mandante == visitante:
            self._exibir_erro_validacao(f"A seleção mandante e visitante não podem ser a mesma ({mandante}). Escolha seleções diferentes.")
            return

        if not torneio:
            self._exibir_erro_validacao("Selecione uma competição válida.")
            return

        self._stream_id += 1
        self.btn_prever.config(state="disabled")
        self.progress.pack(fill="x", pady=(0, 15))
        self.progress.start(10)
        threading.Thread(target=self._executar_predicao_async, args=(mandante, visitante, torneio, neutro), daemon=True).start()

    def _executar_predicao_async(self, mandante, visitante, torneio, neutro):
        try:
            time.sleep(0.3)
            df_in = pd.DataFrame([{
                "mandante": mandante,
                "visitante": visitante,
                "torneio": torneio,
                "campo_neutro": neutro
            }])

            classes = list(self.modelo.classes_)
            pred = self.modelo.predict(df_in)[0]
            probas = self.modelo.predict_proba(df_in)[0]

            mapa_probas = {classe: float(prob) for classe, prob in zip(classes, probas)}
            hist_info = self._obter_historico(mandante, visitante)

            diagnostico = {
                "mandante": mandante,
                "visitante": visitante,
                "torneio": torneio,
                "campo_neutro": neutro,
                "predicao": str(pred),
                "probas": mapa_probas,
                "historico": hist_info
            }

            self.after(0, lambda: self._iniciar_streaming_resultado(diagnostico, self._stream_id))
        except Exception as e:
            self.after(0, lambda: self._exibir_erro_validacao(f"Falha técnica ao executar modelo: {str(e)}"))

    def _obter_historico(self, mandante, visitante):
        if self.df_historico is None:
            return None
        mask_direto = (self.df_historico["mandante"] == mandante) & (self.df_historico["visitante"] == visitante)
        mask_inverso = (self.df_historico["mandante"] == visitante) & (self.df_historico["visitante"] == mandante)
        todos_jogos = self.df_historico[mask_direto | mask_inverso]
        total = len(todos_jogos)
        if total == 0:
            return {"total": 0, "detalhe": "Nenhum confronto histórico registrado nesta base de dados."}

        vits_m = len(self.df_historico[(self.df_historico["mandante"] == mandante) & (self.df_historico["visitante"] == visitante) & (self.df_historico["resultado"] == "vitoria_mandante")]) + \
                 len(self.df_historico[(self.df_historico["mandante"] == visitante) & (self.df_historico["visitante"] == mandante) & (self.df_historico["resultado"] == "vitoria_visitante")])
        vits_v = len(self.df_historico[(self.df_historico["mandante"] == mandante) & (self.df_historico["visitante"] == visitante) & (self.df_historico["resultado"] == "vitoria_visitante")]) + \
                 len(self.df_historico[(self.df_historico["mandante"] == visitante) & (self.df_historico["visitante"] == mandante) & (self.df_historico["resultado"] == "vitoria_mandante")])
        empates = len(todos_jogos[todos_jogos["resultado"] == "empate"])

        return {
            "total": total,
            "vits_m": vits_m,
            "vits_v": vits_v,
            "empates": empates
        }

    def _gerar_efeito_digitacao(self, widget: tk.Label, texto: str, delay: int = 25, chunk_size: int = 1):
        for i in range(chunk_size, len(texto) + chunk_size, chunk_size):
            widget.config(text=texto[:i] + "▌")
            yield delay
        widget.config(text=texto)

    def _desenhar_barra_probabilidade(self, parent, prob_m, prob_e, prob_v):
        canvas = tk.Canvas(parent, height=22, bg="#313244", highlightthickness=0)
        canvas.pack(fill="x", pady=(4, 8))

        def desenhar(event=None):
            w = canvas.winfo_width()
            if w <= 1:
                return
            canvas.delete("all")
            w_m = int(w * prob_m)
            w_e = int(w * prob_e)
            w_v = w - w_m - w_e
            canvas.create_rectangle(0, 0, w_m, 22, fill=self.COLOR_SUCCESS, outline="")
            canvas.create_rectangle(w_m, 0, w_m + w_e, 22, fill=self.COLOR_WARNING, outline="")
            canvas.create_rectangle(w_m + w_e, 0, w, 22, fill=self.COLOR_BLUE, outline="")

        canvas.bind("<Configure>", desenhar)
        self.after(50, desenhar)

    def _iniciar_streaming_resultado(self, diag: dict, stream_id: int):
        if stream_id != self._stream_id:
            return
        self.progress.stop()
        self.progress.pack_forget()
        for widget in self.res_container.winfo_children():
            widget.destroy()

        def stream_generator():
            pred = diag["predicao"]
            titulos = {
                "vitoria_mandante": f"Vitória de {diag['mandante']} (Mandante)",
                "empate": "Previsão de Empate",
                "vitoria_visitante": f"Vitória de {diag['visitante']} (Visitante)"
            }
            cores = {
                "vitoria_mandante": self.COLOR_SUCCESS,
                "empate": self.COLOR_WARNING,
                "vitoria_visitante": self.COLOR_BLUE
            }

            p_m = diag["probas"].get("vitoria_mandante", 0.0)
            p_e = diag["probas"].get("empate", 0.0)
            p_v = diag["probas"].get("vitoria_visitante", 0.0)
            p_top = diag["probas"].get(pred, 0.0) * 100.0

            valores_ordenados = sorted([p_m, p_e, p_v], reverse=True)
            diff_topo = valores_ordenados[0] - valores_ordenados[1]
            eh_inconclusivo = (diff_topo < 0.04) and (not diag["historico"] or diag["historico"]["total"] == 0)

            if eh_inconclusivo:
                box_inc = tk.Frame(self.res_container, bg="#3b2d18", padx=12, pady=8, highlightbackground="#f9e2af", highlightthickness=1)
                box_inc.pack(fill="x", pady=(0, 10))
                lbl_inc = tk.Label(box_inc, text="", font=("Segoe UI", 9, "bold"), bg="#3b2d18", fg="#f9e2af", wraplength=430, justify="left")
                lbl_inc.pack(anchor="w")
                yield from self._gerar_efeito_digitacao(lbl_inc, "⚠️ Alerta: Alta incerteza estatística (diferença inferior a 4% entre desfechos e sem histórico direto prévio).", delay=15)

            box_destaque = tk.Frame(self.res_container, bg="#1e1e2e", padx=15, pady=12, highlightbackground=cores.get(pred, self.COLOR_ACCENT), highlightthickness=1)
            box_destaque.pack(fill="x", pady=(0, 10))

            lbl_pred = tk.Label(box_destaque, text="", font=("Segoe UI", 15, "bold"), bg="#1e1e2e", fg=cores.get(pred, self.COLOR_ACCENT), justify="left", wraplength=430)
            lbl_pred.pack(anchor="w", fill="x")
            yield from self._gerar_efeito_digitacao(lbl_pred, f"Resultado Previsto: {titulos.get(pred, pred)}", delay=30)

            lbl_sub = tk.Label(box_destaque, text="", font=("Segoe UI", 9, "bold"), bg="#1e1e2e", fg=self.COLOR_TEXT_MUTED)
            lbl_sub.pack(anchor="w", pady=(2, 4))
            yield from self._gerar_efeito_digitacao(lbl_sub, f"Probabilidade Estimada: {p_top:.1f}% | Torneio: {diag['torneio']}", delay=20)

            lbl_tit_probs = tk.Label(self.res_container, text="📊 Distribuição de Probabilidades Estimadas", font=("Segoe UI", 10, "bold"), bg=self.COLOR_CARD, fg=self.COLOR_PURPLE)
            lbl_tit_probs.pack(anchor="w", pady=(4, 2))
            yield 100

            frame_bar = tk.Frame(self.res_container, bg=self.COLOR_CARD)
            frame_bar.pack(fill="x", pady=(0, 6))
            self._desenhar_barra_probabilidade(frame_bar, p_m, p_e, p_v)

            frame_leg = tk.Frame(self.res_container, bg=self.COLOR_CARD)
            frame_leg.pack(fill="x", pady=(0, 8))
            frame_leg.columnconfigure(0, weight=1)
            frame_leg.columnconfigure(1, weight=1)
            frame_leg.columnconfigure(2, weight=1)

            lbl_lm = tk.Label(frame_leg, text="", font=("Segoe UI", 9, "bold"), bg=self.COLOR_CARD, fg=self.COLOR_SUCCESS)
            lbl_lm.grid(row=0, column=0, sticky="w")
            yield from self._gerar_efeito_digitacao(lbl_lm, f"Mandante: {p_m*100:.1f}%", delay=18)

            lbl_le = tk.Label(frame_leg, text="", font=("Segoe UI", 9, "bold"), bg=self.COLOR_CARD, fg=self.COLOR_WARNING)
            lbl_le.grid(row=0, column=1, sticky="n")
            yield from self._gerar_efeito_digitacao(lbl_le, f"Empate: {p_e*100:.1f}%", delay=18)

            lbl_lv = tk.Label(frame_leg, text="", font=("Segoe UI", 9, "bold"), bg=self.COLOR_CARD, fg=self.COLOR_BLUE)
            lbl_lv.grid(row=0, column=2, sticky="e")
            yield from self._gerar_efeito_digitacao(lbl_lv, f"Visitante: {p_v*100:.1f}%", delay=18)

            hist = diag["historico"]
            lbl_tit_hist = tk.Label(self.res_container, text="📜 Histórico Geral no Dataset (1872 - 2024)", font=("Segoe UI", 10, "bold"), bg=self.COLOR_CARD, fg=self.COLOR_TEXT)
            lbl_tit_hist.pack(anchor="w", pady=(6, 2))
            yield 100

            box_hist = tk.Frame(self.res_container, bg="#1e1e2e", padx=12, pady=8, highlightbackground=self.COLOR_CARD_BORDER, highlightthickness=1)
            box_hist.pack(fill="x", pady=(0, 10))

            if hist and hist["total"] > 0:
                texto_hist = f"Total de confrontos diretos registrados: {hist['total']} partidas\n" \
                             f"• Vitórias de {diag['mandante']}: {hist['vits_m']} ({(hist['vits_m']/hist['total'])*100:.1f}%)\n" \
                             f"• Vitórias de {diag['visitante']}: {hist['vits_v']} ({(hist['vits_v']/hist['total'])*100:.1f}%)\n" \
                             f"• Empates: {hist['empates']} ({(hist['empates']/hist['total'])*100:.1f}%)"
            else:
                texto_hist = "Não há registro de confrontos diretos prévios entre essas seleções na base de dados."

            lbl_th = tk.Label(box_hist, text="", font=("Segoe UI", 9), bg="#1e1e2e", fg=self.COLOR_TEXT, justify="left")
            lbl_th.pack(anchor="w")
            yield from self._gerar_efeito_digitacao(lbl_th, texto_hist, delay=15)

            lbl_tit_diag = tk.Label(self.res_container, text="🧠 Fatores Técnicos Avaliados pelo Modelo", font=("Segoe UI", 10, "bold"), bg=self.COLOR_CARD, fg=self.COLOR_PURPLE)
            lbl_tit_diag.pack(anchor="w", pady=(4, 2))
            yield 100

            razoes = []
            if diag["campo_neutro"] == "Sim":
                razoes.append("Campo Neutro ativado: o efeito tradicional de pressão de torcida e mando de campo foi neutralizado no pipeline.")
            else:
                razoes.append(f"Fator Casa ativo: {diag['mandante']} conta com o efeito estatístico favorável de mando de campo.")

            diff = abs(p_m - p_v)
            if diff < 0.10:
                razoes.append("Confronto de alto equilíbrio: a diferença de probabilidade entre as seleções é estreita, indicando alta incerteza.")
            elif p_m > p_v + 0.25:
                razoes.append(f"Superioridade estatística acentuada a favor de {diag['mandante']} com base no histórico global.")
            elif p_v > p_m + 0.25:
                razoes.append(f"Superioridade estatística acentuada a favor de {diag['visitante']} mesmo atuando fora de seus domínios.")

            frame_rz = tk.Frame(self.res_container, bg=self.COLOR_CARD)
            frame_rz.pack(fill="x")
            for r in razoes:
                r_box = tk.Frame(frame_rz, bg=self.COLOR_CARD)
                r_box.pack(fill="x", anchor="w", pady=1)
                tk.Label(r_box, text="•", font=("Segoe UI", 10, "bold"), bg=self.COLOR_CARD, fg=self.COLOR_ACCENT).pack(side="left", padx=(0, 5))
                lbl_r = tk.Label(r_box, text="", font=("Segoe UI", 9), bg=self.COLOR_CARD, fg=self.COLOR_TEXT, justify="left", wraplength=420)
                lbl_r.pack(side="left", anchor="w")
                yield from self._gerar_efeito_digitacao(lbl_r, r, delay=20)
                yield 80

        gen = stream_generator()
        self._avancar_stream(gen, stream_id)

    def _avancar_stream(self, gen, stream_id: int):
        if stream_id != self._stream_id:
            return
        try:
            delay = next(gen)
            self.after(delay, lambda: self._avancar_stream(gen, stream_id))
        except StopIteration:
            self.btn_prever.config(state="normal")

if __name__ == "__main__":
    app = AppPreditorFutebol()
    app.mainloop()

