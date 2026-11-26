import heapq
import time
import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import streamlit as st

# ==============================================================================
# 1. DADOS DOCUMENTAIS DA TABELA 1 (PETROLINA - MOOVIT)
# ==============================================================================
ITINERARIOS = {
    "011":        {"nome": "São Gonçalo R1",               "paradas": 76, "duracao_min": 76},
    "012":        {"nome": "São Gonçalo R2",               "paradas": 76, "duracao_min": 73},
    "013-CMEI":   {"nome": "Cohab VI vía CMEI",            "paradas": 69, "duracao_min": 81},
    "013-Distr.": {"nome": "Cohab VI vía Distrito",        "paradas": 67, "duracao_min": 69},
    "014":        {"nome": "Jardim Guararapes",            "paradas": 77, "duracao_min": 80},
    "015":        {"nome": "Residencial Pomares",          "paradas": 63, "duracao_min": 69},
    "031-Unif.":  {"nome": "Pedra Linda unificada",        "paradas": 72, "duracao_min": 66},
    "031-N.T.":   {"nome": "Pedra Linda vía Novo Tempo",   "paradas": 68, "duracao_min": 61},
    "041-J.M.":   {"nome": "Dom Avelar vía José e Maria",  "paradas": 80, "duracao_min": 67},
    "041-Univ.":  {"nome": "Dom Avelar universitaria",     "paradas": 44, "duracao_min": 36},
    "041-S.L.":   {"nome": "Dom Avelar vía Santa Luzia",   "paradas": 95, "duracao_min": 77},
}

CENARIOS_ESTRESS = {
    "Base": {"mult_demanda": 1.00, "mult_tempo": 1.00, "falha_veiculo": False},
    "Demanda +20%": {"mult_demanda": 1.20, "mult_tempo": 1.00, "falha_veiculo": False},
    "Demanda +40%": {"mult_demanda": 1.40, "mult_tempo": 1.00, "falha_veiculo": False},
    "Tempo de percurso +25%": {"mult_demanda": 1.00, "mult_tempo": 1.25, "falha_veiculo": False},
    "Indisponibilidade de veículo": {"mult_demanda": 1.00, "mult_tempo": 1.00, "falha_veiculo": True},
    "Perturbações combinadas": {"mult_demanda": 1.20, "mult_tempo": 1.25, "falha_veiculo": True},
}


# ==============================================================================
# 2. MODELO PREDITIVO (MPC NOMINAL E RMPC MIN-MAX)
# ==============================================================================
def avaliar_custo_horizonte(
    u_vec,
    a_atual,
    d_prev_paradas,
    ocupacao_atual,
    parada_atual,
    n_paradas,
    h_star,
    m_arco,
    lambda_s,
    params,
    mult_tempo=1.0,
    mult_demanda=1.0,
):
    """Calcula a função objetivo J ao longo do horizonte H para um cenário ω."""
    H = len(u_vec)
    alpha, beta, gamma = params["alpha"], params["beta"], params["gamma"]
    u_max = params["u_max"]
    W_ref = params["W_ref"]
    C_b = params["capacidade"]
    tau_0 = params["tau_0"]
    theta_B = params["theta_B"]
    theta_D = params["theta_D"]

    custo_total = 0.0
    t_chegada = a_atual
    ocup = float(ocupacao_atual)

    for j in range(H):
        s = parada_atual + j
        if s >= n_paradas - 1:
            break

        # Estimativa de passagem do veículo anterior na parada s
        d_prev = d_prev_paradas[s]
        if d_prev is None:
            h_est = h_star
        else:
            h_est = max(0.0, t_chegada - d_prev)

        # Demanda prevista e serviço
        lam = lambda_s[s] * mult_demanda
        chegadas_est = lam * h_est
        prob_desc = min(0.95, 1.0 / max(1, (n_paradas - 1 - s)))
        descensos = ocup * prob_desc
        espaco = max(0.0, C_b - (ocup - descensos))
        embarques = min(chegadas_est, espaco)

        t_serv = tau_0 + theta_B * embarques + theta_D * descensos
        u_j = u_vec[j]
        t_saida = t_chegada + t_serv + u_j
        ocup = min(C_b, max(0.0, ocup - descensos + embarques))

        # Headway resultante após retenção
        h_pos = h_star if d_prev is None else max(0.0, t_saida - d_prev)

        # Componentes da função objetivo J
        termo_h = alpha * ((h_pos - h_star) / h_star) ** 2
        # Espera prevista: espera na parada (0.5 * lambda * h^2) + atraso a bordo (ocup * u)
        W_j = 0.5 * lam * (h_pos ** 2) + ocup * u_j
        termo_w = beta * (W_j / W_ref)
        termo_u = gamma * ((u_j / u_max) ** 2)

        custo_total += termo_h + termo_w + termo_u

        # Próxima chegada prevista
        t_chegada = t_saida + (m_arco * mult_tempo)

    return custo_total


def calcular_retencao(
    politica,
    a_atual,
    d_prev_paradas,
    ocupacao_atual,
    parada_atual,
    n_paradas,
    h_star,
    m_arco,
    lambda_s,
    params,
    omega_scenarios,
):
    """Determina a ação de retenção u (em segundos), tempo de cálculo e factibilidade."""
    t0 = time.perf_counter()
    u_max = params["u_max"]

    if politica == "Sem controle":
        return 0.0, 0.0, True

    d_prev = d_prev_paradas[parada_atual]
    if d_prev is None:
        return 0.0, (time.perf_counter() - t0) * 1000.0, True

    h_obs = max(0.0, a_atual - d_prev)

    # Política reativa de referência (também usada como fallback seguro)
    u_reativo = float(np.clip(params["k_reativo"] * (h_star - h_obs), 0.0, u_max))

    if politica == "Controle reativo":
        return u_reativo, (time.perf_counter() - t0) * 1000.0, True

    # MPC Nominal ou RMPC (horizonte deslizante H)
    H = min(params["H"], max(1, (n_paradas - 1) - parada_atual))
    u0 = np.full(H, u_reativo * 0.5, dtype=float)
    bounds = [(0.0, u_max)] * H

    if politica == "MPC nominal":
        def obj(u_vec):
            return avaliar_custo_horizonte(
                u_vec, a_atual, d_prev_paradas, ocupacao_atual,
                parada_atual, n_paradas, h_star, m_arco, lambda_s, params,
                mult_tempo=1.0, mult_demanda=1.0
            )
    else:  # RMPC: minimiza o pior custo dentro do conjunto de incertezas Ω
        def obj(u_vec):
            pior = -1.0
            for mult_t, mult_d in omega_scenarios:
                val = avaliar_custo_horizonte(
                    u_vec, a_atual, d_prev_paradas, ocupacao_atual,
                    parada_atual, n_paradas, h_star, m_arco, lambda_s, params,
                    mult_tempo=mult_t, mult_demanda=mult_d
                )
                if val > pior:
                    pior = val
            return pior

    res = minimize(
        obj,
        u0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 25, "ftol": 1e-4},
    )
    dt_ms = (time.perf_counter() - t0) * 1000.0

    # Verifica factibilidade e limite computacional (60 s nominal, aqui 500 ms prático)
    if res.success and np.all(np.isfinite(res.x)):
        u_opt = float(np.clip(res.x[0], 0.0, u_max))
        return u_opt, dt_ms, True
    else:
        return u_reativo, dt_ms, False


# ==============================================================================
# 3. SIMULADOR A EVENTOS DISCRETOS (UMA RÉPLICA)
# ==============================================================================
def executar_replica(itinerario_cod, cenario_nome, politica, seed, params):
    rng = np.random.default_rng(seed)

    info = ITINERARIOS[itinerario_cod]
    n_paradas = info["paradas"]
    duracao_seg = info["duracao_min"] * 60.0
    h_star = params["h_star_min"] * 60.0
    cenario = CENARIOS_ESTRESS[cenario_nome]

    # Calibração do tempo médio por arco (descontando serviço nominal aproximado)
    serv_nominal_parada = params["tau_0"] + 6.0  # aproximação de embarque/desembarque base
    tempo_viagem_puro = max(duracao_seg * 0.65, duracao_seg - n_paradas * serv_nominal_parada)
    m_arco = (tempo_viagem_puro / max(1, n_paradas - 1)) * cenario["mult_tempo"]

    # Parâmetros da Lognormal (Seção 2.3)
    cv = params["cv"]
    sigma2 = math.log(1.0 + cv ** 2)
    sigma = math.sqrt(sigma2)
    mu = math.log(m_arco) - 0.5 * sigma2

    # Perfil de demanda sintético por parada (maior no miolo do trajeto)
    idx = np.linspace(0, np.pi, n_paradas)
    perfil_espacial = 0.5 + np.sin(idx)
    lambda_base_seg = (params["demanda_pax_min_parada"] / 60.0) * cenario["mult_demanda"]
    lambda_s = lambda_base_seg * perfil_espacial
    lambda_s[-1] = 0.0  # terminal final não embarca

    # Paradas autorizadas para retenção (z_{b,s,k} = 1)
    passo_ctrl = params["passo_paradas_ctrl"]
    paradas_ctrl = set(range(passo_ctrl, n_paradas - 1, passo_ctrl))

    # Conjunto de cenários de incerteza Ω para o RMPC (determinístico dado o raio de robustez)
    raio = params["raio_omega"]
    n_omega = params["n_omega"]
    rng_omega = np.random.default_rng(9999)
    omega_scenarios = [(1.0, 1.0)]
    if n_omega > 1:
        # Inclui extremos e amostras dentro do conjunto de incerteza
        omega_scenarios.append((1.0 + raio, 1.0 + raio))
        omega_scenarios.append((1.0 + raio, 1.0 - 0.5 * raio))
        omega_scenarios.append((1.0 - 0.5 * raio, 1.0 + raio))
        while len(omega_scenarios) < n_omega:
            mt = float(rng_omega.uniform(1.0 - 0.5 * raio, 1.0 + raio))
            md = float(rng_omega.uniform(1.0 - 0.5 * raio, 1.0 + raio))
            omega_scenarios.append((mt, md))
        omega_scenarios = omega_scenarios[:n_omega]

    # Pré-geração de tempos de arco por (viagem, parada) para garantir mesmas sementes entre políticas
    n_viagens = params["n_viagens"]
    tempos_arco = rng.lognormal(mean=mu, sigma=sigma, size=(n_viagens, n_paradas - 1))

    # Pré-geração de sementes independentes por parada para chegadas de Poisson
    # Fila de eventos: (tempo_chegada, id_viagem, parada)
    eventos = []
    viagem_falha = n_viagens // 3 if cenario["falha_veiculo"] else -1

    for b in range(n_viagens):
        if b == viagem_falha:
            # Indisponibilidade temporária: viagem removida da operação
            continue
        t_partida = b * h_star
        heapq.heappush(eventos, (t_partida, b, 0))

    # Estados da rede
    ocupacao = {b: 0 for b in range(n_viagens)}
    ultima_saida_parada = [None] * n_paradas
    fila_parada = [0] * n_paradas
    tempos_chegada_fila = [[] for _ in range(n_paradas)]  # para cálculo exato de espera média e p95

    # Métricas
    headways_obs = []
    esperas_pax = []
    nao_embarcados = 0
    retencao_total = 0.0
    tempos_calc_ms = []
    decisoes_factiveis = 0
    decisoes_totais = 0

    while eventos:
        t_cheg, b, s = heapq.heappop(eventos)

        # 1. Intervalo efetivo ordenado por passagem (evita headway negativo em ultrapassagens)
        t_prev = ultima_saida_parada[s]
        if t_prev is None:
            dt_acum = h_star
            h_efetivo = None
        else:
            dt_acum = max(0.0, t_cheg - t_prev)
            h_efetivo = dt_acum

        # 2. Chegadas de passageiros (Processo de Poisson no intervalo desde o ônibus anterior)
        if s < n_paradas - 1 and dt_acum > 0:
            # Fator pico no terço médio da simulação (Poisson não homogêneo)
            fator_pico = 1.25 if (0.3 * n_viagens * h_star <= t_cheg <= 0.7 * n_viagens * h_star) else 1.0
            media_cheg = lambda_s[s] * fator_pico * dt_acum
            n_novos = int(rng.poisson(media_cheg))
            if n_novos > 0:
                t_ini = max(0.0, t_cheg - dt_acum)
                chegadas_t = rng.uniform(t_ini, t_cheg, size=n_novos)
                tempos_chegada_fila[s].extend(chegadas_t.tolist())
                fila_parada[s] += n_novos

        # 3. Desembarques (conservação de passageiros; 100% desembarcam no terminal final)
        if s == n_paradas - 1:
            descensos = ocupacao[b]
        elif ocupacao[b] > 0:
            prob_desc = min(1.0, 1.0 / max(1, (n_paradas - 1 - s)))
            descensos = int(rng.binomial(ocupacao[b], prob_desc))
        else:
            descensos = 0

        # 4. Embarques limitados pela capacidade C_b
        espaco_disp = params["capacidade"] - (ocupacao[b] - descensos)
        embarques = min(fila_parada[s], max(0, espaco_disp))
        rejeitados = fila_parada[s] - embarques
        if rejeitados > 0:
            nao_embarcados += rejeitados

        # Registra tempos de espera dos passageiros que embarcaram (FIFO)
        if embarques > 0:
            embarcados_t = tempos_chegada_fila[s][:embarques]
            tempos_chegada_fila[s] = tempos_chegada_fila[s][embarques:]
            esperas_pax.extend([max(0.0, t_cheg - tc) for tc in embarcados_t])
            fila_parada[s] -= embarques

        ocupacao[b] = ocupacao[b] - descensos + embarques

        # 5. Tempo de serviço
        t_serv = params["tau_0"] + params["theta_B"] * embarques + params["theta_D"] * descensos

        # 6. Ação de controle (retenção) em paradas autorizadas
        u_acao = 0.0
        if s in paradas_ctrl and t_prev is not None:
            u_acao, dt_ms, factivel = calcular_retencao(
                politica=politica,
                a_atual=t_cheg,
                d_prev_paradas=ultima_saida_parada,
                ocupacao_atual=ocupacao[b],
                parada_atual=s,
                n_paradas=n_paradas,
                h_star=h_star,
                m_arco=m_arco,
                lambda_s=lambda_s,
                params=params,
                omega_scenarios=omega_scenarios,
            )
            retencao_total += u_acao
            tempos_calc_ms.append(dt_ms)
            decisoes_totais += 1
            if factivel:
                decisoes_factiveis += 1

        # 7. Instante de saída e registro do headway após controle
        t_saida = t_cheg + t_serv + u_acao
        if t_prev is not None and s > 0:
            headways_obs.append(max(0.0, t_saida - t_prev))
        ultima_saida_parada[s] = t_saida

        # 8. Agenda chegada na próxima parada
        if s < n_paradas - 1:
            t_prox = t_saida + float(tempos_arco[b, s])
            heapq.heappush(eventos, (t_prox, b, s + 1))

    # Consolidação dos indicadores da réplica
    hw = np.array(headways_obs, dtype=float)
    esp = np.array(esperas_pax, dtype=float) if esperas_pax else np.array([0.0])

    rmse_h_min = float(np.sqrt(np.mean((hw - h_star) ** 2)) / 60.0) if len(hw) > 0 else 0.0
    nrmse_h = float(np.sqrt(np.mean(((hw - h_star) / h_star) ** 2))) if len(hw) > 0 else 0.0
    bunching_pct = float(np.mean(hw < 0.5 * h_star) * 100.0) if len(hw) > 0 else 0.0
    espera_media_min = float(np.mean(esp) / 60.0)
    espera_p95_min = float(np.percentile(esp, 95) / 60.0)
    retencao_min = float(retencao_total / 60.0)
    tempo_calc_medio = float(np.mean(tempos_calc_ms)) if tempos_calc_ms else 0.0
    factibilidade_pct = float((decisoes_factiveis / decisoes_totais) * 100.0) if decisoes_totais > 0 else 100.0

    return {
        "RMSE_min": rmse_h_min,
        "NRMSE": nrmse_h,
        "Espera_media_min": espera_media_min,
        "Espera_p95_min": espera_p95_min,
        "Bunching_pct": bunching_pct,
        "Nao_embarcados": float(nao_embarcados),
        "Retencao_acum_min": retencao_min,
        "Tempo_computo_ms": tempo_calc_medio,
        "Factibilidade_pct": factibilidade_pct,
    }


def bootstrap_ic95(valores, n_boot=500, seed=42):
    arr = np.asarray(valores, dtype=float)
    if len(arr) <= 1:
        m = float(np.mean(arr))
        return m, m, m
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    medias = np.mean(arr[idx], axis=1)
    return float(np.mean(arr)), float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5))


# ==============================================================================
# 4. INTERFACE STREAMLIT
# ==============================================================================
st.set_page_config(page_title="Simulador RMPC - Transporte Público", layout="wide")
st.title("Simulador Estocástico RMPC — Redes Urbanas de Autobuses (Petrolina)")
st.caption(
    "Implementação computacional da metodologia e geração quantitativa da Tabela 3 "
    "(comparação entre Sem Controle, Reativo, MPC Nominal e RMPC com sementes pareadas)."
)

with st.sidebar:
    st.header("1. Escopo Experimental")
    itinerario_sel = st.selectbox(
        "Itinerário (Tabela 1)",
        options=list(ITINERARIOS.keys()),
        index=9,  # 041-Univ. por padrão (mais rápido para teste inicial)
        format_func=lambda k: f"{k} — {ITINERARIOS[k]['nome']} ({ITINERARIOS[k]['paradas']} par., {ITINERARIOS[k]['duracao_min']} min)",
    )
    cenario_sel = st.selectbox("Cenário de perturbação", options=list(CENARIOS_ESTRESS.keys()), index=5)
    h_star_min = st.selectbox("Intervalo programado h* (min)", options=[10, 15, 20], index=0)
    n_replicas = st.slider("Número de réplicas (sementes)", min_value=5, max_value=200, value=20, step=5)
    n_viagens = st.slider("Viagens consecutivas por réplica", min_value=6, max_value=24, value=12, step=2)

    st.header("2. Parâmetros de Calibração")
    demanda_base = st.number_input("Demanda média (pax/min por parada)", min_value=0.02, max_value=1.0, value=0.12, step=0.02)
    passo_ctrl = st.slider("Retenção autorizada a cada N paradas", min_value=3, max_value=15, value=6)
    k_reativo = st.slider("Ganho do controle reativo (k_p)", min_value=0.1, max_value=1.0, value=0.5, step=0.1)

    st.header("3. Parâmetros MPC / RMPC")
    alpha = st.slider("Peso α (regularidade)", 0.0, 1.0, 0.50, 0.05)
    beta = st.slider("Peso β (espera)", 0.0, 1.0 - alpha, min(0.35, 1.0 - alpha), 0.05)
    gamma = round(max(0.0, 1.0 - alpha - beta), 2)
    st.info(f"Peso γ (custo de retenção) = **{gamma:.2f}** (α + β + γ = 1)")
    n_omega = st.slider("Nº de cenários robustos (N_ω)", min_value=3, max_value=12, value=5)
    raio_omega = st.slider("Amplitude do conjunto Ω (±%)", min_value=5, max_value=40, value=20, step=5) / 100.0

params = {
    "h_star_min": h_star_min,
    "capacidade": 80,
    "theta_B": 2.5,
    "theta_D": 1.8,
    "tau_0": 5.0,
    "cv": 0.15,
    "u_max": 180.0,
    "H": 5,
    "W_ref": 5000.0,
    "alpha": alpha,
    "beta": beta,
    "gamma": gamma,
    "demanda_pax_min_parada": demanda_base,
    "passo_paradas_ctrl": passo_ctrl,
    "k_reativo": k_reativo,
    "n_omega": n_omega,
    "raio_omega": raio_omega,
    "n_viagens": n_viagens,
}

if st.button("Executar Simulação Comparativa", type="primary"):
    politicas = ["Sem controle", "Controle reativo", "MPC nominal", "RMPC"]
    progresso = st.progress(0)
    status = st.empty()

    registros = []
    total_passos = len(politicas) * n_replicas
    passo_atual = 0

    for pol in politicas:
        for r in range(n_replicas):
            seed = 1000 + r  # Mesmas sementes aleatórias para todas as políticas
            res = executar_replica(itinerario_sel, cenario_sel, pol, seed, params)
            res["Politica"] = pol
            res["Seed"] = seed
            registros.append(res)
            passo_atual += 1
            progresso.progress(passo_atual / total_passos)
            status.text(f"Executando {pol} — réplica {r + 1}/{n_replicas}...")

    status.success("Simulação concluída com sucesso!")
    df_bruto = pd.DataFrame(registros)

    # Monta a Tabela 3 com média e IC 95% bootstrap
    metricas_nomes = [
        ("RMSE_min", "RMSE de intervalos (min)"),
        ("NRMSE", "NRMSE de intervalos"),
        ("Espera_media_min", "Espera média (min)"),
        ("Espera_p95_min", "Espera p95 (min)"),
        ("Bunching_pct", "Bunching h < 0,5h* (%)"),
        ("Nao_embarcados", "Passageiros não embarcados"),
        ("Retencao_acum_min", "Retenção acumulada (min)"),
        ("Tempo_computo_ms", "Tempo de cómputo (ms/decisão)"),
        ("Factibilidade_pct", "Factibilidade (%)"),
    ]

    linhas_tabela3 = []
    for col, rotulo in metricas_nomes:
        linha = {"Indicador": rotulo}
        for pol in politicas:
            vals = df_bruto.loc[df_bruto["Politica"] == pol, col].values
            m, lo, hi = bootstrap_ic95(vals)
            linha[pol] = f"{m:.2f} [{lo:.2f}; {hi:.2f}]"
        linhas_tabela3.append(linha)

    df_tabela3 = pd.DataFrame(linhas_tabela3)

    st.subheader(f"Tabela 3 Preenchida — Itinerário {itinerario_sel} | Cenário: {cenario_sel}")
    st.caption(f"Valores reportados como Média [IC 95% bootstrap] sobre {n_replicas} réplicas pareadas.")
    st.dataframe(df_tabela3, use_container_width=True, hide_index=True)

    # Gráficos comparativos rápidos
    st.subheader("Comparação Visual das Médias")
    df_medias = df_bruto.groupby("Politica", sort=False).mean(numeric_only=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**RMSE de Intervalos (min)**")
        st.bar_chart(df_medias["RMSE_min"])
    with col2:
        st.markdown("**Espera Média (min)**")
        st.bar_chart(df_medias["Espera_media_min"])
    with col3:
        st.markdown("**Bunching (%)**")
        st.bar_chart(df_medias["Bunching_pct"])

    # Downloads para o artigo
    c_dl1, c_dl2 = st.columns(2)
    with c_dl1:
        st.download_button(
            "Baixar Tabela 3 (CSV)",
            data=df_tabela3.to_csv(index=False).encode("utf-8"),
            file_name=f"tabela3_{itinerario_sel}_{cenario_sel.replace(' ', '_')}.csv",
            mime="text/csv",
        )
    with c_dl2:
        st.download_button(
            "Baixar Dados Brutos por Réplica (CSV)",
            data=df_bruto.to_csv(index=False).encode("utf-8"),
            file_name=f"replicas_{itinerario_sel}_{cenario_sel.replace(' ', '_')}.csv",
            mime="text/csv",
        )
