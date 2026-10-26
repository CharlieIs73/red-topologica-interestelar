"""
Red Topológica Interestelar — prototipo 2D
Modelo exploratorio, no una simulación de física validada.

Instalación:
    pip install -r requirements.txt
Ejecución:
    streamlit run app.py
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import networkx as nx
import numpy as np
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="Red Topológica Interestelar", page_icon="✦", layout="wide")

@dataclass
class Model:
    graph: nx.Graph
    positions: dict
    rest_lengths: dict
    stiffness: dict
    displacement: dict
    active: dict
    threshold: dict


def edge_key(u, v):
    return tuple(sorted((int(u), int(v))))


def make_model(n_nodes=36, extra_edges=18, seed=7) -> Model:
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    # Distribución de nodos en una región elíptica, con separación mínima aproximada.
    coords = []
    attempts = 0
    while len(coords) < n_nodes and attempts < n_nodes * 200:
        attempts += 1
        x = rng.uniform(-1.0, 1.0)
        y = rng.uniform(-0.72, 0.72)
        if (x*x / 1.0 + y*y / 0.52) > 1:
            continue
        if all((x-a)**2 + (y-b)**2 > 0.075**2 for a, b in coords):
            coords.append((x, y))
    while len(coords) < n_nodes:
        coords.append((rng.uniform(-0.8, 0.8), rng.uniform(-0.6, 0.6)))

    positions = {i: np.array(coords[i], dtype=float) for i in range(n_nodes)}
    graph = nx.Graph()
    graph.add_nodes_from(range(n_nodes))

    # Árbol de expansión mínimo: garantiza conectividad inicial.
    complete = nx.Graph()
    complete.add_nodes_from(range(n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            d = float(np.linalg.norm(positions[i] - positions[j]))
            complete.add_edge(i, j, weight=d)
    tree = nx.minimum_spanning_tree(complete, weight="weight")
    graph.add_edges_from(tree.edges())

    # Corredores extra para crear ciclos y rutas alternativas.
    candidates = [(i, j, complete[i][j]["weight"])
                  for i, j in complete.edges()
                  if not graph.has_edge(i, j)]
    py_rng.shuffle(candidates)
    for i, j, _ in sorted(candidates, key=lambda item: item[2])[:extra_edges]:
        graph.add_edge(i, j)

    rest_lengths, stiffness, displacement, active, threshold = {}, {}, {}, {}, {}
    for u, v in graph.edges():
        key = edge_key(u, v)
        length = float(np.linalg.norm(positions[u] - positions[v]))
        rest_lengths[key] = length
        stiffness[key] = float(rng.uniform(0.7, 1.8))
        displacement[key] = 0.0
        active[key] = True
        threshold[key] = float(rng.uniform(0.55, 1.05))

    return Model(graph, positions, rest_lengths, stiffness, displacement, active, threshold)


def active_graph(model: Model) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(model.graph.nodes())
    for u, v in model.graph.edges():
        if model.active[edge_key(u, v)]:
            g.add_edge(u, v)
    return g


def total_energy(model: Model) -> float:
    # E = 1/2 sum k*x^2; x es una deformación normalizada de juguete.
    return float(sum(0.5 * model.stiffness[k] * x*x
                     for k, x in model.displacement.items()
                     if model.active[k]))


def apply_perturbation(model: Model, node: int, amplitude: float, spread: float):
    """Carga de juguete: distribuye deformación entre aristas cercanas al nodo."""
    for u, v in model.graph.edges(node):
        key = edge_key(u, v)
        if not model.active[key]:
            continue
        other = v if u == node else u
        d = float(np.linalg.norm(model.positions[node] - model.positions[other]))
        # El efecto decae con la distancia; no representa una ley física universal.
        contribution = amplitude * math.exp(-d / max(spread, 0.05))
        model.displacement[key] += contribution


def propagate_one_step(model: Model, coupling: float, damping: float):
    """Propagación heurística entre aristas que comparten un nodo."""
    before = dict(model.displacement)
    for u, v in model.graph.edges():
        key = edge_key(u, v)
        if not model.active[key]:
            continue
        neighbors = set()
        for node in (u, v):
            for a, b in model.graph.edges(node):
                k2 = edge_key(a, b)
                if k2 != key and model.active.get(k2, False):
                    neighbors.add(k2)
        if neighbors:
            mean_neighbor = float(np.mean([before[k] for k in neighbors]))
            model.displacement[key] = max(
                0.0, (1.0 - damping) * before[key] + coupling * mean_neighbor
            )
        else:
            model.displacement[key] = max(0.0, (1.0 - damping) * before[key])

    failed_now = []
    for u, v in model.graph.edges():
        key = edge_key(u, v)
        if model.active[key] and model.displacement[key] > model.threshold[key]:
            model.active[key] = False
            failed_now.append(key)
    return failed_now


def build_figure(model: Model, show_inactive: bool = True):
    fig = go.Figure()

    # Corredores inactivos primero, para que los activos queden encima.
    if show_inactive:
        for u, v in model.graph.edges():
            key = edge_key(u, v)
            if model.active[key]:
                continue
            x0, y0 = model.positions[u]
            x1, y1 = model.positions[v]
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1], mode="lines",
                line=dict(color="rgba(255,80,100,0.35)", width=1, dash="dot"),
                hoverinfo="text", text=f"Corredor {u}–{v}: INACTIVO",
                showlegend=False
            ))

    for u, v in model.graph.edges():
        key = edge_key(u, v)
        if not model.active[key]:
            continue
        x0, y0 = model.positions[u]
        x1, y1 = model.positions[v]
        x = model.displacement[key]
        t = min(x / max(model.threshold[key], 0.01), 1.0)
        # Color de azul/cian hacia amarillo/rojo según carga relativa.
        if t < 0.5:
            color = f"rgb({int(40+40*t*2)},{int(130+100*t*2)},{int(245-25*t*2)})"
        else:
            q = (t - 0.5) * 2
            color = f"rgb({int(120+135*q)},{int(230-150*q)},{int(220-180*q)})"
        width = 1.5 + 3.5 * min(t, 1)
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines",
            line=dict(color=color, width=width),
            hoverinfo="text",
            text=(f"Corredor {u}–{v}<br>Activo<br>Deformación: {x:.3f}"
                  f"<br>Umbral: {model.threshold[key]:.3f}"
                  f"<br>Rigidez: {model.stiffness[key]:.2f}"),
            showlegend=False
        ))

    node_x = [model.positions[n][0] for n in model.graph.nodes()]
    node_y = [model.positions[n][1] for n in model.graph.nodes()]
    ag = active_graph(model)
    components = list(nx.connected_components(ag))
    component_id = {}
    for idx, comp in enumerate(components):
        for n in comp:
            component_id[n] = idx
    node_colors = [component_id.get(n, 0) for n in model.graph.nodes()]
    sizes = [12 if ag.degree(n) else 8 for n in model.graph.nodes()]
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=[str(n) for n in model.graph.nodes()], textposition="top center",
        marker=dict(size=sizes, color=node_colors, colorscale="Turbo",
                    line=dict(color="white", width=1), showscale=False),
        hovertemplate="Nodo %{text}<extra></extra>",
        showlegend=False
    ))
    fig.update_layout(
        template="plotly_dark", height=610, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="#06111f", plot_bgcolor="#06111f",
        xaxis=dict(visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(visible=False),
        clickmode="event+select",
    )
    return fig


if "model" not in st.session_state:
    st.session_state.model = make_model()
if "step" not in st.session_state:
    st.session_state.step = 0
if "failed_total" not in st.session_state:
    st.session_state.failed_total = 0

st.title("✦ Red Topológica Interestelar")
st.caption("Prototipo 2D interactivo · modelo exploratorio de grafos, energía y fallos en cascada")
st.warning(
    "Este prototipo simula reglas hipotéticas, no física validada. "
    "La deformación y la propagación son heurísticas para explorar el comportamiento de la red."
)

model: Model = st.session_state.model

with st.sidebar:
    st.header("Controles")
    n_nodes = st.slider("Número de nodos al reiniciar", 12, 80, 36, 4)
    extra_edges = st.slider("Corredores extra", 0, 80, 18, 2)
    seed = st.number_input("Semilla aleatoria", min_value=0, max_value=99999, value=7, step=1)
    st.divider()
    selected_node = st.selectbox("Nodo donde inyectar perturbación", list(model.graph.nodes()), index=0)
    amplitude = st.slider("Amplitud de perturbación", 0.05, 1.5, 0.55, 0.05)
    spread = st.slider("Alcance de perturbación", 0.1, 1.5, 0.55, 0.05)
    coupling = st.slider("Acoplamiento entre corredores", 0.0, 0.8, 0.25, 0.05)
    damping = st.slider("Amortiguamiento", 0.0, 0.8, 0.12, 0.02)
    show_inactive = st.checkbox("Mostrar corredores colapsados", True)
    st.divider()
    if st.button("↻ Reiniciar red", use_container_width=True):
        st.session_state.model = make_model(int(n_nodes), int(extra_edges), int(seed))
        st.session_state.step = 0
        st.session_state.failed_total = 0
        st.rerun()
    if st.button("⚡ Inyectar perturbación", use_container_width=True):
        apply_perturbation(model, int(selected_node), float(amplitude), float(spread))
        st.rerun()
    if st.button("▶ Avanzar 1 paso", use_container_width=True):
        failed = propagate_one_step(model, float(coupling), float(damping))
        st.session_state.step += 1
        st.session_state.failed_total += len(failed)
        st.rerun()
    if st.button("⏩ Avanzar 10 pasos", use_container_width=True):
        for _ in range(10):
            failed = propagate_one_step(model, float(coupling), float(damping))
            st.session_state.step += 1
            st.session_state.failed_total += len(failed)
        st.rerun()

ag = active_graph(model)
components = list(nx.connected_components(ag))
largest = max((len(c) for c in components), default=0)
active_edges = sum(model.active.values())
all_edges = model.graph.number_of_edges()
energy = total_energy(model)
lambda2 = 0.0
if ag.number_of_nodes() > 1:
    try:
        eig = nx.laplacian_spectrum(ag)
        lambda2 = float(sorted(eig)[1]) if len(eig) > 1 else 0.0
        if abs(lambda2) < 1e-8:
            lambda2 = 0.0
    except Exception:
        lambda2 = float("nan")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Nodos", model.graph.number_of_nodes())
m2.metric("Corredores activos", f"{active_edges}/{all_edges}")
m3.metric("Energía del modelo", f"{energy:.3f}")
m4.metric("Componentes conectados", len(components))
m5.metric("Paso / fallos acumulados", f"{st.session_state.step} / {st.session_state.failed_total}")

left, right = st.columns([1.65, 1])
with left:
    st.subheader("Mapa de la red")
    st.plotly_chart(build_figure(model, show_inactive), use_container_width=True)
    st.caption("Pasa el cursor sobre un corredor para ver su deformación, umbral y rigidez. Los corredores rojos discontinuos han colapsado.")
with right:
    st.subheader("Diagnóstico")
    st.write(f"**Conectividad:** {'conexa' if len(components) == 1 else 'desconectada'}")
    st.write(f"**Mayor componente:** {largest} de {model.graph.number_of_nodes()} nodos")
    st.write(f"**Segundo autovalor de la laplaciana:** {lambda2:.4f}")
    st.caption("λ₂ > 0 indica conectividad para el grafo no dirigido con pesos no negativos. Aquí se calcula sobre la red activa sin pesos.")
    st.subheader("Energía por corredor")
    edge_labels, edge_energy = [], []
    for u, v in model.graph.edges():
        key = edge_key(u, v)
        if model.active[key]:
            edge_labels.append(f"{u}–{v}")
            edge_energy.append(0.5 * model.stiffness[key] * model.displacement[key]**2)
    if edge_energy:
        order = np.argsort(edge_energy)[-12:]
        fig_e = go.Figure(go.Bar(
            x=[edge_energy[i] for i in order],
            y=[edge_labels[i] for i in order],
            orientation="h",
            marker_color="#42c8f5",
        ))
        fig_e.update_layout(template="plotly_dark", height=360,
                            margin=dict(l=10, r=10, t=10, b=10),
                            paper_bgcolor="#06111f", plot_bgcolor="#06111f",
                            xaxis_title="Energía elástica (unidades arbitrarias)",
                            yaxis_title="Corredor")
        st.plotly_chart(fig_e, use_container_width=True)
    else:
        st.info("No quedan corredores activos.")

st.divider()
st.subheader("Supuestos matemáticos del prototipo")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**Energía elástica**")
    st.latex(r"E=\frac{1}{2}\sum_{(i,j)\in E_{\rm activo}} K_{ij}x_{ij}^{2}")
    st.caption("Modelo armónico; K > 0. Las unidades son arbitrarias.")
with c2:
    st.markdown("**Regla de fallo**")
    st.latex(r"|x_{ij}|>x_{ij}^{\rm crit}\Rightarrow \text{corredor inactivo}")
    st.caption("Umbral de fallo propuesto, no una ley física conocida.")
with c3:
    st.markdown("**Conectividad**")
    st.latex(r"\lambda_2(L)>0")
    st.caption("Criterio estándar para conectividad de un grafo no dirigido.")

with st.expander("Limitaciones y próximos pasos"):
    st.markdown("""
    - Las posiciones 2D solo sirven para visualizar; no representan una geometría espacial real.
    - La distancia efectiva todavía no modifica la posición de los nodos ni el tiempo de viaje. Puede añadirse como coste de arista.
    - La propagación de deformación es una regla heurística; aún no deriva de un sistema de ecuaciones físicas.
    - La energía calculada no incluye energía cinética, fuentes externas ni una ley de conservación.
    - La estabilidad exponencial, la resonancia y el anclaje dimensional requieren definiciones formales antes de integrarse.
    - Próxima mejora sugerida: permitir elegir origen y destino y comparar el camino más corto por distancia ordinaria y por coste topológico.
    """)
