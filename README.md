# Red Topológica Interestelar — prototipo 2D

## Requisitos
Python 3.10 o posterior recomendado.

## Instalación
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Ejecutar
```bash
streamlit run app.py
```
Streamlit abrirá la interfaz en el navegador.

## Qué incluye
- Generación de una red de nodos y corredores.
- Modelo de energía elástica E = 1/2 sum(K*x^2).
- Inyección de perturbaciones.
- Propagación heurística y colapso de corredores por umbral.
- Medidas de conectividad y segundo autovalor de la laplaciana.
- Visualización Plotly interactiva y gráfico de energía por corredor.

## Importante
Es un prototipo exploratorio. Las reglas de deformación, propagación y colapso son hipótesis de juguete; no representan una teoría física validada ni demuestran que los atajos topológicos sean posibles.
