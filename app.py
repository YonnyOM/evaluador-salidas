# -*- coding: utf-8 -*-
"""Versión web (Streamlit) del evaluador de salidas de la clasificadora."""
import pandas as pd
import streamlit as st

import evaluador_salidas as ev

COLS = ["Línea", "Calibre", "Desde", "Hasta", "Color", "Calidad",
        "C1", "C2", "C3", "C4", "Unid.", "% Peso"]
PALETA = {
    "azul": ("#B5D4F4", "#185FA5"), "rojo": ("#F7C1C1", "#A32D2D"),
    "verde": ("#C0DD97", "#3B6D11"), "gris": ("#EEEEEE", "#888888"),
}


def filas_a_df(filas):
    reg = []
    for f in filas:
        c = (f.outs + [0, 0, 0, 0])[:4]
        reg.append([f.linea, f.calibre, f.desde, f.hasta, f.color,
                    f.calidad, *c, f.unidades, f.pct])
    return pd.DataFrame(reg, columns=COLS)


def df_a_filas(df):
    filas = []
    for _, r in df.iterrows():
        if pd.isna(r["Calibre"]) or str(r["Calibre"]).strip() == "":
            continue
        n = lambda k: 0 if pd.isna(r[k]) else int(r[k])   # noqa: E731
        outs = [n(k) for k in ("C1", "C2", "C3", "C4") if n(k) != 0]
        filas.append(ev.Fila(
            n("Línea"), str(r["Calibre"]), n("Desde"), n("Hasta"),
            "" if pd.isna(r["Color"]) else str(r["Color"]),
            "" if pd.isna(r["Calidad"]) else str(r["Calidad"]),
            outs, n("Unid."),
            0.0 if pd.isna(r["% Peso"]) else float(r["% Peso"])))
    return filas


def svg_diagrama(colores):
    x0, paso, ancho = 50, 42, 40
    p = ['<svg viewBox="0 0 680 270" width="100%" '
         'xmlns="http://www.w3.org/2000/svg" font-family="sans-serif" '
         'font-size="13">',
         f'<text x="{x0}" y="20" fill="#666">Parte trasera: descarte</text>',
         f'<text x="{x0}" y="222" fill="#666">Parte delantera: '
         'exportación</text>',
         f'<rect x="{x0}" y="108" width="{14 * paso - 2}" height="26" '
         'fill="none" stroke="#888" stroke-dasharray="4 3"/>',
         f'<text x="{x0 + 7 * paso}" y="125" text-anchor="middle" '
         'fill="#666">Banda transportadora</text>']
    for i, o in enumerate(range(4, 18)):
        x = x0 + i * paso
        fill, borde = PALETA[colores.get(o, "gris")]
        y1, y2 = (44, 102) if o in ev.DESCARTES else (140, 198)
        p.append(f'<rect x="{x}" y="{y1}" width="{ancho}" '
                 f'height="{y2 - y1}" rx="4" fill="{fill}" '
                 f'stroke="{borde}"/>')
        p.append(f'<text x="{x + ancho / 2}" y="{(y1 + y2) / 2 + 5}" '
                 f'text-anchor="middle" font-weight="bold" '
                 f'fill="{borde}">{o}</text>')
    x = 50
    for k, txt in (("azul", "Salidas actuales"), ("rojo", "No habilitable"),
                   ("verde", "Habilitable")):
        fill, borde = PALETA[k]
        p.append(f'<rect x="{x}" y="244" width="14" height="14" rx="3" '
                 f'fill="{fill}" stroke="{borde}"/>')
        p.append(f'<text x="{x + 20}" y="256" fill="#555">{txt}</text>')
        x += 170
    p.append("</svg>")
    return "".join(p)


def main():
    st.set_page_config(page_title="Evaluador de salidas", layout="wide")
    st.title("Evaluador de salidas · clasificadora de limón")
    st.caption("Sube la captura de la tabla «Programas», revisa los números "
               "y escribe el calibre que quieres ampliar.")

    if "version" not in st.session_state:
        st.session_state.version = 0
        st.session_state.df = pd.DataFrame(columns=COLS)
        st.session_state.archivo = None

    with st.sidebar:
        st.header("Parámetros")
        limite = st.number_input("Salida llena (% por salida)", 0.0, 100.0,
                                 ev.LIMITE_LLENA, 0.5)
        umbral = st.number_input("Bajo flujo (máx. % por salida)", 0.0,
                                 100.0, ev.UMBRAL_BAJO, 0.5)
        if st.button("Limpiar todo"):
            st.session_state.version += 1
            st.session_state.df = pd.DataFrame(columns=COLS)
            st.session_state.archivo = None
            st.rerun()

    subida = st.file_uploader("Captura de la tabla",
                              type=["png", "jpg", "jpeg", "webp", "bmp"],
                              key=f"up{st.session_state.version}")
    if subida is not None:
        clave = (subida.name, subida.size)
        if st.session_state.archivo != clave:
            with st.spinner("Leyendo la imagen…"):
                try:
                    filas, _ = ev.ocr_imagen(subida)
                except Exception as ex:   # noqa: BLE001
                    st.error(f"No pude leer la imagen: {ex}")
                    filas = []
            st.session_state.df = filas_a_df(filas)
            st.session_state.archivo = clave
            st.session_state.version += 1
            st.rerun()
        with st.expander("Ver imagen subida"):
            st.image(subida)
        if st.session_state.df.empty:
            st.warning("No reconocí filas. Prueba con una captura más "
                       "grande o agrega las filas a mano.")

    st.subheader("Tabla leída (puedes corregir cualquier celda)")
    df = st.data_editor(
        st.session_state.df, num_rows="dynamic", width="stretch",
        key=f"tabla{st.session_state.version}",
        column_config={"% Peso": st.column_config.NumberColumn(format="%.2f")})
    filas = df_a_filas(df)

    calibre = st.text_input("Calibre a ampliar (ej. 200)")
    if not calibre.strip():
        return
    objetivos = ev.buscar_lineas(filas, calibre)
    if not objetivos:
        st.info("No encontré líneas de exportación con ese calibre.")
        return

    for f in objetivos:
        r = ev.evaluar(filas, f, limite, umbral)
        st.markdown("---")
        st.subheader(f.titulo)
        if f.outs:
            st.write(f"Carga: **{ev.fmt(f.pct)}%** en la línea → "
                     f"**{ev.fmt(f.por_salida)}%** por salida.")
        for a in r.avisos:
            st.warning(a)
        (st.success if r.posible else st.error)(r.veredicto)
        for c, (ok, razones) in r.candidatas.items():
            if ok:
                st.write(f"✅ **Salida {c}:** libre o con bajo flujo y "
                         "cumple las reglas.")
            else:
                st.write(f"❌ **Salida {c}:**")
                for z in razones:
                    st.write(f"   - {z}")
        st.markdown(svg_diagrama(r.colores), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
