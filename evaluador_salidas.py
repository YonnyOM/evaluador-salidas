#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluador de salidas - clasificadora de limón
=============================================

1. Carga una captura de la tabla "Programas" del software de clasificación.
2. Lee la tabla con OCR (puedes corregir cualquier celda a mano).
3. Escribes el calibre y el programa dice si se puede habilitar una salida
   más, o por qué no, con un diagrama verde / rojo.

Instalación (Ubuntu):
    sudo apt install python3-tk tesseract-ocr tesseract-ocr-spa
        python3-pytesseract python3-pil python3-numpy python3-tkinterdnd2
    (python3-tkinterdnd2 es opcional: permite arrastrar la imagen)

Uso:
    python3 evaluador_salidas.py

Reglas que aplica (todas editables en la parte superior del código):
  - Salidas de exportación, en orden físico: 4, 5, 8, 9, 11, 12, 14, 15
  - Salidas de descarte (nunca habilitables): 6, 7, 10, 13, 16, 17
  - Una salida está llena si lleva más del 15% del total; el % de una línea
    se reparte por igual entre sus salidas.
  - La salida nueva debe ser la siguiente de exportación, en cualquier
    sentido (izquierda o derecha), saltando los descartes.
  - La salida nueva debe tener bajo flujo.
  - Ninguna línea puede quedarse sin salida.
  - Una salida no se comparte entre líneas con rangos de peso distintos.
"""

import re
import statistics
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Configuración de la máquina
# --------------------------------------------------------------------------
CADENA_EXPORTACION = [4, 5, 8, 9, 11, 12, 14, 15]
DESCARTES = [6, 7, 10, 13, 16, 17]
LIMITE_LLENA = 15.0   # % por salida a partir del cual está llena
UMBRAL_BAJO = 5.0     # % por salida máximo para considerar "bajo flujo"


# --------------------------------------------------------------------------
# Modelo de datos
# --------------------------------------------------------------------------
@dataclass
class Fila:
    linea: int
    calibre: str
    desde: int
    hasta: int
    color: str
    calidad: str
    outs: list          # salidas asignadas (C1..C4 distintas de 0)
    unidades: int
    pct: float          # % peso de la línea

    @property
    def rango(self):
        return (self.desde, self.hasta)

    @property
    def por_salida(self):
        return self.pct / len(self.outs) if self.outs else 0.0

    @property
    def titulo(self):
        s = ", ".join(str(o) for o in self.outs) or "ninguna"
        return (f"Línea {self.linea} · {self.calibre} "
                f"({self.desde}–{self.hasta}) · salidas: {s}")


def fmt(x):
    return f"{x:.2f}".replace(".", ",")


# --------------------------------------------------------------------------
# Lectura de la tabla (OCR)
# --------------------------------------------------------------------------
_ES_INT = re.compile(r"^\d+$")
_ES_DEC = re.compile(r"^\d+[.,]\d+$")


def parsear_linea(texto):
    """Convierte una línea de texto OCR en una Fila (o None si no es fila)."""
    tokens = texto.split()
    if tokens:
        # Número de línea mal leído (ej. "£" en vez de 8).
        tokens[0] = tokens[0].translate(str.maketrans(
            {"£": "8", "S": "5", "B": "8", "l": "1", "I": "1", "|": "1"}))
    tokens = [t for t in tokens if re.search(r"\w", t)]
    # El OCR confunde la letra O con el cero: se corrige donde hay números.
    tokens = ["0" if t in ("O", "o") else t for t in tokens]
    if len(tokens) < 12 or not _ES_INT.match(tokens[0]):
        return None
    decs = [i for i, t in enumerate(tokens) if _ES_DEC.match(t)]
    if not decs:
        return None
    # Si la imagen incluye "Peso Total" hay dos decimales al final.
    pct_i = decs[-2] if len(decs) >= 2 else decs[-1]
    if pct_i < 6:
        return None
    try:
        unidades = int(tokens[pct_i - 1])
        c = [int(t) for t in tokens[pct_i - 5:pct_i - 1]]
    except ValueError:
        return None
    pct = float(tokens[pct_i].replace(",", "."))
    cabeza = tokens[1:pct_i - 5]
    k = None
    for j in range(len(cabeza) - 2):
        if (_ES_INT.match(cabeza[j]) and _ES_INT.match(cabeza[j + 1])
                and not _ES_INT.match(cabeza[j + 2])):
            k = j
            break
    if k is None or k == 0:
        return None
    calibre = " ".join(cabeza[:k])
    desde, hasta = int(cabeza[k]), int(cabeza[k + 1])
    resto = " ".join(cabeza[k + 2:])
    m = re.search(r"(Sin calidad.*|Industria.*)$", resto)
    if m:
        calidad, color = m.group(1), resto[:m.start()].strip()
    else:
        color, calidad = resto, ""
    return Fila(int(tokens[0]), calibre, desde, hasta, color, calidad,
                [x for x in c if x != 0], unidades, pct)


def preparar_imagen(img):
    """Mejora la imagen para OCR (y arregla la fila naranja seleccionada)."""
    import numpy as np
    from PIL import Image
    rgb = img.convert("RGB")
    arr = np.array(rgb).astype(int)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    naranja = (r > 200) & (g > 60) & (g < 130) & (b < 90)
    banda = naranja.mean(axis=1) > 0.5
    gris = np.array(rgb.convert("L"))
    for y in np.where(banda)[0]:
        gris[y] = np.where(b[y] > 150, 0, 255)   # texto blanco -> negro
    out = Image.fromarray(gris.astype("uint8"))
    return out.resize((out.width * 2, out.height * 2), Image.LANCZOS)


def ocr_imagen(ruta):
    """Devuelve (filas, texto_ocr) a partir de una captura."""
    import pytesseract
    from PIL import Image
    img = preparar_imagen(Image.open(ruta))
    cfg = "--psm 6 -c preserve_interword_spaces=1"
    try:
        datos = pytesseract.image_to_data(
            img, lang="spa+eng", config=cfg,
            output_type=pytesseract.Output.DICT)
    except pytesseract.TesseractError:
        datos = pytesseract.image_to_data(
            img, lang="eng", config=cfg,
            output_type=pytesseract.Output.DICT)

    palabras = []
    for i, t in enumerate(datos["text"]):
        if t.strip() and float(datos["conf"][i]) >= 0:
            top, h = datos["top"][i], datos["height"][i]
            palabras.append((top + h / 2, datos["left"][i], t.strip(), h))
    if not palabras:
        return [], ""
    tol = statistics.median(p[3] for p in palabras) * 0.6
    palabras.sort()
    grupos, actual, y0 = [], [palabras[0]], palabras[0][0]
    for p in palabras[1:]:
        if abs(p[0] - y0) <= tol:
            actual.append(p)
        else:
            grupos.append(actual)
            actual, y0 = [p], p[0]
    grupos.append(actual)

    lineas = [" ".join(w[2] for w in sorted(g, key=lambda w: w[1]))
              for g in grupos]
    filas = [f for f in (parsear_linea(t) for t in lineas) if f]
    return filas, "\n".join(lineas)


# --------------------------------------------------------------------------
# Lógica de decisión
# --------------------------------------------------------------------------
@dataclass
class Resultado:
    fila: Fila
    avisos: list = field(default_factory=list)
    candidatas: dict = field(default_factory=dict)   # salida -> (ok, razones)
    veredicto: str = ""
    posible: bool = False
    colores: dict = field(default_factory=dict)      # salida -> color


def evaluar(filas, objetivo, limite=LIMITE_LLENA, umbral=UMBRAL_BAJO):
    res = Resultado(objetivo)
    outs = objetivo.outs

    if not outs:
        res.veredicto = ("Esta línea no tiene ninguna salida asignada: "
                         "primero hay que asignarle una, no ampliarla.")
        res.colores = {o: "rojo" for o in range(4, 18)}
        return res
    if all(o in DESCARTES for o in outs):
        res.veredicto = "Es una línea de descarte: sus salidas no se amplían."
        res.colores = {o: "rojo" for o in range(4, 18)}
        for o in outs:
            res.colores[o] = "azul"
        return res

    if objetivo.por_salida <= limite:
        res.avisos.append(
            f"La línea todavía no está llena: {fmt(objetivo.por_salida)}% "
            f"por salida (límite {fmt(limite)}%).")

    vecinas = []
    for o in outs:
        if o in CADENA_EXPORTACION:
            i = CADENA_EXPORTACION.index(o)
            for j in (i - 1, i + 1):
                if 0 <= j < len(CADENA_EXPORTACION):
                    c = CADENA_EXPORTACION[j]
                    if c not in outs and c not in vecinas:
                        vecinas.append(c)

    for c in vecinas:
        razones = []
        duenos = [f for f in filas if f is not objetivo and c in f.outs]
        flujo = sum(f.por_salida for f in duenos)
        if duenos and flujo > umbral:
            razones.append(
                f"no es de bajo flujo: lleva {fmt(flujo)}% por salida "
                f"(máximo {fmt(umbral)}%)")
        for f in duenos:
            if f.rango == objetivo.rango:
                continue   # mismo rango de peso: se puede compartir
            if len(f.outs) == 1:
                razones.append(
                    f"es la única salida de {f.calibre} "
                    f"({f.desde}–{f.hasta}): no se puede quitar y no se "
                    f"comparte con otro rango de peso")
            else:
                resto = f.pct / (len(f.outs) - 1)
                if resto > limite:
                    razones.append(
                        f"quitarla a {f.calibre} ({f.desde}–{f.hasta}) "
                        f"subiría su carga a {fmt(resto)}% por salida")
        res.candidatas[c] = (not razones, razones)

    ok = [c for c, (v, _) in res.candidatas.items() if v]
    res.posible = bool(ok)
    if ok:
        res.veredicto = ("SÍ es posible. Salida(s) que se pueden habilitar: "
                         + ", ".join(str(c) for c in ok) + ".")
    elif not vecinas:
        res.veredicto = ("NO es posible: no hay otra salida de exportación "
                         "contigua (la cadena termina en el extremo y los "
                         "descartes no se habilitan).")
    else:
        res.veredicto = "NO es posible habilitar otra salida."

    for o in range(4, 18):
        if o in outs:
            res.colores[o] = "azul"
        elif o in ok:
            res.colores[o] = "verde"
        else:
            res.colores[o] = "rojo"
    return res


def buscar_lineas(filas, texto):
    """Líneas cuyo nombre de calibre contiene el texto buscado."""
    t = texto.strip().lower()
    if not t:
        return []
    return [f for f in filas
            if t in f.calibre.lower()
            and (not f.outs or any(o in CADENA_EXPORTACION for o in f.outs))]


# --------------------------------------------------------------------------
# Interfaz gráfica
# --------------------------------------------------------------------------
COLUMNAS = ["Línea", "Calibre", "Desde", "Hasta", "Color", "Calidad",
            "C1", "C2", "C3", "C4", "Unid.", "% Peso"]
ANCHOS = [50, 150, 55, 55, 170, 170, 40, 40, 40, 40, 60, 65]
PALETA = {
    "azul": ("#B5D4F4", "#185FA5"),
    "rojo": ("#F7C1C1", "#A32D2D"),
    "verde": ("#C0DD97", "#3B6D11"),
    "gris": ("#EEEEEE", "#888888"),
}


def lanzar_gui():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    try:
        from tkinterdnd2 import TkinterDnD, DND_FILES
        base_tk, arrastre_ok = TkinterDnD.Tk, True
    except ImportError:
        DND_FILES, base_tk, arrastre_ok = None, tk.Tk, False

    class App(base_tk):
        def __init__(self):
            super().__init__()
            self.title("Evaluador de salidas - clasificadora de limón")
            self.geometry("1100x860")
            self.resultados = []
            self._construir()
            self._dibujar({})
            self._activar_arrastre()

        # ---------------------------------------------------------- UI
        def _construir(self):
            barra = ttk.Frame(self)
            barra.pack(fill="x", padx=8, pady=6)
            ttk.Button(barra, text="Cargar imagen…",
                       command=self.cargar_imagen).pack(side="left")
            ttk.Button(barra, text="Agregar fila",
                       command=self.agregar_fila).pack(side="left", padx=4)
            ttk.Button(barra, text="Borrar fila",
                       command=self.borrar_fila).pack(side="left")
            ttk.Button(barra, text="Limpiar todo",
                       command=self.limpiar).pack(side="left", padx=4)
            self.estado = ttk.Label(
                barra, text="Carga una captura o agrega filas a mano. "
                            "Doble clic en una celda para corregirla.")
            self.estado.pack(side="left", padx=10)

            marco = ttk.Frame(self)
            marco.pack(fill="both", expand=False, padx=8)
            self.tree = ttk.Treeview(marco, columns=COLUMNAS,
                                     show="headings", height=11)
            for c, w in zip(COLUMNAS, ANCHOS):
                self.tree.heading(c, text=c)
                self.tree.column(c, width=w, anchor="w")
            sb = ttk.Scrollbar(marco, orient="vertical",
                               command=self.tree.yview)
            self.tree.configure(yscrollcommand=sb.set)
            self.tree.pack(side="left", fill="both", expand=True)
            sb.pack(side="right", fill="y")
            self.tree.bind("<Double-1>", self._editar_celda)

            par = ttk.Frame(self)
            par.pack(fill="x", padx=8, pady=6)
            ttk.Label(par, text="Límite de salida llena (%):").pack(side="left")
            self.v_limite = tk.StringVar(value=str(LIMITE_LLENA))
            ttk.Spinbox(par, from_=1, to=100, increment=0.5, width=6,
                        textvariable=self.v_limite).pack(side="left", padx=4)
            ttk.Label(par, text="Bajo flujo (máx. % por salida):"
                      ).pack(side="left", padx=(14, 0))
            self.v_umbral = tk.StringVar(value=str(UMBRAL_BAJO))
            ttk.Spinbox(par, from_=0, to=100, increment=0.5, width=6,
                        textvariable=self.v_umbral).pack(side="left", padx=4)

            ev = ttk.Frame(self)
            ev.pack(fill="x", padx=8, pady=4)
            ttk.Label(ev, text="Calibre a ampliar (ej. 200):").pack(side="left")
            self.v_calibre = tk.StringVar()
            e = ttk.Entry(ev, textvariable=self.v_calibre, width=18)
            e.pack(side="left", padx=6)
            e.bind("<Return>", lambda _e: self.evaluar())
            ttk.Button(ev, text="Evaluar", command=self.evaluar
                       ).pack(side="left")
            ttk.Label(ev, text="   Resultado mostrado:").pack(side="left")
            self.combo = ttk.Combobox(ev, state="readonly", width=60)
            self.combo.pack(side="left", padx=6)
            self.combo.bind("<<ComboboxSelected>>", self._cambiar_resultado)

            self.canvas = tk.Canvas(self, width=680, height=250,
                                    bg="white", highlightthickness=0)
            self.canvas.pack(padx=8, pady=6)

            self.texto = tk.Text(self, height=12, wrap="word",
                                 font=("TkDefaultFont", 11))
            self.texto.pack(fill="both", expand=True, padx=8, pady=(0, 8))
            self.texto.tag_configure("ok", foreground="#3B6D11",
                                     font=("TkDefaultFont", 11, "bold"))
            self.texto.tag_configure("mal", foreground="#A32D2D",
                                     font=("TkDefaultFont", 11, "bold"))
            self.texto.tag_configure("aviso", foreground="#B25E00")
            self.texto.tag_configure("negrita",
                                     font=("TkDefaultFont", 11, "bold"))

        # ------------------------------------------------- tabla
        def _poner_filas(self, filas):
            self.tree.delete(*self.tree.get_children())
            for f in filas:
                c = (f.outs + [0, 0, 0, 0])[:4]
                self.tree.insert("", "end", values=(
                    f.linea, f.calibre, f.desde, f.hasta, f.color,
                    f.calidad, *c, f.unidades, fmt(f.pct)))

        def _leer_filas(self):
            filas = []
            for it in self.tree.get_children():
                v = self.tree.item(it, "values")
                try:
                    outs = [int(x) for x in v[6:10] if int(x) != 0]
                    filas.append(Fila(
                        int(v[0]), str(v[1]), int(v[2]), int(v[3]),
                        str(v[4]), str(v[5]), outs, int(v[10]),
                        float(str(v[11]).replace(",", "."))))
                except ValueError:
                    messagebox.showerror(
                        "Dato inválido",
                        f"Revisa la fila de la línea {v[0]}: hay un valor "
                        "que no es número donde debería serlo.")
                    return None
            return filas

        def _reset_resultados(self):
            self.resultados = []
            self.combo.set("")
            self.combo["values"] = []
            self.texto.config(state="normal")
            self.texto.delete("1.0", "end")
            self._dibujar({})

        def limpiar(self):
            self.tree.delete(*self.tree.get_children())
            self.v_calibre.set("")
            self._reset_resultados()
            self.estado.config(text="Tabla limpia. Arrastra una captura "
                                    "nueva o pulsa «Cargar imagen».")

        def agregar_fila(self):
            self.tree.insert("", "end", values=(
                len(self.tree.get_children()), "CALIBRE", 0, 0, "", "",
                0, 0, 0, 0, 0, "0,00"))

        def borrar_fila(self):
            for it in self.tree.selection():
                self.tree.delete(it)

        def _editar_celda(self, ev):
            it = self.tree.identify_row(ev.y)
            col = self.tree.identify_column(ev.x)
            if not it or not col:
                return
            idx = int(col[1:]) - 1
            x, y, w, h = self.tree.bbox(it, col)
            ent = ttk.Entry(self.tree)
            ent.insert(0, self.tree.item(it, "values")[idx])
            ent.place(x=x, y=y, width=w, height=h)
            ent.focus()

            def guardar(_e=None):
                v = list(self.tree.item(it, "values"))
                v[idx] = ent.get()
                self.tree.item(it, values=v)
                ent.destroy()

            ent.bind("<Return>", guardar)
            ent.bind("<FocusOut>", guardar)
            ent.bind("<Escape>", lambda _e: ent.destroy())

        # ------------------------------------------------- imagen
        def _activar_arrastre(self):
            if arrastre_ok:
                for w in (self, self.tree, self.canvas, self.texto):
                    w.drop_target_register(DND_FILES)
                    w.dnd_bind("<<Drop>>", self._soltar)
                self.estado.config(
                    text="Arrastra aquí la captura o pulsa «Cargar imagen». "
                         "Doble clic en una celda para corregirla.")
            else:
                self.estado.config(
                    text="Pulsa «Cargar imagen». (Para arrastrar archivos: "
                         "sudo apt install python3-tkinterdnd2)")

        def _soltar(self, ev):
            rutas = self.tk.splitlist(ev.data)
            if rutas:
                self.cargar_ruta(rutas[0])

        def cargar_imagen(self):
            ruta = filedialog.askopenfilename(
                filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp *.webp"),
                           ("Todos", "*.*")])
            if ruta:
                self.cargar_ruta(ruta)

        def cargar_ruta(self, ruta):
            self.estado.config(text="Leyendo la imagen…")
            self.update_idletasks()
            try:
                filas, _ = ocr_imagen(ruta)
            except ImportError as ex:
                falta = ex.name or str(ex)
                messagebox.showerror(
                    "Falta una librería",
                    f"A Python le falta: {falta}\n\n"
                    "Instala todo con:\n"
                    "sudo apt install tesseract-ocr tesseract-ocr-spa "
                    "python3-pytesseract python3-pil python3-numpy\n\n"
                    "Mientras tanto puedes escribir la tabla a mano.")
                self.estado.config(text="OCR no disponible.")
                return
            except Exception as ex:   # noqa: BLE001
                messagebox.showerror("Error leyendo la imagen", str(ex))
                self.estado.config(text="Error de lectura.")
                return
            self._poner_filas(filas)
            self._reset_resultados()
            self.estado.config(
                text=f"{len(filas)} filas leídas. REVISA los números "
                     "(doble clic para corregir) antes de evaluar.")
            if not filas:
                messagebox.showwarning(
                    "Sin filas",
                    "No pude reconocer filas. Prueba con una captura más "
                    "grande o agrega las filas a mano.")

        # ------------------------------------------------- evaluación
        def evaluar(self):
            filas = self._leer_filas()
            if filas is None:
                return
            try:
                limite = float(self.v_limite.get().replace(",", "."))
                umbral = float(self.v_umbral.get().replace(",", "."))
            except ValueError:
                messagebox.showerror("Parámetros", "Límite o umbral inválido.")
                return
            objetivos = buscar_lineas(filas, self.v_calibre.get())
            if not objetivos:
                messagebox.showinfo(
                    "Sin coincidencias",
                    "No encontré líneas de exportación con ese calibre.")
                return
            self.resultados = [evaluar(filas, f, limite, umbral)
                               for f in objetivos]
            self.combo["values"] = [r.fila.titulo for r in self.resultados]
            self.combo.current(0)
            self._mostrar_todos()
            self._dibujar(self.resultados[0].colores)

        def _cambiar_resultado(self, _e=None):
            i = self.combo.current()
            if i >= 0:
                self._dibujar(self.resultados[i].colores)

        def _mostrar_todos(self):
            t = self.texto
            t.config(state="normal")
            t.delete("1.0", "end")
            for r in self.resultados:
                f = r.fila
                t.insert("end", f.titulo + "\n", "negrita")
                if f.outs:
                    t.insert("end", f"Carga: {fmt(f.pct)}% en la línea → "
                                    f"{fmt(f.por_salida)}% por salida.\n")
                for a in r.avisos:
                    t.insert("end", "Aviso: " + a + "\n", "aviso")
                t.insert("end", r.veredicto + "\n",
                         "ok" if r.posible else "mal")
                for c, (v, razones) in r.candidatas.items():
                    if v:
                        t.insert("end", f"  • Salida {c}: SÍ. Está libre o "
                                        "con bajo flujo y cumple las "
                                        "reglas.\n", "ok")
                    else:
                        t.insert("end", f"  • Salida {c}: NO\n", "mal")
                        for z in razones:
                            t.insert("end", f"       - {z}\n")
                t.insert("end", "\n")
            t.config(state="disabled")

        # ------------------------------------------------- diagrama
        def _dibujar(self, colores):
            cv = self.canvas
            cv.delete("all")
            x0, paso, ancho = 50, 42, 40
            cv.create_text(x0, 14, anchor="w",
                           text="Parte trasera: descarte", fill="#555")
            cv.create_text(x0, 214, anchor="w",
                           text="Parte delantera: exportación", fill="#555")
            cv.create_rectangle(x0, 104, x0 + 14 * paso - 2, 130,
                                outline="#888", dash=(4, 3))
            cv.create_text(x0 + 7 * paso, 117,
                           text="Banda transportadora", fill="#555")
            for i, o in enumerate(range(4, 18)):
                x = x0 + i * paso
                color = colores.get(o, "gris")
                fill, borde = PALETA[color]
                if o in DESCARTES:
                    y1, y2 = 40, 98
                else:
                    y1, y2 = 136, 194
                cv.create_rectangle(x, y1, x + ancho, y2,
                                    fill=fill, outline=borde, width=1)
                cv.create_text(x + ancho / 2, (y1 + y2) / 2, text=str(o),
                               font=("TkDefaultFont", 11, "bold"),
                               fill=borde)
            leyenda = [("azul", "Salidas actuales"), ("rojo", "No habilitable"),
                       ("verde", "Habilitable")]
            x = 50
            for k, txt in leyenda:
                fill, borde = PALETA[k]
                cv.create_rectangle(x, 232, x + 14, 246, fill=fill,
                                    outline=borde)
                cv.create_text(x + 20, 239, anchor="w", text=txt,
                               fill="#333")
                x += 170

    App().mainloop()


if __name__ == "__main__":
    lanzar_gui()
