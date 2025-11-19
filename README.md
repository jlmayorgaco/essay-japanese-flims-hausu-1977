# 👻 Hausu (1977): Un Estudio Cultural y Computacional

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![LaTeX](https://img.shields.io/badge/built%20with-LaTeX-green.svg)
![Status](https://img.shields.io/badge/status-Draft-orange.svg)
![Hausu](https://img.shields.io/badge/aesthetic-Psychedelic-purple)

> **Título completo:** *Hausu (1977): Entre el Teatro Nō, la Televisualidad Pop y el Trauma Posbélico. Un estudio cultural, histórico y computacional.*

Este repositorio contiene el código fuente (LaTeX), los scripts de análisis de datos y la bibliografía del ensayo académico sobre la película de culto **Hausu** (House), dirigida por Nobuhiko Obayashi.

---

## 📖 Resumen del Proyecto

*Hausu* (1977) ha sido tradicionalmente descrita como una película "inclasificable". Este estudio sostiene que el film es una **singularidad estética** irrepetible producto de la convergencia de cuatro fuerzas culturales en el Japón de los 70:
1.  🎭 **Teatro Nō:** La dramaturgia del fantasma femenino.
2.  📺 **Televisualidad Pop:** La estética de los anuncios comerciales (CMs).
3.  🌸 **Pink Eiga:** La lógica del cuerpo femenino fragmentado.
4.  💥 **Trauma Posbélico:** La memoria de Hiroshima procesada mediante códigos infantiles.

## 📂 Estructura del Repositorio

```text
.
├── main.tex              # Archivo principal del documento
├── style.sty             # 🎨 "The Obayashi Cut": Estilo LaTeX personalizado (TikZ/PGFPlots)
├── references.bib        # Base de datos bibliográfica (BibLaTeX)
├── chapters/             # Capítulos del ensayo (tex files)
│   ├── ch1_introduccion.tex
│   ├── ...
│   └── ch10_conclusion.tex
├── analysis/             # 💻 Componente Computacional
│   ├── color_barcode.py  # Script para generar el código de barras de color del film
│   └── asl_analysis.R    # Análisis de longitud media de planos (Average Shot Length)
└── images/               # Gráficos generados y figuras
```

## 🎨 Sobre el Estilo LaTeX ("The Obayashi Cut")

Este *paper* utiliza un paquete de estilos personalizado (`style.sty`) diseñado para reflejar la estética de la película:
* **Film Strip Header:** Una cinta de película generada procedimentalmente con TiKZ en cada página.
* **Paleta de Colores:** Variables definidas (`hausuBlood`, `hausuOrange`, `hausuGreen`) extraídas de los fotogramas originales.
* **Cajas Semánticas:** Entornos personalizados (`popanalysis`, `bloodnote`) para destacar teoría fílmica y notas históricas.

## 💻 Metodología Computacional

Para justificar el enfoque cuantitativo del título, este repositorio incluye scripts que analizan la materialidad del film:

1.  **Análisis Cromático:** Extracción de la paleta de colores dominante por escena para evidenciar la influencia de la estética publicitaria saturada.
2.  **Métricas de Montaje:** Comparativa del ASL (*Average Shot Length*) de *Hausu* frente a *Jaws* (1975) para demostrar la frenética edición televisiva de Obayashi.

## 🚀 Compilación

Para generar el PDF desde el código fuente, necesitas una distribución de LaTeX (TeX Live, MacTeX o MikTeX) con `biber` instalado.

```bash
# 1. Compilar esqueleto
pdflatex main.tex

# 2. Procesar bibliografía y referencias cruzadas
biber main

# 3. Compilar texto y paginación
pdflatex main.tex
pdflatex main.tex
```

*Nota: El archivo `style.sty` requiere las librerías `tikz`, `pgfplots` y `fontawesome5`.*

## ✍️ Autor

**Jorge Luis Mayorga Taborda**
* Ingeniero Electrónico & Investigador Visual
* Intereses: Control Theory, Cine Japonés, Humanidades Digitales.

---
*"Es como si un niño hubiera dibujado una pesadilla y luego la hubiéramos filmado."* — Nobuhiko Obayashi
