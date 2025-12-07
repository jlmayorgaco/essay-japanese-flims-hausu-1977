from __future__ import annotations

import os
import csv
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any

# ============================================================
# Configuración global
# ============================================================

# Colores básicos
WHITE = (255, 255, 255)
LIGHT_GRAY = (225, 225, 225)
DARK_GRAY = (210, 210, 210)

# Estilo de figuras (fuentes grandes para dashboard 5K/6K)
FIG_DPI = 380
RC_BASE = {
    "font.size": 42,
    "axes.labelsize": 52,
    "axes.titlesize": 52,
    "xtick.labelsize": 42,
    "ytick.labelsize": 42,
    "legend.fontsize": 42,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
}

# “Slots” de thumbnails en la timeline
THUMB_W, THUMB_H = 224, 256  # ajusta si quieres thumbs más grandes


# ============================================================
# Utilidades de imagen / color
# ============================================================

def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def safe_resize(img: np.ndarray, max_side: int = 720) -> np.ndarray:
    """Redimensiona manteniendo aspecto para que el lado mayor sea <= max_side."""
    h, w = img.shape[:2]
    s = max(h, w)
    if s <= max_side:
        return img
    scale = max_side / s
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

def resize_exact(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Redimensiona a tamaño exacto (NO mantiene aspecto)."""
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

def resize_fit_pad(img: np.ndarray, target_w: int, target_h: int, pad_color: Tuple[int, int, int] = LIGHT_GRAY) -> np.ndarray:
    """
    Ajuste proporcional sin estirar:
    - escala para que quepa en el canvas
    - añade letterbox centrado hasta (target_h, target_w)
    """
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)
    y0 = (target_h - new_h) // 2
    x0 = (target_w - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas

def fit_by_height_crop_width(img: np.ndarray, slot_w: int, slot_h: int, pad_color: Tuple[int,int,int]=LIGHT_GRAY) -> np.ndarray:
    """
    Escala para igualar la ALTURA del slot (manteniendo aspecto). Luego:
      - si el ancho resultante >= slot_w -> recorte centrado a slot_w
      - si el ancho resultante  < slot_w -> padding lateral hasta slot_w
    Resultado: (slot_h, slot_w) SIN estirar.
    """
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.full((slot_h, slot_w, 3), pad_color, dtype=np.uint8)

    scale = slot_h / h
    new_w = max(1, int(round(w * scale)))
    resized = cv2.resize(img, (new_w, slot_h), interpolation=cv2.INTER_AREA)

    if new_w >= slot_w:
        x0 = (new_w - slot_w) // 2
        return resized[:, x0:x0+slot_w]
    else:
        out = np.full((slot_h, slot_w, 3), pad_color, dtype=np.uint8)
        x0 = (slot_w - new_w) // 2
        out[:, x0:x0+new_w] = resized
        return out

def pad_with_border(img: np.ndarray, border_px: int, color: Tuple[int, int, int] = LIGHT_GRAY) -> np.ndarray:
    if border_px <= 0:
        return img
    h, w = img.shape[:2]
    out = np.full((h + 2 * border_px, w + 2 * border_px, 3), color, dtype=np.uint8)
    out[border_px:border_px + h, border_px:border_px + w] = img
    return out

def put_title_band(img: np.ndarray, title: str, band_h: int = 90) -> np.ndarray:
    """Banda superior semitransparente con título (fuente escala con ancho)."""
    if not title or band_h <= 0:
        return img
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (img.shape[1], band_h), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
    w = img.shape[1]
    font_scale = max(1.6, w / 3200 * 2.2)  # escala suave con ancho del cover
    thickness = max(2, int(font_scale * 1.4))
    cv2.putText(
        img, title, (int(w * 0.03), int(band_h * 0.72)),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, WHITE, thickness, cv2.LINE_AA
    )
    return img


# ============================================================
# Utilidades generales
# ============================================================

def sec_to_hhmmss(t: float) -> str:
    t = max(0, float(t))
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(round(t % 60))
    return f"{h:02d}_{m:02d}_{s:02d}"

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def basename_title(video_path: str) -> str:
    base = os.path.basename(video_path)
    return os.path.splitext(base)[0].replace("_", " ")


# ============================================================
# Manejo de video
# ============================================================

def get_duration_seconds(cap: cv2.VideoCapture) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    return frames / fps if fps > 0 else 0.0

def read_frame_at_time(cap: cv2.VideoCapture, t_sec: float) -> np.ndarray | None:
    """Seek robusto a tiempo t_sec con fallback secuencial corto."""
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps > 0:
        frame_idx = int(round(t_sec * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    else:
        cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
    ok, frame = cap.read()
    if ok:
        return frame
    if fps > 0:
        start = max(0, frame_idx - 10)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for _ in range(25):
            ok, frame = cap.read()
            if ok:
                return frame
    return None

def read_random_cover_frame(cap: cv2.VideoCapture, duration: float, max_side: int = 1920) -> np.ndarray:
    t = duration * 0.5 if duration <= 1.0 else random.uniform(1.0, max(1.0, duration - 1.0))
    frame_bgr = read_frame_at_time(cap, t)
    if frame_bgr is None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame_bgr = cap.read()
        if not ok:
            raise RuntimeError("No se pudo leer ningún frame para la portada.")
    return bgr_to_rgb(safe_resize(frame_bgr, max_side))


# ============================================================
# Gráficas de series
# ============================================================

def calc_rgb_luma_series(cap: cv2.VideoCapture, duration: float, sample_every_sec: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t_samples = np.arange(0, duration, sample_every_sec, dtype=float)
    means = []
    for t in t_samples:
        frame_bgr = read_frame_at_time(cap, t)
        if frame_bgr is None:
            means.append((np.nan, np.nan, np.nan))
            continue
        mean_rgb = bgr_to_rgb(safe_resize(frame_bgr, 720)).reshape(-1, 3).mean(axis=0)
        means.append(tuple(mean_rgb.tolist()))
    r = np.array([m[0] for m in means])
    g = np.array([m[1] for m in means])
    b = np.array([m[2] for m in means])
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b  # Luma Rec.709
    return t_samples, r, g, b, y

def save_rgb_luma_plots(ts: np.ndarray, r: np.ndarray, g: np.ndarray, b: np.ndarray, y: np.ndarray, out_dir: str) -> Tuple[str, str]:
    old_rc = plt.rcParams.copy()
    plt.rcParams.update(RC_BASE)

    # RGB
    plt.figure(figsize=(40, 6.5))
    plt.plot(ts, r, label="Rojo", color="#D93025", linewidth=4)
    plt.plot(ts, g, label="Verde", color="#1E8E3E", linewidth=4)
    plt.plot(ts, b, label="Azul", color="#4285F4", linewidth=4)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("RGB(t)")
    plt.title("Curvas R, G, B promedio en el tiempo")
    plt.legend(loc="upper right")
    plt.tight_layout()
    rgb_plot = os.path.join(out_dir, "rgb_time_plot.png")
    plt.savefig(rgb_plot, dpi=FIG_DPI)
    plt.close()

    # Luma
    plt.figure(figsize=(40, 6.5))
    plt.plot(ts, y, label="Luminancia (Y, Rec.709)", color="#3C4043", linewidth=4)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("L(t)")
    plt.title("Luminancia promedio en el tiempo")
    plt.legend(loc="upper right")
    plt.tight_layout()
    luma_plot = os.path.join(out_dir, "luma_time_plot.png")
    plt.savefig(luma_plot, dpi=FIG_DPI)
    plt.close()

    plt.rcParams.update(old_rc)
    return rgb_plot, luma_plot


# =============== NUEVO: métricas adicionales vs tiempo ===============

def _shannon_entropy_gray(gray: np.ndarray) -> float:
    """Entropía de Shannon con histograma 256 bins (sin dependencias externas)."""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    p = hist / max(1.0, hist.sum())
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

def compute_extra_metrics(cap: cv2.VideoCapture, ts: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Lee frames en los tiempos 'ts' y calcula:
      - contraste (std de luminancia)
      - saturación media (HSV S)
      - darkness ratio (% Y < 30)
      - scene change (diff abs media entre frames consecutivos en gris)
      - edges activity (media de Canny)
      - hue_mean (media circular del Hue en OpenCV [0,180))
      - entropy (Shannon en gris)
    Devuelve diccionario con arrays alineados a ts.
    """
    contrast = []
    saturation = []
    darkness = []
    scene_change = []
    edges_act = []
    hue_mean = []
    entropy = []

    prev_gray = None

    for t in ts:
        f_bgr = read_frame_at_time(cap, float(t))
        if f_bgr is None:
            contrast.append(np.nan); saturation.append(np.nan); darkness.append(np.nan)
            scene_change.append(np.nan); edges_act.append(np.nan); hue_mean.append(np.nan); entropy.append(np.nan)
            prev_gray = None
            continue

        f_small = safe_resize(f_bgr, 720)
        gray = cv2.cvtColor(f_small, cv2.COLOR_BGR2GRAY)
        hsv  = cv2.cvtColor(f_small, cv2.COLOR_BGR2HSV)

        # Luminancia aproximada desde BGR (para contraste/oscuridad)
        y = 0.2126 * f_small[:,:,2] + 0.7152 * f_small[:,:,1] + 0.0722 * f_small[:,:,0]
        contrast.append(float(np.std(y)))
        darkness.append(float((y < 30).mean() * 100.0))  # porcentaje

        # Saturación media
        saturation.append(float(np.mean(hsv[:,:,1])))

        # Hue medio (media circular)
        h = hsv[:,:,0].astype(np.float32) * (np.pi / 90.0)  # 0..180 -> 0..2pi
        sin_m = float(np.mean(np.sin(h))); cos_m = float(np.mean(np.cos(h)))
        ang = np.arctan2(sin_m, cos_m)
        if ang < 0: ang += 2*np.pi
        hue_mean.append(float(ang * 90.0 / np.pi))  # vuelve a 0..180

        # Entropía
        entropy.append(_shannon_entropy_gray(gray))

        # Bordes (actividad)
        edges = cv2.Canny(gray, 80, 160)
        edges_act.append(float(np.mean(edges)))  # 0..255 de borde

        # Cambio de escena (diff con anterior)
        if prev_gray is None:
            scene_change.append(0.0)
        else:
            diff = np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32)))
            scene_change.append(float(diff))
        prev_gray = gray

    return {
        "contrast": np.array(contrast, dtype=float),
        "saturation": np.array(saturation, dtype=float),
        "darkness_pct": np.array(darkness, dtype=float),
        "scene_change": np.array(scene_change, dtype=float),
        "edges_activity": np.array(edges_act, dtype=float),
        "hue_mean": np.array(hue_mean, dtype=float),
        "entropy": np.array(entropy, dtype=float),
    }

def _save_metric_plot(ts: np.ndarray, y: np.ndarray, title: str, ylabel: str, color: str, out_path: str):
    """Guarda una figura individual con estilo consistente."""
    old_rc = plt.rcParams.copy()
    plt.rcParams.update(RC_BASE)
    plt.figure(figsize=(40, 12))
    plt.plot(ts, y, color=color, linewidth=4, label=title)
    plt.xlabel("Tiempo (s)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=FIG_DPI)
    plt.close()
    plt.rcParams.update(old_rc)

def save_extra_metric_plots(ts: np.ndarray, metrics: Dict[str, np.ndarray], out_dir: str) -> Dict[str, str]:
    """Genera PNGs por métrica y devuelve rutas."""
    paths = {}
    mapping = [
        ("contrast",      "Contraste (σ de L)",          "σ(L)",          "#5F6368", "contrast_time_plot.png"),
        ("saturation",    "Saturación promedio",         "S(t)",          "#1E8E3E", "saturation_time_plot.png"),
        ("darkness_pct",  "Porcentaje de oscuridad (<30)","% píxeles",    "#000000", "darkness_time_plot.png"),
        ("scene_change",  "Cambio de escena (Δ gris)",   "Δ(t)",          "#C5221F", "scenechange_time_plot.png"),
        ("edges_activity","Actividad de bordes (Canny)", "Nivel",         "#0B8043", "edges_time_plot.png"),
        ("hue_mean",      "Hue promedio (0–180)",        "Hue",           "#A142F4", "hue_time_plot.png"),
        ("entropy",       "Entropía visual (Shannon)",   "H",             "#3C4043", "entropy_time_plot.png"),
    ]
    for key, title, ylabel, color, fname in mapping:
        y = metrics.get(key, None)
        if y is None: continue
        out_path = os.path.join(out_dir, fname)
        _save_metric_plot(ts, y, title, ylabel, color, out_path)
        paths[key] = out_path

    return paths


# ============================================================
# Paletas / intervalos / timeline
# ============================================================

def dominant_colors_kmeans(img_rgb: np.ndarray, k: int = 8, max_samples: int = 300_000) -> Tuple[np.ndarray, np.ndarray]:
    data = img_rgb.reshape(-1, 3)
    if data.shape[0] > max_samples:
        idx = np.random.choice(data.shape[0], size=max_samples, replace=False)
        data = data[idx]
    data = np.float32(data)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        data, K=k, bestLabels=None, criteria=criteria, attempts=3, flags=cv2.KMEANS_PP_CENTERS
    )
    labels = labels.flatten()
    counts = np.bincount(labels, minlength=k).astype(np.int64)
    order = np.argsort(-counts)
    return centers[order], counts[order]

def make_palette_image(colors_rgb: np.ndarray, counts: np.ndarray, width: int = 1400, height: int = 140) -> np.ndarray:
    palette = np.zeros((height, width, 3), dtype=np.uint8)
    total = counts.sum() if counts.sum() > 0 else len(counts)
    fracs = (counts / total).tolist()
    x0 = 0
    for c, frac in zip(colors_rgb, fracs):
        w = max(1, int(round(frac * width)))
        x1 = min(width, x0 + w)
        palette[:, x0:x1, :] = np.uint8(np.clip(c, 0, 255))
        x0 = x1
    if x0 < width and len(colors_rgb) > 0:
        palette[:, x0:, :] = np.uint8(np.clip(colors_rgb[-1], 0, 255))
    return palette

def process_intervals(cap: cv2.VideoCapture, duration: float, intervals: int, palette_k: int, interval_window_ratio: float, out_dir: str) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    edges = np.linspace(0.0, duration, intervals + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    items: List[Dict[str, Any]] = []
    for idx, c in enumerate(centers, start=1):
        # Frame representativo
        if interval_window_ratio > 0:
            half = 0.5 * interval_window_ratio * (edges[1] - edges[0])
            frames = []
            for tt in np.linspace(max(0.0, c - half), min(duration, c + half), 7):
                f = read_frame_at_time(cap, float(tt))
                if f is not None:
                    frames.append(bgr_to_rgb(safe_resize(f, 720)))
            if not frames:
                continue
            used_rgb = np.uint8(np.clip(np.stack(frames, axis=0).astype(np.float32).mean(axis=0), 0, 255))
        else:
            f = read_frame_at_time(cap, float(c))
            if f is None:
                continue
            used_rgb = bgr_to_rgb(safe_resize(f, 720))

        # Paleta
        colors, counts = dominant_colors_kmeans(used_rgb, k=palette_k)
        palette_img = make_palette_image(colors, counts, width=1400, height=140)

        # Guardar
        ts_str = sec_to_hhmmss(c)
        tag = f"f{idx}"
        pal_path = os.path.join(out_dir, f"{tag}_palette_{ts_str}.png")
        scr_path = os.path.join(out_dir, f"{tag}_screenshot_{ts_str}.jpg")
        cv2.imwrite(pal_path, cv2.cvtColor(palette_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(scr_path, cv2.cvtColor(used_rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 92])

        items.append({
            "interval_index": idx,
            "frame_tag": tag,
            "start_sec": float(edges[idx - 1]),
            "center_sec": float(c),
            "end_sec": float(edges[idx]),
            "timestamp_hh_mm_ss": ts_str,
            "palette_image": pal_path,
            "screenshot_image": scr_path,
        })

    return items, centers

def build_timeline(
    items: List[Dict[str, Any]],
    centers: np.ndarray,
    duration: float,
    out_dir: str,
    filename_prefix: str,
    source: str = "thumbs",  # "thumbs" | "palettes"
    show_markers: bool = True,
) -> Tuple[str | None, str | None]:
    """
    Devuelve (timeline_base_png, timeline_axes_png).
    Para 'thumbs' devolvemos la tira base y (opcionalmente) None en axes (no hay ejes).
    """
    if not items:
        return None, None

    tl_tag = sec_to_hhmmss(float(centers[0])) if len(centers) else "00_00_00"
    base_path, axes_path = None, None

    strip_list: List[np.ndarray] = []
    if source == "thumbs":
        # ancho fijo por slot, alto fijo de la tira; recorte/pad del ancho sin estirar.
        slot_w, slot_h = THUMB_W, THUMB_H
        for it in items:
            img = cv2.cvtColor(cv2.imread(it["screenshot_image"]), cv2.COLOR_BGR2RGB)
            if img is None:
                continue
            thumb = fit_by_height_crop_width(img, slot_w=slot_w, slot_h=slot_h, pad_color=LIGHT_GRAY)
            strip_list.append(thumb)
    else:
        slot_w, slot_h = THUMB_W, THUMB_H
        for it in items:
            pal = cv2.cvtColor(cv2.imread(it["palette_image"]), cv2.COLOR_BGR2RGB)
            if pal is None:
                continue
            strip_list.append(fit_by_height_crop_width(pal, slot_w=slot_w, slot_h=slot_h, pad_color=WHITE))

    if not strip_list:
        return None, None

    gap = np.full((THUMB_H, 10, 3), WHITE, dtype=np.uint8)
    strip = strip_list[0]
    for im in strip_list[1:]:
        strip = np.hstack([strip, gap, im])

    # Base (sin ejes)
    base_name = f"{'thumbs' if source=='thumbs' else 'palettes'}_timeline_{filename_prefix}_{tl_tag}.png"
    base_path = os.path.join(out_dir, base_name)
    cv2.imwrite(base_path, cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))

    if source == "thumbs":
        # SIN ejes para thumbnails
        return base_path, None

    # Con ejes (para paletas)
    old_rc = plt.rcParams.copy()
    plt.rcParams.update(RC_BASE)
    plt.figure(figsize=(34, 6))
    extent = [0, duration, 0, 1]
    plt.imshow(strip, extent=extent, origin="upper", aspect="auto")
    plt.yticks([])
    plt.xlabel("Tiempo (s)")
    plt.title("Timeline de paletas")
    if show_markers:
        for i, c in enumerate(centers, start=1):
            plt.axvline(c, color="#9AA0A6", linestyle=":", linewidth=1.4)
            plt.text(c, 1.02, f"f{i}", ha="center", va="bottom")
    plt.tight_layout()
    axes_name = f"palettes_timeline_axes_{filename_prefix}_{tl_tag}.png"
    axes_path = os.path.join(out_dir, axes_name)
    plt.savefig(axes_path, dpi=FIG_DPI)
    plt.close()
    plt.rcParams.update(old_rc)

    return base_path, axes_path

def build_sidebar(items: List[Dict[str, Any]], out_dir: str, filename_prefix: str, border_px: int) -> str | None:
    """Columna de paletas más grande para mejor legibilidad."""
    if not items:
        return None
    try:
        pals = [cv2.cvtColor(cv2.imread(it["palette_image"]), cv2.COLOR_BGR2RGB) for it in items]
        pals = [resize_exact(p, 1400, 140) for p in pals]  # ↑ más ancho y alto
        max_w = max(p.shape[1] for p in pals)
        normed = [resize_exact(p, max_w, p.shape[0]) for p in pals]
        sep = np.full((10, max_w, 3), (245, 245, 245), dtype=np.uint8)
        out = normed[0]
        for im in normed[1:]:
            out = np.vstack([out, sep, im])
        out = pad_with_border(out, border_px, color=LIGHT_GRAY)
        path = os.path.join(out_dir, f"palettes_sidebar_{filename_prefix}.png")
        cv2.imwrite(path, cv2.COLOR_RGB2BGR and cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        return path
    except Exception as e:
        print("[WARN] No se pudo construir el sidebar:", e)
        return None


# ============================================================
# Composición del dashboard
# ============================================================

def compose_dashboard(
    cover_path: str,
    rgb_plot_path: str,
    luma_plot_path: str,
    timeline_path_for_dashboard: str,  # base (thumbs) o con ejes (paletas)
    sidebar_path: str | None,
    out_dir: str,
    filename_prefix: str,
    dashboard_w: int,
    dashboard_h: int,
    border_px: int,
    outer_border_px: int,
    name_ts: str,
    sidebar_frac: float = 0.34,  # sidebar más ancha
) -> str | None:
    """Construye el dashboard final (ajuste proporcional; sin estirar fotos)."""
    try:
        cover = cv2.cvtColor(cv2.imread(cover_path), cv2.COLOR_BGR2RGB)
        rgb_plot = cv2.cvtColor(cv2.imread(rgb_plot_path), cv2.COLOR_BGR2RGB)
        luma_plot = cv2.cvtColor(cv2.imread(luma_plot_path), cv2.COLOR_BGR2RGB)
        timeline_img = cv2.cvtColor(cv2.imread(timeline_path_for_dashboard), cv2.COLOR_BGR2RGB)
        sidebar = cv2.cvtColor(cv2.imread(sidebar_path), cv2.COLOR_BGR2RGB) if sidebar_path and os.path.exists(sidebar_path) else None
    except Exception as e:
        print("[WARN] Error cargando componentes del dashboard:", e)
        return None

    W, H = dashboard_w, dashboard_h
    left_w = int(W * (1.0 - sidebar_frac))
    right_w = W - left_w

    if sidebar is not None:
        sidebar = resize_fit_pad(sidebar, right_w, H, pad_color=(245, 245, 245))
    else:
        sidebar = np.full((H, right_w, 3), 250, dtype=np.uint8)

    top_h = int(H * 0.5)
    bottom_h = H - top_h
    row_h = bottom_h // 3

    cover = resize_fit_pad(cover, left_w, top_h, pad_color=LIGHT_GRAY)
    cover = pad_with_border(cover, border_px, color=LIGHT_GRAY)

    rgb_plot = pad_with_border(resize_exact(rgb_plot, left_w, row_h), border_px, color=LIGHT_GRAY)
    luma_plot = pad_with_border(resize_exact(luma_plot, left_w, row_h), border_px, color=LIGHT_GRAY)
    timeline = pad_with_border(resize_fit_pad(timeline_img, left_w, bottom_h - 2 * row_h, pad_color=WHITE), border_px, color=LIGHT_GRAY)

    sep_h = np.full((border_px, left_w + 2 * border_px, 3), LIGHT_GRAY, dtype=np.uint8)
    left_panel = np.vstack([cover, sep_h, rgb_plot, sep_h, luma_plot, sep_h, timeline])
    left_panel = resize_exact(left_panel, left_w + 2 * border_px, H)

    sep_v = np.full((H, border_px, 3), LIGHT_GRAY, dtype=np.uint8)
    dash = np.hstack([left_panel, sep_v, sidebar])
    dash = pad_with_border(dash, outer_border_px, color=DARK_GRAY)

    out_path = os.path.join(out_dir, f"dashboard_{filename_prefix}_{name_ts}.png")
    cv2.imwrite(out_path, cv2.cvtColor(dash, cv2.COLOR_RGB2BGR))
    return out_path


# ============================================================
# Pipeline principal
# ============================================================

def process_video_colors(
    video_path: str,
    out_dir: str = "video_colors_out",
    sample_every_sec: float = 60.0,
    intervals: int = 10,
    palette_k: int = 8,
    interval_window_ratio: float = 0.0,
    filename_prefix: str = "f_all",
    make_dashboard: bool = True,
    dashboard_width_px: int = 5760,
    dashboard_height_px: int = 3240,
    border_px: int = 12,
    outer_border_px: int = 18,
    timeline_source: str = "thumbs",  # "thumbs" o "palettes"
    show_interval_markers: bool = True,
) -> Dict[str, Any]:
    """
    Genera:
      - CSV y plots RGB/Luma
      - Paletas + screenshots por intervalo
      - Timeline (thumbnails o paletas)
      - Sidebar de paletas (más grande)
      - Dashboard final (opcional)
      - NUEVO: PNGs de métricas (contraste, saturación, oscuridad, cambio de escena, bordes, hue, entropía)
    """
    safe_mkdir(out_dir)
    movie_title = basename_title(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    duration = get_duration_seconds(cap)
    if duration <= 0:
        raise RuntimeError("No fue posible obtener la duración del video.")

    # Cover
    cover_rgb = put_title_band(read_random_cover_frame(cap, duration, max_side=1920), movie_title, band_h=90)
    cover_path = os.path.join(out_dir, f"cover_{filename_prefix}.jpg")
    cv2.imwrite(cover_path, cv2.cvtColor(cover_rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    # Series y gráficos RGB/Luma
    ts, r, g, b, y = calc_rgb_luma_series(cap, duration, sample_every_sec)
    rgb_plot_path, luma_plot_path = save_rgb_luma_plots(ts, r, g, b, y, out_dir)

    # CSV series
    rgb_csv_path = os.path.join(out_dir, "rgb_time_series.csv")
    with open(rgb_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_sec", "R_mean", "G_mean", "B_mean", "Y_luma"])
        for t, rv, gv, bv, yv in zip(ts, r, g, b, y):
            w.writerow([t, rv, gv, bv, yv])

    # NUEVO: Métricas extra y sus plots (no afectan dashboard)
    extra = compute_extra_metrics(cap, ts)
    # Añade temperatura de color a partir de r y b ya calculados
    extra["color_temp_index"] = (b - r) / (b + r + 1e-6)

    extra_plots = save_extra_metric_plots(ts, extra, out_dir)
    # Guardar también la temperatura de color
    cti_path = os.path.join(out_dir, "colortemp_time_plot.png")
    _save_metric_plot(ts, extra["color_temp_index"], "Índice temperatura de color (B−R)/(B+R)", "Índice", "#1A73E8", cti_path)
    extra_plots["color_temp_index"] = cti_path

    # Intervalos (paletas + screenshots)
    items, centers = process_intervals(cap, duration, intervals, palette_k, interval_window_ratio, out_dir)

    # CSV paletas
    pal_csv_path = os.path.join(out_dir, "palettes_by_interval.csv")
    with open(pal_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["interval_index", "frame_tag", "start_sec", "center_sec", "end_sec", "timestamp_hh_mm_ss", "palette_image", "screenshot_image"])
        for it in items:
            w.writerow([it["interval_index"], it["frame_tag"], it["start_sec"], it["center_sec"], it["end_sec"], it["timestamp_hh_mm_ss"], it["palette_image"], it["screenshot_image"]])

    # Timeline (base + con ejes para paletas; thumbs sin ejes)
    base_tl, axes_tl = build_timeline(
        items, centers, duration, out_dir, filename_prefix,
        source=timeline_source, show_markers=show_interval_markers
    )

    # Sidebar
    sidebar_path = build_sidebar(items, out_dir, filename_prefix, border_px=border_px)

    # Dashboard
    dashboard_path = None
    if make_dashboard and (base_tl or axes_tl):
        name_ts = sec_to_hhmmss(duration / 2.0)  # etiqueta neutra
        tl_for_dashboard = base_tl if timeline_source == "thumbs" else (axes_tl or base_tl)
        dashboard_path = compose_dashboard(
            cover_path, rgb_plot_path, luma_plot_path, tl_for_dashboard, sidebar_path,
            out_dir, filename_prefix, dashboard_width_px, dashboard_height_px,
            border_px, outer_border_px, name_ts, sidebar_frac=0.34
        )

    cap.release()
    return {
        "duration_sec": duration,
        "rgb_csv": rgb_csv_path,
        "rgb_plot": rgb_plot_path,
        "luma_plot": luma_plot_path,
        "palettes_csv": pal_csv_path,
        "timeline_base": base_tl,
        "timeline_axes": axes_tl,
        "palettes_sidebar": sidebar_path,
        "cover_image": cover_path,
        "dashboard": dashboard_path,
        # Rutas de nuevas figuras:
        "extra_plots": extra_plots,
        "out_dir": out_dir,
    }


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="RGB/Luma vs tiempo, paletas por intervalos, timeline (thumbs o paletas), dashboard Hi-Fi y métricas extra."
    )
    parser.add_argument("video", help="Ruta al archivo .mp4")
    parser.add_argument("--out", default="video_colors_out", help="Carpeta de salida")
    parser.add_argument("--every", type=float, default=60.0, help="Muestreo de frames para RGB/Luma (segundos)")
    parser.add_argument("--intervals", type=int, default=10, help="Número de intervalos para paletas / thumbs")
    parser.add_argument("--k", type=int, default=8, help="Número de colores en cada paleta (K)")
    parser.add_argument("--window-ratio", type=float, default=0.0, help="0 = solo centro; >0 promedio en fracción del intervalo (ej. 0.5)")
    parser.add_argument("--prefix", default="f_all", help="Prefijo global para nombres agregados")
    parser.add_argument("--dash-width", type=int, default=5760, help="Ancho del dashboard (px)")
    parser.add_argument("--dash-height", type=int, default=3240, help="Alto del dashboard (px)")
    parser.add_argument("--border", type=int, default=12, help="Borde interno / separadores (px)")
    parser.add_argument("--outer-border", type=int, default=18, help="Borde exterior del dashboard (px)")
    parser.add_argument("--no-dashboard", action="store_true", help="No construir el dashboard")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--timeline-thumbs", action="store_true", help="Usar thumbnails en la timeline (sin ejes)")
    group.add_argument("--timeline-palettes", action="store_true", help="Usar paletas en la timeline (con ejes)")
    parser.add_argument("--no-interval-markers", action="store_true", help="Ocultar marcadores f1..fN (solo si usas paletas con ejes)")

    args = parser.parse_args()
    source = "palettes" if args.timeline_palettes else "thumbs"

    res = process_video_colors(
        video_path=args.video,
        out_dir=args.out,
        sample_every_sec=args.every,
        intervals=args.intervals,
        palette_k=args.k,
        interval_window_ratio=args.window_ratio,
        filename_prefix=args.prefix,
        make_dashboard=not args.no_dashboard,
        dashboard_width_px=args.dash_width,
        dashboard_height_px=args.dash_height,
        border_px=args.border,
        outer_border_px=args.outer_border,
        timeline_source=source,
        show_interval_markers=not args.no_interval_markers,
    )

    print("\n✔ Proceso completado.")
    print("Carpeta:", res["out_dir"])
    for k, v in res.items():
        if k != "out_dir":
            print(f"- {k}: {v}")
