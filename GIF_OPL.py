# ============================================================
# GIFOPL.py — Python pur (offline)
# PDN (coller / fichier) + FEN -> GIF animé OPL premium
# Python 3.12 + Pillow requis
# ============================================================

import os
import re
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageTk


# =========================
# DOSSIERS / CONFIG
# =========================

ROOT_DIR = r"C:\Users\Jacky\Documents\GIF_OPL"
DEFAULT_TEMP_DIR = os.path.join(ROOT_DIR, "temp")
DEFAULT_GIF_DIR = os.path.join(ROOT_DIR, "gif")
CONFIG_FILE = os.path.join(ROOT_DIR, "config.json")

os.makedirs(ROOT_DIR, exist_ok=True)
os.makedirs(DEFAULT_TEMP_DIR, exist_ok=True)
os.makedirs(DEFAULT_GIF_DIR, exist_ok=True)


def load_config() -> dict:
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_config(cfg: dict) -> None:
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# =========================
# MANOURY 1..50 -> (r,c)
# =========================

def manoury_to_rc(n: int) -> Tuple[int, int]:
    n -= 1
    r = n // 5
    k = n % 5
    c = 1 + 2 * k if r % 2 == 0 else 2 * k
    return r, c


def rc_to_manoury(r: int, c: int) -> Optional[int]:
    if r < 0 or r > 9 or c < 0 or c > 9:
        return None
    if (r + c) % 2 != 1:
        return None
    k = (c - 1) // 2 if r % 2 == 0 else c // 2
    if k < 0 or k > 4:
        return None
    return r * 5 + k + 1


# =========================
# PDN / FEN extraction
# =========================

TAG_RE = re.compile(r'^\s*\[(\w+)\s+"([^"]*)"\s*\]\s*$')
MOVE_RE = re.compile(r"^\d{1,2}([\-x]\d{1,2})+$")


def parse_tags(text: str) -> Dict[str, str]:
    tags = {}
    for ln in text.replace("\r", "").split("\n"):
        m = TAG_RE.match(ln)
        if m:
            tags[m.group(1).strip().lower()] = m.group(2).strip()
    return tags


def extract_moves(text: str) -> List[str]:
    # remove tags, comments
    t = re.sub(r"\[[^\]]+\]", " ", text)
    t = re.sub(r"\{[^}]+\}", " ", t)
    t = re.sub(r";[^\n]*", " ", t)
    # remove results
    t = re.sub(r"\b(1-0|0-1|1/2-1/2|\*)\b", " ", t)
    tokens = t.replace("\n", " ").split()
    moves = [tok for tok in tokens if MOVE_RE.match(tok)]
    return moves


def extract_fen(text: str) -> Optional[str]:
    # [FEN "W:W31-50:B1-20"] typical
    m = re.search(r"\[FEN\s+\"([^\"]+)\"\]", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # alternative: FEN: W:...
    m2 = re.search(r"\bFEN\s*:\s*([WB]:[^\s]+)", text, re.IGNORECASE)
    if m2:
        return m2.group(1).strip()
    return None


# =========================
# Position / Game
# pieces:
#   "w" = pion blanc, "W" = dame blanche
#   "b" = pion noir,  "B" = dame noire
# =========================

def initial_position() -> Dict[int, str]:
    pos: Dict[int, str] = {}
    for i in range(1, 21):
        pos[i] = "b"
    for i in range(31, 51):
        pos[i] = "w"
    return pos


def crown_if_needed(piece: str, to_sq: int) -> str:
    r, _ = manoury_to_rc(to_sq)
    if piece == "w" and r == 0:
        return "W"
    if piece == "b" and r == 9:
        return "B"
    return piece


def between_square(a: int, b: int) -> Optional[int]:
    ra, ca = manoury_to_rc(a)
    rb, cb = manoury_to_rc(b)
    dr = rb - ra
    dc = cb - ca
    if abs(dr) == 2 and abs(dc) == 2:
        return rc_to_manoury(ra + dr // 2, ca + dc // 2)
    return None


def apply_move(pos: Dict[int, str], move: str, side: str) -> Tuple[int, int]:
    # returns (from,to) for arrow
    is_cap = "x" in move
    parts = [int(x) for x in move.split("x" if is_cap else "-")]
    frm = parts[0]

    if frm not in pos:
        raise ValueError(f"Coup invalide: {move} (case {frm} vide)")

    piece = pos[frm]
    # side check
    if side == "w" and piece not in ("w", "W"):
        raise ValueError(f"Coup invalide: {move} (ce n'est pas aux blancs)")
    if side == "b" and piece not in ("b", "B"):
        raise ValueError(f"Coup invalide: {move} (ce n'est pas aux noirs)")

    # move
    del pos[frm]

    if not is_cap:
        to = parts[1]
        pos[to] = crown_if_needed(piece, to)
        return frm, to

    cur = frm
    for to in parts[1:]:
        mid = between_square(cur, to)
        if mid is not None and mid in pos:
            del pos[mid]
        cur = to

    pos[cur] = crown_if_needed(piece, cur)
    return frm, cur


# =========================
# FEN parsing (FMJD tolérant)
# Ex: "W:W31-50:B1-20" or "B:W31-32,BK10"
# Kings: "K" before square: WK34 or BK10 or K34 after stripping color
# =========================

def parse_fen(fen: str) -> Tuple[str, Dict[int, str]]:
    fen = fen.strip()
    m = re.match(r"^([WB])\s*:\s*(.+)$", fen, flags=re.I)
    if not m:
        raise ValueError("FEN invalide (attendu 'W:...' ou 'B:...')")

    side = "w" if m.group(1).upper() == "W" else "b"
    rest = m.group(2)
    segments = rest.split(":")

    pos: Dict[int, str] = {}

    def add_segment(seg: str, color: str) -> None:
        s = seg.strip()
        if not s:
            return
        up = s.upper()
        if up.startswith("W"):
            s = s[1:].strip()
            color = "w"
        elif up.startswith("B"):
            s = s[1:].strip()
            color = "b"
        if not s:
            return

        items = [x.strip() for x in s.split(",") if x.strip()]
        for it in items:
            king = False
            t = it.upper()
            if t.startswith("K"):
                king = True
                t = t[1:]

            if "-" in t:
                a_str, b_str = t.split("-", 1)
                a = int(a_str)
                b = int(b_str)
                for sq in range(min(a, b), max(a, b) + 1):
                    if color == "w":
                        pos[sq] = "W" if king else "w"
                    else:
                        pos[sq] = "B" if king else "b"
            else:
                sq = int(t)
                if color == "w":
                    pos[sq] = "W" if king else "w"
                else:
                    pos[sq] = "B" if king else "b"

    # parse segments that begin with W or B
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        up = seg.upper()
        if up.startswith("W"):
            add_segment(seg, "w")
        elif up.startswith("B"):
            add_segment(seg, "b")

    return side, pos


# =========================
# Rendu "OPL premium" (simple)
# =========================

LIGHT = (240, 225, 210)
DARK = (155, 100, 60)

BG_TOP = (255, 255, 255)
BG_BOTTOM = (250, 246, 240)


def gradient_bg(size: int) -> Image.Image:
    img = Image.new("RGB", (size, size), BG_TOP)
    px = img.load()
    for y in range(size):
        t = y / max(1, size - 1)
        r = int(BG_TOP[0] + (BG_BOTTOM[0] - BG_TOP[0]) * t)
        g = int(BG_TOP[1] + (BG_BOTTOM[1] - BG_TOP[1]) * t)
        b = int(BG_TOP[2] + (BG_BOTTOM[2] - BG_TOP[2]) * t)
        for x in range(size):
            px[x, y] = (r, g, b)
    return img.convert("RGBA")


def piece_image(diam: int, is_white: bool, is_king: bool) -> Image.Image:
    img = Image.new("RGBA", (diam, diam), (0, 0, 0, 0))

    # soft shadow
    shadow = Image.new("RGBA", (diam, diam), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.ellipse([diam * 0.18, diam * 0.62, diam * 0.82, diam * 0.90], fill=(0, 0, 0, 90))
    shadow = shadow.filter(ImageFilter.GaussianBlur(6))
    img.alpha_composite(shadow)

    # disc gradient-like
    base1 = (246, 239, 231) if is_white else (42, 32, 26)
    base2 = (227, 211, 200) if is_white else (20, 17, 15)
    ring = (202, 169, 146) if is_white else (91, 64, 51)

    disc = Image.new("RGBA", (diam, diam), (0, 0, 0, 0))
    px = disc.load()
    cx = cy = (diam - 1) / 2.0
    r0 = diam * 0.36

    for y in range(diam):
        for x in range(diam):
            dx = x - cx
            dy = y - cy
            rr = math.sqrt(dx * dx + dy * dy)
            if rr <= r0:
                t = min(1.0, rr / r0)
                hl = max(0.0, 1.0 - math.sqrt((dx + diam * 0.14) ** 2 + (dy + diam * 0.14) ** 2) / (r0 * 1.1))
                r = int(base1[0] + (base2[0] - base1[0]) * t + 40 * hl)
                g = int(base1[1] + (base2[1] - base1[1]) * t + 40 * hl)
                b = int(base1[2] + (base2[2] - base1[2]) * t + 40 * hl)
                px[x, y] = (min(255, r), min(255, g), min(255, b), 255)

    dd = ImageDraw.Draw(disc)
    dd.ellipse([cx - r0, cy - r0, cx + r0, cy + r0], outline=ring + (255,), width=max(2, int(diam * 0.04)))
    img.alpha_composite(disc)

    if is_king:
        mark = Image.new("RGBA", (diam, diam), (0, 0, 0, 0))
        md = ImageDraw.Draw(mark)
        rr = r0 * 0.55
        fill = (176, 123, 72, 220) if is_white else (202, 169, 146, 220)
        md.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=fill, outline=(0, 0, 0, 80), width=2)
        img.alpha_composite(mark)

    return img


def draw_arrow_opl(img: Image.Image, p1: Tuple[float, float], p2: Tuple[float, float]) -> None:
    # arrow drawn over pieces for readability
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    dist = math.hypot(dx, dy)
    if dist < 2:
        return

    ux = dx / dist
    uy = dy / dist
    px = -uy
    py = ux

    # Style (proche de ton exemple)
    shaft_w = 12
    head_len = 30
    head_w = 24
    alpha = 110
    color = (120, 140, 160, alpha)

    # pad to avoid overlapping piece centers too much
    pad = 14
    sx1 = x1 + ux * pad
    sy1 = y1 + uy * pad
    sx2 = x2 - ux * pad
    sy2 = y2 - uy * pad

    hx = sx2 - ux * head_len
    hy = sy2 - uy * head_len

    body = [
        (sx1 + px * shaft_w, sy1 + py * shaft_w),
        (hx + px * shaft_w, hy + py * shaft_w),
        (hx - px * shaft_w, hy - py * shaft_w),
        (sx1 - px * shaft_w, sy1 - py * shaft_w),
    ]
    d.polygon(body, fill=color)

    head = [
        (sx2, sy2),
        (hx + px * head_w, hy + py * head_w),
        (hx - px * head_w, hy - py * head_w),
    ]
    d.polygon(head, fill=color)

    overlay = overlay.filter(ImageFilter.GaussianBlur(1.4))
    img.alpha_composite(overlay)


def render_frame(pos: Dict[int, str], arrow: Optional[Tuple[int, int]], size: int = 600) -> Image.Image:
    img = gradient_bg(size)
    d = ImageDraw.Draw(img)

    sq = size // 10

    # board
    for r in range(10):
        for c in range(10):
            col = DARK if (r + c) % 2 == 1 else LIGHT
            d.rectangle((c * sq, r * sq, (c + 1) * sq, (r + 1) * sq), fill=col)

    # pieces first
    for sqr, p in pos.items():
        r, c = manoury_to_rc(sqr)
        diam = int(sq * 0.82)
        is_white = p in ("w", "W")
        is_king = p in ("W", "B")
        pi = piece_image(diam, is_white, is_king)
        x = int(c * sq + (sq - diam) / 2)
        y = int(r * sq + (sq - diam) / 2)
        img.alpha_composite(pi, (x, y))

    # arrow on top (for readability)
    if arrow is not None:
        frm, to = arrow
        r1, c1 = manoury_to_rc(frm)
        r2, c2 = manoury_to_rc(to)
        p1 = (c1 * sq + sq / 2, r1 * sq + sq / 2)
        p2 = (c2 * sq + sq / 2, r2 * sq + sq / 2)
        draw_arrow_opl(img, p1, p2)

    return img


# =========================
# GIF export
# =========================

def next_gif_path(gif_dir: str) -> str:
    i = 1
    while os.path.exists(os.path.join(gif_dir, f"GIFOPL{i:03}.gif")):
        i += 1
    return os.path.join(gif_dir, f"GIFOPL{i:03}.gif")


def save_gif(frames: List[Image.Image], gif_path: str, delay_ms: int) -> None:
    if not frames:
        raise ValueError("Aucune frame à exporter.")
    duration = max(20, int(delay_ms))

    # convert to P mode for GIF
    pal_frames = []
    for im in frames:
        pal = im.convert("P", palette=Image.Palette.ADAPTIVE, colors=256)
        pal_frames.append(pal)

    pal_frames[0].save(
        gif_path,
        save_all=True,
        append_images=pal_frames[1:],
        duration=duration,
        loop=0,
        optimize=False,
        disposal=2,
    )


# =========================
# UI Tkinter
# =========================

class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("GIFOPL — Python pur")
        self.root.geometry("1120x720")

        cfg = load_config()
        self.temp_dir = cfg.get("temp_dir", DEFAULT_TEMP_DIR)
        self.gif_dir = cfg.get("gif_dir", DEFAULT_GIF_DIR)
        self.delay_ms = int(cfg.get("delay_ms", 520))
        self.save_png = bool(cfg.get("save_png", True))
        self.size = int(cfg.get("size", 600))

        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.gif_dir, exist_ok=True)

        self.frames: List[Image.Image] = []
        self.cur = 0

        self._build_ui()

        # show initial position immediately
        self._show_image(render_frame(initial_position(), None, size=self.size))

    def _build_ui(self):
        left = tk.Frame(self.root, padx=10, pady=10)
        left.pack(side="left", fill="y")

        tk.Label(left, text="PDN (coller) ou charger un fichier", font=("Arial", 10, "bold")).pack(anchor="w")

        self.txt = tk.Text(left, width=48, height=18, font=("Consolas", 10))
        self.txt.pack(anchor="w", pady=(6, 8))

        row = tk.Frame(left)
        row.pack(anchor="w", pady=(0, 8))
        tk.Button(row, text="Charger PDN", command=self.load_file).pack(side="left")
        tk.Button(row, text="Vider", command=lambda: self.txt.delete("1.0", "end")).pack(side="left", padx=8)

        opt = tk.Frame(left)
        opt.pack(anchor="w", pady=(6, 6), fill="x")
        tk.Label(opt, text="Vitesse (ms)", font=("Arial", 9, "bold")).grid(row=0, column=0, sticky="w")
        self.speed_var = tk.StringVar(value=str(self.delay_ms))
        tk.Entry(opt, textvariable=self.speed_var, width=8).grid(row=0, column=1, sticky="w", padx=(6, 18))

        self.png_var = tk.IntVar(value=1 if self.save_png else 0)
        tk.Checkbutton(left, text="Enregistrer aussi les frames PNG dans TEMP", variable=self.png_var).pack(anchor="w")

        row2 = tk.Frame(left)
        row2.pack(anchor="w", pady=(6, 8))
        tk.Button(row2, text="Choisir TEMP", command=self.pick_temp).pack(side="left")
        tk.Button(row2, text="Choisir GIF", command=self.pick_gif).pack(side="left", padx=8)

        self.path_lbl = tk.Label(left, text=self._paths_text(), font=("Arial", 9), fg="#555", justify="left", wraplength=420)
        self.path_lbl.pack(anchor="w", pady=(0, 10))

        row3 = tk.Frame(left)
        row3.pack(anchor="w", pady=(0, 10))
        tk.Button(row3, text="Prévisualiser", command=self.preview).pack(side="left")
        tk.Button(row3, text="Exporter GIF", command=self.export_gif).pack(side="left", padx=8)

        self.status = tk.Label(left, text="", font=("Arial", 9), fg="#555", wraplength=420, justify="left")
        self.status.pack(anchor="w")

        right = tk.Frame(self.root, padx=10, pady=10)
        right.pack(side="right", fill="both", expand=True)

        self.meta_lbl = tk.Label(right, text="—", font=("Arial", 10, "bold"), fg="#5a2f1a")
        self.meta_lbl.pack(anchor="w")

        self.canvas = tk.Canvas(right, bg="#f4efe9", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        ctrl = tk.Frame(right)
        ctrl.pack(fill="x", pady=(6, 0))
        tk.Label(ctrl, text="Frame").pack(side="left")
        self.slider = tk.Scale(ctrl, from_=0, to=0, orient="horizontal", command=self.on_slide)
        self.slider.pack(side="left", fill="x", expand=True, padx=10)
        self.frame_lbl = tk.Label(ctrl, text="0/0", fg="#666")
        self.frame_lbl.pack(side="right")

    def _paths_text(self) -> str:
        return f"TEMP: {self.temp_dir}\nGIF : {self.gif_dir}"

    def _persist(self):
        cfg = {
            "temp_dir": self.temp_dir,
            "gif_dir": self.gif_dir,
            "delay_ms": int(self.speed_var.get() or "520"),
            "save_png": bool(self.png_var.get()),
            "size": self.size,
        }
        save_config(cfg)

    def pick_temp(self):
        p = filedialog.askdirectory(initialdir=self.temp_dir, title="Choisir le dossier TEMP")
        if p:
            self.temp_dir = p
            os.makedirs(self.temp_dir, exist_ok=True)
            self.path_lbl.config(text=self._paths_text())
            self._persist()

    def pick_gif(self):
        p = filedialog.askdirectory(initialdir=self.gif_dir, title="Choisir le dossier GIF")
        if p:
            self.gif_dir = p
            os.makedirs(self.gif_dir, exist_ok=True)
            self.path_lbl.config(text=self._paths_text())
            self._persist()

    def load_file(self):
        fn = filedialog.askopenfilename(title="Ouvrir un fichier PDN", filetypes=[("PDN", "*.pdn"), ("Tous fichiers", "*.*")])
        if not fn:
            return
        try:
            with open(fn, "r", encoding="utf-8", errors="replace") as f:
                data = f.read()
            self.txt.delete("1.0", "end")
            self.txt.insert("1.0", data)
        except Exception as e:
            messagebox.showerror("Erreur", str(e))

    def preview(self):
        try:
            text = self.txt.get("1.0", "end").strip()
            if not text:
                messagebox.showwarning("PDN", "Colle un PDN ou charge un fichier.")
                return

            tags = parse_tags(text)
            moves = extract_moves(text)
            if not moves:
                raise ValueError("Aucun coup détecté dans le PDN.")

            fen = extract_fen(text)
            if fen:
                side, pos = parse_fen(fen)
            else:
                side, pos = "w", initial_position()

            # Build frames:
            frames: List[Image.Image] = []
            # Frame 0 = position de départ (sans flèche)
            frames.append(render_frame(pos, None, size=self.size))

            # then each move frame with arrow
            current_side = side
            for mv in moves:
                frm_to = apply_move(pos, mv, current_side)
                frames.append(render_frame(pos, frm_to, size=self.size))
                current_side = "b" if current_side == "w" else "w"

            self.frames = frames
            self.cur = 0

            white = tags.get("white", "?")
            black = tags.get("black", "?")
            event = tags.get("event", "?")
            self.meta_lbl.config(text=f"{white} (Blancs) vs {black} (Noirs) — {event}")

            self.slider.config(from_=0, to=max(0, len(frames) - 1))
            self.slider.set(0)
            self._show_frame(0)

            self.status.config(text=f"OK — {len(frames)} frames (inclut la position de départ).")
            self._persist()

        except Exception as e:
            self.status.config(text=f"Erreur: {e}")
            self.frames = []
            self.slider.config(from_=0, to=0)
            self.slider.set(0)
            self.frame_lbl.config(text="0/0")

    def export_gif(self):
        if not self.frames:
            messagebox.showwarning("Export", "Aucune frame. Clique d'abord sur Prévisualiser.")
            return
        try:
            os.makedirs(self.temp_dir, exist_ok=True)
            os.makedirs(self.gif_dir, exist_ok=True)

            delay = int(self.speed_var.get() or "520")
            save_png = bool(self.png_var.get())

            if save_png:
                for i, im in enumerate(self.frames):
                    im.save(os.path.join(self.temp_dir, f"frame_{i:04d}.png"), "PNG")

            gif_path = next_gif_path(self.gif_dir)
            save_gif(self.frames, gif_path, delay_ms=delay)

            self.status.config(text=f"GIF OK: {gif_path}")
            messagebox.showinfo("Export", f"GIF généré:\n{gif_path}")
            self._persist()

        except Exception as e:
            messagebox.showerror("Erreur export", str(e))

    def on_slide(self, v):
        if not self.frames:
            return
        idx = int(float(v))
        self.cur = max(0, min(idx, len(self.frames) - 1))
        self._show_frame(self.cur)

    def _show_frame(self, idx: int):
        if not self.frames:
            return
        self._show_image(self.frames[idx])
        self.frame_lbl.config(text=f"{idx + 1}/{len(self.frames)}")

    def _show_image(self, im: Image.Image):
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        if cw < 10 or ch < 10:
            cw, ch = 800, 600

        img = im.copy()
        img.thumbnail((cw - 10, ch - 10), Image.LANCZOS)

        self.tkimg = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self.tkimg, anchor="center")


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
