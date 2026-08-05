"""
Główne okno aplikacji — Crypto Explorer Pro.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from datetime import datetime, timedelta
import threading

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.widgets import SpanSelector
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import pandas as pd

from .config import (
    C_BG, C_PANEL, C_PANEL2, C_GRID, C_BORDER,
    C_TEXT, C_TEXT2, C_TEXT3,
    C_UP, C_DOWN, C_MA7, C_MA25, C_BB, C_VOL,
    COINS, RANGES, MODES,
)
from .api import CoinGecko, APIError
from .indicators import add_indicators, compute_stats

FONT_MONO  = ("Courier New", 10)
FONT_MONO_B = ("Courier New", 10, "bold")
FONT_MONO_L = ("Courier New", 14, "bold")
FONT_MONO_XL = ("Courier New", 22, "bold")
FONT_UI    = ("Segoe UI", 9)
FONT_UI_B  = ("Segoe UI", 9, "bold")
FONT_UI_S  = ("Segoe UI", 8)


def _fmt_price(v: float) -> str:
    if abs(v) >= 10_000:
        return f"${v:,.0f}"
    if abs(v) >= 1:
        return f"${v:,.4f}"
    return f"${v:.6f}"

def _fmt_price_ax(v: float, _=None) -> str:
    if abs(v) >= 10_000:
        return f"{v:,.0f}"
    if abs(v) >= 1:
        return f"{v:.2f}"
    return f"{v:.5f}"

def _fmt_vol(v: float) -> str:
    if v >= 1e9:
        return f"${v/1e9:.2f}B"
    return f"${v/1e6:.1f}M"

def _style_ax(ax, df: pd.DataFrame, *, yright: bool = True):
    ax.set_facecolor(C_PANEL)
    ax.grid(True, color=C_GRID, lw=0.4, ls="--", alpha=0.7)
    ax.tick_params(colors=C_TEXT2, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(C_BORDER)

    if yright:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_price_ax))

    locator = mdates.AutoDateLocator(minticks=5, maxticks=12)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    ax.tick_params(axis="x", colors=C_TEXT2, labelsize=8, rotation=0)


class App:
    def __init__(self):
        self._client  = CoinGecko()
        self._df: pd.DataFrame | None = None
        self._mode    = "buy"
        self._loading = False

        self._last_sub: pd.DataFrame | None = None
        self._last_t0 = None
        self._last_t1 = None
        self._vlines: list = []

        self._build_root()
        self._build_sidebar()
        self._build_charts_area()
        self._build_opt_tab()
        self._build_statusbar()

        self._load_data()

    def _build_root(self):
        self.root = tk.Tk()
        self.root.title("Crypto Explorer Pro")
        self.root.configure(bg=C_BG)
        self.root.geometry("1400x820")
        self.root.minsize(1100, 700)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        hdr = tk.Frame(self.root, bg=C_PANEL2, height=56)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)

        self._lbl_coin = tk.Label(
            hdr, text="─", bg=C_PANEL2, fg=C_TEXT,
            font=FONT_MONO_XL, anchor="w", padx=20,
        )
        self._lbl_coin.pack(side=tk.LEFT)

        self._lbl_price = tk.Label(
            hdr, text="", bg=C_PANEL2, fg=C_TEXT,
            font=("Courier New", 18, "bold"), anchor="w",
        )
        self._lbl_price.pack(side=tk.LEFT, padx=(4, 0))

        self._lbl_change = tk.Label(
            hdr, text="", bg=C_PANEL2, fg=C_TEXT2,
            font=FONT_MONO_L, anchor="w", padx=10,
        )
        self._lbl_change.pack(side=tk.LEFT)

        mode_frame = tk.Frame(hdr, bg=C_PANEL2)
        mode_frame.pack(side=tk.RIGHT, padx=20)

        self._mode_btns: dict[str, tk.Button] = {}
        for key in ("buy", "sell"):
            cfg = MODES[key]
            btn = tk.Button(
                mode_frame,
                text=cfg["label"],
                font=("Courier New", 11, "bold"),
                relief="flat", bd=0, cursor="hand2",
                padx=18, pady=8,
                command=lambda k=key: self._set_mode(k),
            )
            btn.pack(side=tk.LEFT, padx=3)
            self._mode_btns[key] = btn

        self._refresh_mode_btns()
        tk.Frame(self.root, bg=C_BORDER, height=1).pack(fill=tk.X)

        style = ttk.Style()
        style.theme_use('default')
        style.configure('TNotebook', background=C_BG, borderwidth=0)
        style.configure('TNotebook.Tab', background=C_PANEL, foreground=C_TEXT2, padding=[15, 6], borderwidth=0, font=FONT_UI_B)
        style.map('TNotebook.Tab', background=[('selected', C_BORDER)], foreground=[('selected', C_TEXT)])

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self._main = tk.Frame(self.notebook, bg=C_BG)
        self._tab_opt = tk.Frame(self.notebook, bg=C_BG)

        self.notebook.add(self._main, text="  Analiza Rynku  ")
        self.notebook.add(self._tab_opt, text="  Optymalizacja Robustna  ")

    def _build_sidebar(self):
        sidebar = tk.Frame(self._main, bg=C_PANEL2, width=220)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 1))
        sidebar.pack_propagate(False)

        def section(label: str):
            tk.Frame(sidebar, bg=C_PANEL2, height=10).pack(fill=tk.X)
            tk.Label(
                sidebar, text=label.upper(),
                bg=C_PANEL2, fg=C_TEXT3,
                font=("Courier New", 8, "bold"),
                anchor="w", padx=14, pady=2,
            ).pack(fill=tk.X)
            tk.Frame(sidebar, bg=C_BORDER, height=1).pack(fill=tk.X, padx=14)

        section("Kryptowaluta")

        self._coin_var = tk.StringVar(value=list(COINS.keys())[0])
        coin_menu = tk.OptionMenu(
            sidebar, self._coin_var, *COINS.keys(),
            command=lambda _: self._on_coin_change(),
        )
        coin_menu.config(
            bg=C_PANEL, fg=C_TEXT, font=FONT_UI,
            activebackground=C_PANEL2, activeforeground=C_TEXT,
            relief="flat", bd=0, highlightthickness=0,
            indicatoron=True, width=22,
        )
        coin_menu["menu"].config(bg=C_PANEL, fg=C_TEXT, font=FONT_UI)
        coin_menu.pack(fill=tk.X, padx=10, pady=6)

        section("Zakres czasu")

        self._range_var = tk.StringVar(value=list(RANGES.keys())[0])
        for label in RANGES:
            rb = tk.Radiobutton(
                sidebar, text=f"  {label}",
                variable=self._range_var, value=label,
                bg=C_PANEL2, fg=C_TEXT2,
                selectcolor=C_BG,
                activebackground=C_PANEL2, activeforeground=C_TEXT,
                font=FONT_UI, anchor="w",
                command=self._on_range_change,
            )
            rb.pack(fill=tk.X, padx=14, pady=1)

        self._custom_rb = tk.Radiobutton(
            sidebar, text="  Niestandardowy",
            variable=self._range_var, value="custom",
            bg=C_PANEL2, fg=C_TEXT2,
            selectcolor=C_BG,
            activebackground=C_PANEL2, activeforeground=C_TEXT,
            font=FONT_UI, anchor="w",
            command=self._on_range_change,
        )
        self._custom_rb.pack(fill=tk.X, padx=14, pady=(1, 0))

        self._date_frame = tk.Frame(sidebar, bg=C_PANEL2)
        self._date_frame.pack(fill=tk.X, padx=14, pady=(4, 0))

        today = datetime.now()
        ago30 = today - timedelta(days=30)

        for attr, lbl, default in (
            ("_date_from", "Od:", ago30),
            ("_date_to",   "Do:", today),
        ):
            row = tk.Frame(self._date_frame, bg=C_PANEL2)
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=lbl, bg=C_PANEL2, fg=C_TEXT2,
                     font=FONT_UI_S, width=4, anchor="w").pack(side=tk.LEFT)
            var = tk.StringVar(value=default.strftime("%Y-%m-%d"))
            ent = tk.Entry(row, textvariable=var, font=FONT_UI_S,
                           bg=C_PANEL, fg=C_TEXT, insertbackground=C_TEXT,
                           relief="flat", bd=4, width=12)
            ent.pack(side=tk.LEFT)
            setattr(self, attr, var)

        tk.Button(
            self._date_frame, text="Załaduj zakres",
            font=FONT_UI_B,
            bg=C_BORDER, fg=C_TEXT,
            activebackground=C_PANEL, activeforeground="#ffffff",
            relief="flat", bd=0, cursor="hand2",
            pady=5,
            command=self._on_custom_range,
        ).pack(fill=tk.X, pady=(6, 0))

        self._toggle_date_frame()

        section("Statystyki")

        self._stats_vars: dict[str, tk.StringVar] = {}
        stats_labels = [
            ("Otwarcie",   "open"),
            ("Zamknięcie", "close"),
            ("Minimum",    "low"),
            ("Maksimum",   "high"),
            ("MA 7",       "ma7"),
            ("MA 25",      "ma25"),
            ("RSI (14)",   "rsi"),
            ("Zmienność",  "vol"),
            ("Śr. wolumen","avgvol"),
        ]
        for label, key in stats_labels:
            row = tk.Frame(sidebar, bg=C_PANEL2)
            row.pack(fill=tk.X, padx=14, pady=1)
            tk.Label(row, text=label, bg=C_PANEL2, fg=C_TEXT3,
                     font=FONT_UI_S, width=12, anchor="w").pack(side=tk.LEFT)
            var = tk.StringVar(value="─")
            self._stats_vars[key] = var
            tk.Label(row, textvariable=var, bg=C_PANEL2, fg=C_TEXT,
                     font=("Courier New", 8), anchor="e").pack(side=tk.RIGHT)

        self._lbl_status = tk.Label(
            sidebar, text="", bg=C_PANEL2, fg=C_TEXT2,
            font=FONT_UI_S, anchor="w", padx=14, wraplength=196,
        )
        self._lbl_status.pack(fill=tk.X, pady=(12, 4))

    def _build_charts_area(self):
        charts_frame = tk.Frame(self._main, bg=C_BG)
        charts_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig = plt.figure(figsize=(14, 8), dpi=96, facecolor=C_BG)
        self.fig.subplots_adjust(
            left=0.02, right=0.93, top=0.96, bottom=0.08,
            hspace=0.35,
        )

        from matplotlib.gridspec import GridSpec
        gs = GridSpec(
            4, 1, figure=self.fig,
            height_ratios=[45, 12, 13, 25],
            hspace=0.35,
        )
        self.ax_price  = self.fig.add_subplot(gs[0])
        self.ax_vol    = self.fig.add_subplot(gs[1], sharex=self.ax_price)
        self.ax_rsi    = self.fig.add_subplot(gs[2], sharex=self.ax_price)
        self.ax_zoom   = self.fig.add_subplot(gs[3])

        self.canvas = FigureCanvasTkAgg(self.fig, master=charts_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_opt_tab(self):
        opt_sidebar = tk.Frame(self._tab_opt, bg=C_PANEL2, width=280)
        opt_sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 1))
        opt_sidebar.pack_propagate(False)

        tk.Frame(opt_sidebar, bg=C_PANEL2, height=10).pack(fill=tk.X)
        tk.Label(opt_sidebar, text="PARAMETRY OGRANICZEŃ", bg=C_PANEL2, fg=C_TEXT3, font=("Courier New", 10, "bold")).pack(pady=10)
        tk.Frame(opt_sidebar, bg=C_BORDER, height=1).pack(fill=tk.X, padx=14)

        def add_slider(label, from_, to, default, resolution=0.1):
            tk.Label(opt_sidebar, text=label, bg=C_PANEL2, fg=C_TEXT2, font=FONT_UI).pack(anchor="w", padx=14, pady=(10, 0))
            var = tk.DoubleVar(value=default)
            scale = tk.Scale(
                opt_sidebar, from_=from_, to=to, resolution=resolution,
                orient=tk.HORIZONTAL, variable=var,
                bg=C_PANEL2, fg=C_TEXT, highlightthickness=0, bd=0,
                command=self._update_opt_plot
            )
            scale.pack(fill=tk.X, padx=14, pady=2)
            return var

        self.var_a1 = add_slider("Nominalne a1:", 0.5, 5.0, 2.0)
        self.var_da1 = add_slider("Niepewność a1 (odchylenie):", 0.0, 3.0, 1.0)
        self.var_a2 = add_slider("Nominalne a2:", 0.5, 5.0, 3.0)
        self.var_da2 = add_slider("Niepewność a2 (odchylenie):", 0.0, 3.0, 1.0)

        opt_main = tk.Frame(self._tab_opt, bg=C_BG)
        opt_main.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig_opt = plt.figure(figsize=(8, 6), dpi=96, facecolor=C_BG)
        self.ax_opt = self.fig_opt.add_subplot(111)
        self.canvas_opt = FigureCanvasTkAgg(self.fig_opt, master=opt_main)
        self.canvas_opt.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._update_opt_plot()

    def _update_opt_plot(self, *_):
        self.ax_opt.cla()
        self.ax_opt.set_facecolor(C_BG)
        for spine in self.ax_opt.spines.values():
            spine.set_edgecolor(C_BORDER)
        self.ax_opt.tick_params(colors=C_TEXT2, labelsize=9)
        self.ax_opt.grid(True, color=C_GRID, lw=0.4, ls="--", alpha=0.7)

        a1 = self.var_a1.get()
        da1 = self.var_da1.get()
        a2 = self.var_a2.get()
        da2 = self.var_da2.get()

        b = 30.0
        c1, c2 = 3.0, 2.0

        x1_nom = [0, b/a1]
        x2_nom = [b/a2, 0]

        a1_rob = a1 + da1
        a2_rob = a2 + da2
        x1_rob = [0, b/a1_rob]
        x2_rob = [b/a2_rob, 0]

        self.ax_opt.fill_between(x1_nom, x2_nom, 0, color="#60a5fa", alpha=0.15, label="Obszar nominalny")
        self.ax_opt.fill_between(x1_rob, x2_rob, 0, color="#34d399", alpha=0.35, label="Obszar robustny (bezpieczny)")

        self.ax_opt.plot(x1_nom, x2_nom, color="#60a5fa", lw=2, ls="--")
        self.ax_opt.plot(x1_rob, x2_rob, color="#34d399", lw=2)

        def get_opt(a, b_val, cx, cy):
            if cx/a[0] > cy/a[1]:
                return (b_val/a[0], 0), cx*(b_val/a[0])
            else:
                return (0, b_val/a[1]), cy*(b_val/a[1])

        pt_nom, z_nom = get_opt((a1, a2), b, c1, c2)
        pt_rob, z_rob = get_opt((a1_rob, a2_rob), b, c1, c2)

        self.ax_opt.plot(*pt_nom, 'o', color="#3b82f6", markersize=8, zorder=5)
        self.ax_opt.plot(*pt_rob, 'o', color="#10b981", markersize=8, zorder=5)

        self.ax_opt.annotate(f"Optimum Nominalne\nZ = {z_nom:.1f}",
                             xy=pt_nom, xytext=(pt_nom[0]+2, pt_nom[1]+5),
                             color="#60a5fa", fontfamily="Courier New", fontsize=9,
                             arrowprops=dict(arrowstyle="->", color="#60a5fa"))

        self.ax_opt.annotate(f"Optimum Robustne\nZ = {z_rob:.1f}",
                             xy=pt_rob, xytext=(pt_rob[0]+2, pt_rob[1]+5),
                             color="#34d399", fontfamily="Courier New", fontsize=9,
                             arrowprops=dict(arrowstyle="->", color="#34d399"))

        self.ax_opt.set_xlim(0, 60)
        self.ax_opt.set_ylim(0, 60)
        self.ax_opt.set_xlabel("x1 (Waga 1)", color=C_TEXT2)
        self.ax_opt.set_ylabel("x2 (Waga 2)", color=C_TEXT2)

        self.ax_opt.set_title(
            "Optymalizacja Robustna w 2D\n(Funkcja celu: Max Z = 3*x1 + 2*x2 | Limit: <= 30)",
            color=C_TEXT, pad=15, fontfamily="Segoe UI", fontsize=11, fontweight="bold"
        )

        self.ax_opt.legend(facecolor=C_PANEL2, edgecolor=C_BORDER, labelcolor=C_TEXT, loc="upper right")
        self.canvas_opt.draw_idle()

    def _build_statusbar(self):
        bar = tk.Frame(self.root, bg="#0a0a0a", height=40)
        bar.pack(fill=tk.X, side=tk.BOTTOM)
        bar.pack_propagate(False)
        tk.Frame(self.root, bg=C_BORDER, height=1).pack(fill=tk.X, side=tk.BOTTOM)

        self._sel_var = tk.StringVar(value="Przeciągnij myszą po wykresie ceny, aby zaznaczyć zakres.")
        tk.Label(
            bar, textvariable=self._sel_var,
            bg="#0a0a0a", fg=C_TEXT2,
            font=FONT_MONO, anchor="w", padx=16,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        tk.Button(
            bar, text="  Kopiuj  ",
            bg=C_BORDER, fg=C_TEXT2,
            activebackground=C_PANEL, activeforeground="#ffffff",
            font=FONT_MONO_B, relief="flat", bd=0,
            cursor="hand2", padx=12,
            command=self._copy_selection,
        ).pack(side=tk.RIGHT, padx=12, pady=6)

    def _load_data(self, custom_from: datetime | None = None, custom_to: datetime | None = None):
        if self._loading:
            return
        self._loading = True
        self._set_status("Pobieranie danych…", color=C_TEXT2)
        threading.Thread(
            target=self._fetch_thread,
            args=(custom_from, custom_to),
            daemon=True,
        ).start()

    def _fetch_thread(self, custom_from, custom_to):
        coin_label = self._coin_var.get()
        coin_id    = COINS[coin_label]
        try:
            if custom_from and custom_to:
                df = self._client.get_history_range(coin_id, custom_from, custom_to)
                range_label = (
                    f"{custom_from.strftime('%d.%m.%Y')} – {custom_to.strftime('%d.%m.%Y')}"
                )
            else:
                days = RANGES[self._range_var.get()]
                df   = self._client.get_history(coin_id, days)
                range_label = self._range_var.get()

            df = add_indicators(df)
            self.root.after(0, self._on_data_ready, df, coin_label, range_label)
        except APIError as e:
            self.root.after(0, self._on_fetch_error, str(e))
        finally:
            self._loading = False

    def _on_data_ready(self, df: pd.DataFrame, coin_label: str, range_label: str):
        self._df = df.copy()

        if pd.api.types.is_numeric_dtype(self._df["timestamp"]):
            if self._df["timestamp"].max() > 1e11:
                self._df["timestamp"] = pd.to_datetime(self._df["timestamp"], unit="ms")
            else:
                self._df["timestamp"] = pd.to_datetime(self._df["timestamp"], unit="s")
        else:
            self._df["timestamp"] = pd.to_datetime(self._df["timestamp"])

        self._coin_label   = coin_label
        self._range_label  = range_label
        self._last_sub     = None
        self._last_t0      = None
        self._last_t1      = None

        s = compute_stats(self._df)
        self._update_header(coin_label, s)
        self._update_stats_sidebar(s)
        self._redraw_all()
        self._set_status(f"Załadowano {len(self._df):,} punktów danych.", color=C_TEXT2)

    def _on_fetch_error(self, msg: str):
        self._set_status(f"✗ {msg}", color=C_DOWN)

    def _update_header(self, coin_label: str, s):
        self._lbl_coin.config(text=coin_label)
        self._lbl_price.config(text=_fmt_price(s.close))
        sign = "▲" if s.is_up else "▼"
        col  = C_UP if s.is_up else C_DOWN
        self._lbl_change.config(
            text=f"{sign} {abs(s.change_pct):.2f}%  ({'+' if s.is_up else ''}{_fmt_price(s.change_abs)})",
            fg=col,
        )

    def _update_stats_sidebar(self, s):
        self._stats_vars["open"].set(_fmt_price(s.open))
        self._stats_vars["close"].set(_fmt_price(s.close))
        self._stats_vars["low"].set(_fmt_price(s.low))
        self._stats_vars["high"].set(_fmt_price(s.high))
        self._stats_vars["ma7"].set(_fmt_price(s.ma7))
        self._stats_vars["ma25"].set(_fmt_price(s.ma25))

        rsi_val = s.rsi
        rsi_str = f"{rsi_val:.1f}"
        if rsi_val >= 70:
            rsi_str += " (wykup.)"
        elif rsi_val <= 30:
            rsi_str += " (wyprz.)"
        self._stats_vars["rsi"].set(rsi_str)
        self._stats_vars["vol"].set(f"{s.volatility:.2f}%")
        self._stats_vars["avgvol"].set(_fmt_vol(s.avg_volume))

    def _set_status(self, msg: str, color: str = C_TEXT2):
        self._lbl_status.config(text=msg, fg=color)

    def _refresh_mode_btns(self):
        for key, btn in self._mode_btns.items():
            cfg = MODES[key]
            if key == self._mode:
                btn.config(
                    bg=cfg["color"], fg="#0d0d0d",
                    activebackground=cfg["color"], activeforeground="#0d0d0d",
                )
            else:
                btn.config(
                    bg=C_BORDER, fg=C_TEXT3,
                    activebackground=C_PANEL2, activeforeground=C_TEXT2,
                )

    def _redraw_all(self):
        if self._df is None:
            return
        df   = self._df
        mode = MODES[self._mode]
        line_c = mode["color"]

        ax = self.ax_price
        ax.cla()
        ax.set_facecolor(C_PANEL)
        for spine in ax.spines.values():
            spine.set_edgecolor(C_BORDER)

        ax.fill_between(df["timestamp"], df["price"], alpha=0.08, color=line_c, zorder=1)
        ax.plot(df["timestamp"], df["price"],   color=line_c, lw=1.6, zorder=4, label="Cena")
        ax.plot(df["timestamp"], df["ma7"],     color=C_MA7,  lw=1.2, ls="--", zorder=3, label="MA 7")
        ax.plot(df["timestamp"], df["ma25"],    color=C_MA25, lw=1.2, ls="--", zorder=3, label="MA 25")
        ax.fill_between(df["timestamp"], df["bb_upper"], df["bb_lower"],
                        color=C_BB, alpha=0.06, zorder=2)
        ax.plot(df["timestamp"], df["bb_upper"], color=C_BB, lw=0.6, alpha=0.5, zorder=2)
        ax.plot(df["timestamp"], df["bb_lower"], color=C_BB, lw=0.6, alpha=0.5, zorder=2)

        _style_ax(ax, df)
        ax.set_title(
            f"{self._coin_label} / USD  —  {self._range_label}",
            color=C_TEXT, fontsize=10, loc="left", pad=6,
        )

        leg = ax.legend(
            fontsize=8, loc="upper left",
            facecolor=C_PANEL2, edgecolor=C_BORDER,
            labelcolor=C_TEXT, framealpha=0.9,
            handlelength=1.6, borderpad=0.6,
        )
        for line in leg.get_lines():
            line.set_linewidth(1.4)

        def on_select(xmin: float, xmax: float):
            if xmax - xmin < 1e-9:
                return
            t0 = mdates.num2date(xmin).replace(tzinfo=None)
            t1 = mdates.num2date(xmax).replace(tzinfo=None)
            sub = self._df[
                (self._df["timestamp"] >= t0) &
                (self._df["timestamp"] <= t1)
            ]
            if len(sub) < 2:
                return
            self._last_sub = sub
            self._last_t0  = t0
            self._last_t1  = t1
            self._draw_vlines(t0, t1)
            self._update_zoom(sub)
            self._update_sel_stats(sub, t0, t1)

        self.span = SpanSelector(
            ax, on_select,
            direction="horizontal",
            useblit=False,
            props=dict(facecolor=line_c, alpha=0.12, edgecolor="none"),
            interactive=True,
            drag_from_anywhere=True,
            button=[1],
        )

        ax = self.ax_vol
        ax.cla()
        ax.set_facecolor(C_PANEL)
        for spine in ax.spines.values():
            spine.set_edgecolor(C_BORDER)

        vol_colors = [C_UP if c >= o else C_DOWN
                      for o, c in zip(df["price"], df["price"].shift(-1).fillna(df["price"]))]

        bar_width = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds() / (len(df) * 86400) * 0.9

        ax.bar(df["timestamp"], df["volume"], color=vol_colors, alpha=0.55, width=bar_width)
        _style_ax(ax, df)
        ax.yaxis.tick_right()
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: _fmt_vol(v)))
        ax.text(0.005, 0.88, "WOLUMEN", transform=ax.transAxes,
                color=C_TEXT3, fontsize=7, fontfamily="Courier New")

        ax = self.ax_rsi
        ax.cla()
        ax.set_facecolor(C_PANEL)
        for spine in ax.spines.values():
            spine.set_edgecolor(C_BORDER)

        rsi = df["rsi"].copy()
        ax.plot(df["timestamp"], rsi, color="#c084fc", lw=1.2, zorder=3)
        ax.axhline(70, color=C_DOWN, lw=0.7, ls="--", alpha=0.6)
        ax.axhline(30, color=C_UP,   lw=0.7, ls="--", alpha=0.6)
        ax.fill_between(df["timestamp"], rsi, 70,
                        where=(rsi >= 70), color=C_DOWN, alpha=0.15)
        ax.fill_between(df["timestamp"], rsi, 30,
                        where=(rsi <= 30), color=C_UP, alpha=0.15)
        _style_ax(ax, df)
        ax.set_ylim(0, 100)
        ax.yaxis.tick_right()
        ax.yaxis.set_major_locator(mticker.FixedLocator([30, 70]))
        ax.text(0.005, 0.78, "RSI (14)", transform=ax.transAxes,
                color=C_TEXT3, fontsize=7, fontfamily="Courier New")

        ax.text(0.002, 0.28, "30", transform=ax.transAxes,
                color=C_UP, fontsize=7, fontfamily="Courier New", va="center")
        ax.text(0.002, 0.72, "70", transform=ax.transAxes,
                color=C_DOWN, fontsize=7, fontfamily="Courier New", va="center")

        self.ax_zoom.cla()
        self.ax_zoom.set_facecolor(C_PANEL)
        for spine in self.ax_zoom.spines.values():
            spine.set_edgecolor(C_BORDER)

        n   = len(df)
        cut = int(n * 0.75)
        sub = df.iloc[cut:]
        self._last_sub = sub
        self._last_t0  = df["timestamp"].iloc[cut]
        self._last_t1  = df["timestamp"].iloc[-1]
        self._update_zoom(sub)
        self._update_sel_stats(sub, self._last_t0, self._last_t1)

        x0 = mdates.date2num(self._last_t0)
        x1 = mdates.date2num(self._last_t1)
        self.span.extents = (x0, x1)
        self._draw_vlines(self._last_t0, self._last_t1)

        self.ax_price.set_xlim(df["timestamp"].iloc[0], df["timestamp"].iloc[-1])
        self.canvas.draw_idle()

    def _draw_vlines(self, t0, t1):
        for ln in self._vlines:
            try:
                ln.remove()
            except Exception:
                pass
        c = MODES[self._mode]["color"]

        self._vlines = [
            self.ax_price.axvline(mdates.date2num(t0), color=c, lw=1.2, ls=":", alpha=0.7, zorder=5),
            self.ax_price.axvline(mdates.date2num(t1), color=c, lw=1.2, ls=":", alpha=0.7, zorder=5),
        ]

        self.canvas.draw_idle()

    def _update_zoom(self, sub: pd.DataFrame):
        ax   = self.ax_zoom
        mode = MODES[self._mode]
        line_c = mode["color"]
        ax.cla()
        ax.set_facecolor(C_PANEL)
        for spine in ax.spines.values():
            spine.set_edgecolor(C_BORDER)

        if len(sub) < 2:
            ax.text(0.5, 0.5, "Zaznacz fragment powyżej",
                    ha="center", va="center", color=C_TEXT3,
                    transform=ax.transAxes, fontsize=9, fontfamily="Courier New")
            self.canvas.draw_idle()
            return

        ax.fill_between(sub["timestamp"], sub["price"], alpha=0.1, color=line_c)
        ax.plot(sub["timestamp"], sub["price"], color=line_c, lw=1.8, zorder=4)
        ax.plot(sub["timestamp"], sub["ma7"],   color=C_MA7,  lw=1.0, ls="--", alpha=0.8)
        ax.plot(sub["timestamp"], sub["ma25"],  color=C_MA25, lw=1.0, ls="--", alpha=0.8)

        pmin = float(sub["price"].min())
        pmax = float(sub["price"].max())
        pad  = max((pmax - pmin) * 0.06, abs(pmin) * 0.003 + 1e-9)
        ax.set_ylim(pmin - pad, pmax + pad)

        _style_ax(ax, sub)

        mode_label = mode["label"]
        ax.set_title(
            f"Zaznaczony zakres — {len(sub)} punktów  [{mode_label}]",
            color=line_c, fontsize=8, loc="left", pad=4,
        )
        self.canvas.draw_idle()

    def _update_sel_stats(self, sub: pd.DataFrame, t0=None, t1=None):
        if len(sub) < 2:
            self._sel_var.set("─")
            self._sel_raw = ""
            return
        p   = sub["price"]
        mn  = float(p.min())
        mx  = float(p.max())
        avg = float(p.mean())
        chg = (float(p.iloc[-1]) - float(p.iloc[0])) / float(p.iloc[0]) * 100

        rng_str = ""
        if t0 and t1:
            fmt = "%d.%m.%Y %H:%M"
            rng_str = f"  {t0.strftime(fmt)} → {t1.strftime(fmt)}"

        mode_label = MODES[self._mode]["label"]
        line = (
            f"min {_fmt_price(mn)}    max {_fmt_price(mx)}    "
            f"śr {_fmt_price(avg)}    zmiana {chg:+.2f}%"
            f"{rng_str}  [{mode_label}]"
        )
        self._sel_var.set(line)
        raw_vals = [round(v, 4) for v in p.tolist()]
        self._sel_raw = f"{line}\n{raw_vals}"

    def _copy_selection(self):
        raw = getattr(self, "_sel_raw", "")
        if not raw:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(raw)
        old = self._sel_var.get()
        self._sel_var.set("✓ Skopiowano do schowka.")
        self.root.after(1600, lambda: self._sel_var.set(old))

    def _set_mode(self, key: str):
        if key == self._mode:
            return
        self._mode = key
        self._refresh_mode_btns()
        if self._df is not None:
            self._redraw_all()

    def _on_coin_change(self):
        self._load_data()

    def _on_range_change(self):
        self._toggle_date_frame()
        if self._range_var.get() != "custom":
            self._load_data()

    def _on_custom_range(self):
        try:
            df_  = datetime.strptime(self._date_from.get().strip(), "%Y-%m-%d")
            dt_  = datetime.strptime(self._date_to.get().strip(),   "%Y-%m-%d")
            dt_  = dt_.replace(hour=23, minute=59, second=59)
        except ValueError:
            self._set_status("✗ Nieprawidłowy format daty (RRRR-MM-DD).", color=C_DOWN)
            return
        if df_ >= dt_:
            self._set_status("✗ Data 'Od' musi być wcześniejsza niż 'Do'.", color=C_DOWN)
            return
        self._load_data(custom_from=df_, custom_to=dt_)

    def _toggle_date_frame(self):
        if self._range_var.get() == "custom":
            self._date_frame.pack(fill=tk.X, padx=14, pady=(4, 0))
        else:
            self._date_frame.pack_forget()

    def _on_close(self):
        plt.close("all")
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def show_charts(df: pd.DataFrame, coin_label: str, range_label: str) -> None:
    app = App.__new__(App)
    app._client   = None
    app._df       = add_indicators(df)

    if pd.api.types.is_numeric_dtype(app._df["timestamp"]):
        if app._df["timestamp"].max() > 1e11:
            app._df["timestamp"] = pd.to_datetime(app._df["timestamp"], unit="ms")
        else:
            app._df["timestamp"] = pd.to_datetime(app._df["timestamp"], unit="s")
    else:
        app._df["timestamp"] = pd.to_datetime(app._df["timestamp"])

    app._mode     = "buy"
    app._loading  = False
    app._last_sub = None
    app._last_t0  = None
    app._last_t1  = None
    app._vlines   = []
    app._coin_label  = coin_label
    app._range_label = range_label

    app._build_root()
    app._build_sidebar()
    app._build_charts_area()
    app._build_opt_tab()
    app._build_statusbar()

    s = compute_stats(app._df)
    app._update_header(coin_label, s)
    app._update_stats_sidebar(s)
    app._redraw_all()
    app.root.mainloop()