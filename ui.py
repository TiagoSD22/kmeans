"""
KMeans Clustering Visualizer — Tkinter + Matplotlib UI
"""

import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from data import init_dataset, load_dataset, append_point, random_point, clear_dataset
from kmeans import KMeans
from methods import elbow, find_knee, silhouette, best_silhouette_k

CSV_PATH = "data.csv"
AXIS_XLIM = (0.0, 10.0)
AXIS_YLIM = (0.0, 10.0)


class KMeansApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("KMeans Clustering Visualizer")
        self.resizable(True, True)

        self._last_labels = None
        self._last_centroids = None

        self._build_ui()
        self._refresh_scatter()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # ---- left control panel ----
        ctrl = ttk.Frame(self, padding=10)
        ctrl.grid(row=0, column=0, sticky="ns")
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # Action buttons
        ttk.Button(ctrl, text="Add Random Point", command=self._add_random).grid(
            row=0, column=0, columnspan=2, sticky="ew", pady=2
        )
        ttk.Button(ctrl, text="Run KMeans", command=self._run_kmeans).grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=2
        )
        ttk.Button(ctrl, text="Clear All", command=self._clear_all).grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=2
        )

        ttk.Separator(ctrl, orient="horizontal").grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=8
        )

        # Manual point entry
        ttk.Label(ctrl, text="Manual point:").grid(row=4, column=0, columnspan=2, sticky="w")
        ttk.Label(ctrl, text="X:").grid(row=5, column=0, sticky="e")
        self._entry_x = ttk.Entry(ctrl, width=8)
        self._entry_x.grid(row=5, column=1, sticky="w", padx=4)
        ttk.Label(ctrl, text="Y:").grid(row=6, column=0, sticky="e")
        self._entry_y = ttk.Entry(ctrl, width=8)
        self._entry_y.grid(row=6, column=1, sticky="w", padx=4)
        ttk.Button(ctrl, text="Add Point", command=self._add_manual).grid(
            row=7, column=0, columnspan=2, sticky="ew", pady=4
        )

        ttk.Separator(ctrl, orient="horizontal").grid(
            row=8, column=0, columnspan=2, sticky="ew", pady=8
        )

        # Method selection
        ttk.Label(ctrl, text="K detection method:").grid(row=9, column=0, columnspan=2, sticky="w")
        self._method = tk.StringVar(value="elbow")
        ttk.Radiobutton(ctrl, text="Elbow", variable=self._method, value="elbow").grid(
            row=10, column=0, columnspan=2, sticky="w"
        )
        ttk.Radiobutton(ctrl, text="Silhouette", variable=self._method, value="silhouette").grid(
            row=11, column=0, columnspan=2, sticky="w"
        )

        ttk.Separator(ctrl, orient="horizontal").grid(
            row=12, column=0, columnspan=2, sticky="ew", pady=8
        )

        # K range
        ttk.Label(ctrl, text="K range:").grid(row=13, column=0, columnspan=2, sticky="w")
        ttk.Label(ctrl, text="Min:").grid(row=14, column=0, sticky="e")
        self._k_min = ttk.Entry(ctrl, width=5)
        self._k_min.insert(0, "2")
        self._k_min.grid(row=14, column=1, sticky="w", padx=4)
        ttk.Label(ctrl, text="Max:").grid(row=15, column=0, sticky="e")
        self._k_max = ttk.Entry(ctrl, width=5)
        self._k_max.insert(0, "10")
        self._k_max.grid(row=15, column=1, sticky="w", padx=4)

        ttk.Separator(ctrl, orient="horizontal").grid(
            row=16, column=0, columnspan=2, sticky="ew", pady=8
        )

        # Optimal K display
        ttk.Label(ctrl, text="Optimal K:").grid(row=17, column=0, sticky="e")
        self._k_label = ttk.Label(ctrl, text="—", font=("", 14, "bold"))
        self._k_label.grid(row=17, column=1, sticky="w", padx=4)

        # ---- right matplotlib canvas ----
        canvas_frame = ttk.Frame(self)
        canvas_frame.grid(row=0, column=1, sticky="nsew", padx=4, pady=4)
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        self._fig = Figure(figsize=(9, 6))
        self._ax_scatter = self._fig.add_subplot(211)
        self._ax_method = self._fig.add_subplot(212)
        self._fig.tight_layout(pad=3.0)

        self._canvas = FigureCanvasTkAgg(self._fig, master=canvas_frame)
        self._canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self._canvas.mpl_connect("button_press_event", self._on_plot_click)

    # ------------------------------------------------------------------
    # Scatter plot helpers
    # ------------------------------------------------------------------

    def _refresh_scatter(self, labels=None, centroids=None):
        X = load_dataset(CSV_PATH)
        ax = self._ax_scatter
        ax.clear()
        ax.set_xlim(AXIS_XLIM)
        ax.set_ylim(AXIS_YLIM)
        ax.set_title("Data Points & Clusters")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        if len(X) == 0:
            self._canvas.draw()
            return

        if labels is not None and centroids is not None:
            ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", s=40, alpha=0.8)
            ax.scatter(
                centroids[:, 0], centroids[:, 1],
                marker="X", s=180, c="black", zorder=5, label="Centroids"
            )
            ax.legend(loc="upper right", fontsize=8)
        else:
            ax.scatter(X[:, 0], X[:, 1], color="steelblue", s=40, alpha=0.8)

        self._canvas.draw()

    # ------------------------------------------------------------------
    # Method chart helpers
    # ------------------------------------------------------------------

    def _draw_elbow(self, ks, inertias, optimal_k):
        ax = self._ax_method
        ax.clear()
        ax.plot(ks, inertias, "o-", color="royalblue")
        ax.axvline(optimal_k, color="red", linestyle="--", label=f"K={optimal_k}")
        ax.set_title("Elbow Method (WCSS)")
        ax.set_xlabel("k")
        ax.set_ylabel("Inertia")
        ax.legend()
        self._canvas.draw()

    def _draw_silhouette(self, ks, scores, optimal_k):
        ax = self._ax_method
        ax.clear()
        colors = ["red" if k == optimal_k else "steelblue" for k in ks]
        ax.bar(ks, scores, color=colors)
        ax.set_title("Silhouette Scores")
        ax.set_xlabel("k")
        ax.set_ylabel("Score")
        ax.set_xticks(ks)
        # Red bar legend
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color="red", label=f"Best K={optimal_k}")])
        self._canvas.draw()

    # ------------------------------------------------------------------
    # Button / event handlers
    # ------------------------------------------------------------------

    def _invalidate_clustering(self):
        """Clear stale cluster results whenever the dataset changes."""
        self._last_labels = None
        self._last_centroids = None
        self._k_label.config(text="—")

    def _add_random(self):
        x, y = random_point(AXIS_XLIM, AXIS_YLIM)
        append_point(x, y, CSV_PATH)
        self._invalidate_clustering()
        self._refresh_scatter()

    def _add_manual(self):
        try:
            x = float(self._entry_x.get())
            y = float(self._entry_y.get())
        except ValueError:
            messagebox.showerror("Input error", "X and Y must be numeric values.")
            return
        append_point(x, y, CSV_PATH)
        self._entry_x.delete(0, tk.END)
        self._entry_y.delete(0, tk.END)
        self._invalidate_clustering()
        self._refresh_scatter()

    def _on_plot_click(self, event):
        if event.inaxes is not self._ax_scatter:
            return
        x, y = round(event.xdata, 4), round(event.ydata, 4)
        append_point(x, y, CSV_PATH)
        self._invalidate_clustering()
        self._refresh_scatter()

    def _run_kmeans(self):
        X = load_dataset(CSV_PATH)
        try:
            k_min = int(self._k_min.get())
            k_max = int(self._k_max.get())
        except ValueError:
            messagebox.showerror("Input error", "K min/max must be integers.")
            return

        if k_min < 2 or k_max < k_min:
            messagebox.showerror("Input error", "K min must be ≥ 2 and ≤ K max.")
            return

        min_needed = k_max + 1  # silhouette needs at least k+1 points
        if len(X) < min_needed:
            messagebox.showerror(
                "Not enough data",
                f"Need at least {min_needed} points for K up to {k_max}.\n"
                f"Currently have {len(X)} points."
            )
            return

        k_range = range(k_min, k_max + 1)
        method = self._method.get()

        if method == "elbow":
            ks, inertias = elbow(X, k_range)
            optimal_k = find_knee(ks, inertias)
            model = KMeans(k=optimal_k, seed=42).fit(X)
            self._last_labels = model.labels_
            self._last_centroids = model.centroids_
            self._refresh_scatter(model.labels_, model.centroids_)
            self._draw_elbow(ks, inertias, optimal_k)
        else:
            ks, scores = silhouette(X, k_range)
            optimal_k = best_silhouette_k(ks, scores)
            model = KMeans(k=optimal_k, seed=42).fit(X)
            self._last_labels = model.labels_
            self._last_centroids = model.centroids_
            self._refresh_scatter(model.labels_, model.centroids_)
            self._draw_silhouette(ks, scores, optimal_k)

        self._k_label.config(text=str(optimal_k))

    def _clear_all(self):
        clear_dataset(CSV_PATH)
        self._last_labels = None
        self._last_centroids = None
        self._k_label.config(text="—")
        self._ax_method.clear()
        self._refresh_scatter()
