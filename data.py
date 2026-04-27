import csv
import random
from pathlib import Path

import numpy as np


def init_dataset(path: str = "data.csv", n: int = 80, seed: int = 42) -> None:
    """Generate n random 2-D points and save to CSV. No-op if file already exists."""
    p = Path(path)
    if p.exists():
        return
    rng = random.Random(seed)
    # Create 3 rough clusters so the dataset is interesting out of the box
    clusters = [
        (2.0, 2.0), (7.0, 7.0), (2.0, 8.0)
    ]
    with p.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y"])
        for i in range(n):
            cx, cy = clusters[i % len(clusters)]
            x = round(cx + rng.gauss(0, 1.2), 4)
            y = round(cy + rng.gauss(0, 1.2), 4)
            writer.writerow([x, y])


def load_dataset(path: str = "data.csv") -> np.ndarray:
    """Read CSV and return array of shape (N, 2). Returns empty (0,2) array if file missing."""
    p = Path(path)
    if not p.exists():
        return np.empty((0, 2))
    points = []
    with p.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            points.append([float(row["x"]), float(row["y"])])
    return np.array(points) if points else np.empty((0, 2))


def append_point(x: float, y: float, path: str = "data.csv") -> None:
    """Append a single (x, y) point to the CSV."""
    p = Path(path)
    write_header = not p.exists()
    with p.open("a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["x", "y"])
        writer.writerow([round(x, 4), round(y, 4)])


def random_point(
    xlim: tuple = (0.0, 10.0), ylim: tuple = (0.0, 10.0)
) -> tuple:
    """Return a random (x, y) tuple within the given axis limits."""
    x = random.uniform(xlim[0], xlim[1])
    y = random.uniform(ylim[0], ylim[1])
    return round(x, 4), round(y, 4)


def clear_dataset(path: str = "data.csv") -> None:
    """Reset CSV to header-only (empty dataset)."""
    with Path(path).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y"])
