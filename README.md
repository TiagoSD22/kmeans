# KMeans Clustering Visualizer

An interactive desktop application that implements K-Means clustering **from scratch** (no scikit-learn) and visualizes datapoints and clusters in real time using Tkinter and Matplotlib.

---

## Features

- **Hand-rolled K-Means** — pure NumPy implementation (random init, assign, update, convergence)
- **Two optimal-K detection methods** — Elbow (WCSS knee) and Silhouette (hand-rolled scoring)
- **Interactive scatter plot** — click directly on the chart to add points
- **Manual and random point addition**
- **Persistent dataset** — all points saved to `data.csv`
- **Live method chart** — Elbow curve or Silhouette bar chart rendered alongside the scatter plot

---

## Requirements

| Dependency | Version |
|---|---|
| Python | 3.9+ |
| numpy | any recent |
| matplotlib | any recent |

Tkinter is included with the standard Python distribution.

---

## Setup

### 1. Create and activate a virtual environment (pyenv recommended)

```bash
pyenv local 3.11.x          # or whichever version you have
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install numpy matplotlib
```

### 3. Run the application

```bash
python main.py
```

On first launch, `data.csv` is automatically created with **80 pre-generated points** spread across 3 natural clusters, so you can immediately try the clustering methods.

---

## Project Structure

```
kmeans/
├── main.py       # Entry point
├── data.py       # CSV dataset management
├── kmeans.py     # K-Means algorithm implementation
├── methods.py    # Elbow & Silhouette methods
├── ui.py         # Tkinter + Matplotlib UI
└── data.csv      # Auto-generated on first run
```

---

## How It Works

### `data.py` — Dataset Management

| Function | Description |
|---|---|
| `init_dataset(path, n, seed)` | Generates `n` 2-D points across 3 Gaussian clusters and saves to CSV. Called at startup only if the file doesn't exist. |
| `load_dataset(path)` | Reads `data.csv` → NumPy array of shape `(N, 2)`. |
| `append_point(x, y, path)` | Appends a single `(x, y)` row to the CSV. |
| `random_point(xlim, ylim)` | Returns a random `(x, y)` tuple within the given axis bounds. |
| `clear_dataset(path)` | Resets the CSV to header-only (empty dataset). |

CSV format — two columns with a header row:
```
x,y
2.1234,3.5678
...
```

---

### `kmeans.py` — K-Means Algorithm

Fully hand-rolled using NumPy only. No sklearn, no scipy.

```
KMeans(k, max_iter, tol, seed)
  .fit(X)       → stores .labels_, .centroids_, .inertia_
  .predict(X)   → cluster labels for new points
```

**Algorithm steps:**

1. **Initialisation** — randomly sample `k` distinct rows from `X` as starting centroids
2. **Assignment** — compute Euclidean distance from every point to every centroid:
   ```
   dist(a, b) = sqrt(sum((a - b)²))
   ```
   Each point is assigned to the nearest centroid.
3. **Update** — recompute each centroid as the mean of its assigned points.
   If a cluster becomes empty, its centroid is reinitialised to a random point.
4. **Convergence** — repeat steps 2–3 until the maximum centroid shift drops below `tol`
   or `max_iter` iterations are reached.
5. **Inertia** — after convergence, compute WCSS (Within-Cluster Sum of Squares):
   ```
   inertia = Σ Σ ||x - centroid_j||²
   ```

---

### `methods.py` — Optimal K Detection

#### Elbow Method

Fits K-Means for every `k` in the selected range and records the inertia (WCSS). The optimal K is the **knee** of the resulting curve — detected using the maximum absolute second difference of the inertia values:

```
second_diff = diff(inertias, n=2)
optimal_k   = ks[ argmax(|second_diff|) + 1 ]
```

The UI renders the full Elbow curve with a red dashed line at the chosen K.

#### Silhouette Method

For each `k`, fits K-Means then computes the **mean silhouette coefficient** — entirely by hand:

For each point `i`:
```
a(i) = mean distance to all other points in the same cluster
b(i) = min over other clusters of mean distance to points in that cluster
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```

The mean of all `s(i)` is the silhouette score for that `k`. The optimal K is `argmax` of scores.

The UI renders a bar chart; the best K bar is highlighted in red.

---

### `ui.py` — Application UI

**Layout:**

```
┌──────────────────────────────────────────────────────┐
│  KMeans Clustering Visualizer                        │
├─────────────────┬────────────────────────────────────┤
│  CONTROLS       │  SCATTER PLOT                      │
│                 │  (coloured by cluster, ✕=centroid) │
│  [Add Random]   │                                    │
│  [Run KMeans]   ├────────────────────────────────────┤
│  [Clear All]    │  METHOD CHART                      │
│                 │  (Elbow curve or Silhouette bars)  │
│  X: [____]      │                                    │
│  Y: [____]      │                                    │
│  [Add Point]    │                                    │
│                 │                                    │
│  ◉ Elbow        │                                    │
│  ○ Silhouette   │                                    │
│                 │                                    │
│  K min: [2]     │                                    │
│  K max: [10]    │                                    │
│                 │                                    │
│  Optimal K: 3   │                                    │
└─────────────────┴────────────────────────────────────┘
```

---

## Using the Application

### Adding Points

| Action | How |
|---|---|
| Click on the scatter plot | Adds a point at the clicked coordinate |
| "Add Random Point" button | Generates a point at random within the plot bounds |
| X / Y fields + "Add Point" | Adds a point at the exact coordinates you enter |

All added points are immediately saved to `data.csv`.

### Running K-Means

1. Select a method — **Elbow** or **Silhouette**
2. Set the **K range** (default: 2 – 10)
3. Click **Run KMeans**

The app will:
- Determine the optimal K using the chosen method
- Fit K-Means with that K
- Colour each point by its cluster on the scatter plot
- Mark centroids with a black ✕
- Draw the Elbow curve or Silhouette bar chart in the lower panel
- Display the optimal K in the control panel

### Clearing Data

Click **Clear All** to reset `data.csv` to empty and clear both charts.

---

## Notes

- The dataset persists between runs via `data.csv`. Delete the file to regenerate the default dataset on next launch.
- The Silhouette method is significantly slower than Elbow for large datasets (O(N²) pairwise distances), so keep the dataset to a few hundred points for snappy interaction.
- Setting K max close to the number of points will trigger a validation error — at minimum, you need `K max + 1` points in the dataset.
