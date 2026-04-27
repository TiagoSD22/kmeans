from data import init_dataset
from ui import KMeansApp

CSV_PATH = "data.csv"


def main():
    init_dataset(path=CSV_PATH, n=80, seed=42)
    app = KMeansApp()
    app.mainloop()


if __name__ == "__main__":
    main()
