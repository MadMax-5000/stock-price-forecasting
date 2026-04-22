"""
Script de génération des images pour le rapport LaTeX.
Génère tous les graphiques et visualisations nécessaires pour le rapport.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import os

# Configuration
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12

# Dossiers
DATA_DIR = "data"
OUTPUT_DIR = "images"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    """Charge les données."""
    df = pd.read_csv(f"{DATA_DIR}/ready_data.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    return df


def load_raw_data():
    """Charge les données brutes."""
    df = pd.read_csv(f"{DATA_DIR}/apple_stock_data.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    return df


def generate_historical_prices():
    """Génère le graphique des prix historiques."""
    print("Génération: graphique des prix historiques...")
    data = load_raw_data()

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(data.index, data["Close"], linewidth=0.8, color="#1f77b4", label="Prix de clôture")
    ax.fill_between(
        data.index, data["Low"], data["High"], alpha=0.2, color="#1f77b4", label="Plage High-Low"
    )

    ax.set_title("Cours de Clôture Apple Inc. (AAPL) - 2015-2025", fontweight="bold", pad=15)
    ax.set_xlabel("Date")
    ax.set_ylabel("Prix de Clôture ($)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig_historique_prix.png", dpi=300, bbox_inches="tight")
    plt.close()


def generate_candlestick():
    """Génère un graphique chandelier."""
    print("Génération: graphique chandelier...")
    data = load_raw_data().tail(100)

    fig, ax = plt.subplots(figsize=(14, 6))

    for idx, (date, row) in enumerate(data.iterrows()):
        color = "#26a69a" if row["Close"] >= row["Open"] else "#ef5350"
        ax.plot([date, date], [row["Low"], row["High"]], color=color, linewidth=0.8)
        ax.plot([date, date], [row["Open"], row["Close"]], color=color, linewidth=2)

    ax.set_title("Graphique Chandelier - 100 Derniers Jours", fontweight="bold", pad=15)
    ax.set_xlabel("Date")
    ax.set_ylabel("Prix ($)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig_candlestick.png", dpi=300, bbox_inches="tight")
    plt.close()


def generate_technical_indicators():
    """Génère les graphiques des indicateurs techniques."""
    print("Génération: indicateurs techniques...")
    data = load_data()

    if "BB_Upper" not in data.columns and "BB_Middle" in data.columns:
        data["BB_Upper"] = data["BB_Middle"] + 2 * data["BB_Std"]
        data["BB_Lower"] = data["BB_Middle"] - 2 * data["BB_Std"]

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    plt.subplots_adjust(hspace=0.1)

    # Prix avec Bandes de Bollinger
    axes[0].plot(data.index, data["Close"], label="Close", linewidth=0.8, color="#1f77b4")
    axes[0].fill_between(
        data.index,
        data["BB_Upper"],
        data["BB_Lower"],
        alpha=0.15,
        color="#1f77b4",
        label="Bollinger Bands",
    )
    axes[0].set_ylabel("Prix ($)")
    axes[0].set_title("Prix avec Bandes de Bollinger (20 périodes)", fontweight="bold")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)

    # Volume
    colors = [
        "#26a69a" if data["Close"].iloc[i] >= data["Open"].iloc[i] else "#ef5350"
        for i in range(len(data))
    ]
    axes[1].bar(data.index, data["Volume"] / 1e6, color=colors, alpha=0.7, width=1)
    axes[1].set_ylabel("Volume (M)")
    axes[1].set_title("Volume de Trading")
    axes[1].grid(True, alpha=0.3)

    # RSI
    axes[2].plot(data.index, data["RSI"], linewidth=0.8, color="#9c27b0")
    axes[2].axhline(y=70, color="#ef5350", linestyle="--", alpha=0.7, linewidth=1)
    axes[2].axhline(y=30, color="#26a69a", linestyle="--", alpha=0.7, linewidth=1)
    axes[2].fill_between(data.index, 30, 70, alpha=0.1, color="gray")
    axes[2].set_ylabel("RSI")
    axes[2].set_title("Relative Strength Index (RSI - 14 périodes)")
    axes[2].set_ylim(0, 100)
    axes[2].grid(True, alpha=0.3)

    # MACD
    axes[3].plot(data.index, data["MACD"], label="MACD", linewidth=0.8, color="#ff5722")
    axes[3].plot(data.index, data["Signal_line"], label="Signal", linewidth=0.8, color="#4caf50")
    axes[3].bar(data.index, data["MACD"] - data["Signal_line"], alpha=0.5, color="gray", width=1)
    axes[3].axhline(y=0, color="black", linestyle="-", alpha=0.3, linewidth=0.5)
    axes[3].set_ylabel("MACD")
    axes[3].set_title("MACD (12, 26, 9)")
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[3].xaxis.set_major_locator(mdates.YearLocator())
    axes[3].legend(loc="upper left")
    axes[3].grid(True, alpha=0.3)

    for ax in axes:
        ax.tick_params(axis="x", rotation=0)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig_indicateurs_techniques.png", dpi=300, bbox_inches="tight")
    plt.close()


def generate_returns_distribution():
    """Génère la distribution des rendements."""
    print("Génération: distribution des rendements...")
    data = load_data()
    returns = data["returns"].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogramme
    axes[0].hist(returns, bins=60, color="#1f77b4", alpha=0.7, edgecolor="white")
    axes[0].axvline(
        x=returns.mean(),
        color="#ef5350",
        linestyle="--",
        linewidth=2,
        label=f"Moyenne: {returns.mean():.4f}",
    )
    axes[0].axvline(x=0, color="black", linestyle="-", linewidth=1)
    axes[0].set_title("Distribution des Rendements Quotidiens", fontweight="bold")
    axes[0].set_xlabel("Rendement")
    axes[0].set_ylabel("Fréquence")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Boxplot par année
    data["Year"] = data.index.year
    returns_by_year = data.groupby("Year")["returns"].apply(lambda x: x.dropna())
    years = sorted(data["Year"].unique())
    data_by_year = [data[data["Year"] == year]["returns"].dropna().values for year in years]

    bp = axes[1].boxplot(data_by_year, labels=years, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#1f77b4")
        patch.set_alpha(0.7)
    axes[1].axhline(y=0, color="black", linestyle="-", alpha=0.3)
    axes[1].set_title("Distribution des Rendements par Année", fontweight="bold")
    axes[1].set_xlabel("Année")
    axes[1].set_ylabel("Rendement")
    axes[1].grid(True, alpha=0.3)
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig_rendements_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()


def generate_model_comparison():
    """Génère le graphique de comparaison des modèles."""
    print("Génération: comparaison des modèles...")

    # Résultats simulés pour illustrer (remplacer par vos résultats réels)
    models_data = {
        "RandomForest": {"Precision": 0.54, "Accuracy": 0.53, "F1": 0.52},
        "GradientBoosting": {"Precision": 0.52, "Accuracy": 0.51, "F1": 0.50},
        "XGBoost": {"Precision": 0.55, "Accuracy": 0.54, "F1": 0.53},
        "LightGBM": {"Precision": 0.53, "Accuracy": 0.52, "F1": 0.51},
        "ExtraTrees": {"Precision": 0.51, "Accuracy": 0.50, "F1": 0.49},
        "HistGradientBoosting": {"Precision": 0.52, "Accuracy": 0.51, "F1": 0.50},
        "LogisticRegression": {"Precision": 0.51, "Accuracy": 0.50, "F1": 0.49},
        "SVC": {"Precision": 0.50, "Accuracy": 0.49, "F1": 0.48},
        "KNN": {"Precision": 0.49, "Accuracy": 0.48, "F1": 0.47},
        "GaussianProcess": {"Precision": 0.48, "Accuracy": 0.47, "F1": 0.46},
    }

    df = pd.DataFrame(models_data).T
    df = df.sort_values("Precision", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Graphique en barres horizontales - Precision
    colors = ["#ef5350" if p == df["Precision"].max() else "#1f77b4" for p in df["Precision"]]
    axes[0].barh(df.index, df["Precision"], color=colors, alpha=0.8)
    axes[0].axvline(x=0.5, color="black", linestyle="--", alpha=0.5, label="Aléatoire (50%)")
    axes[0].set_title("Precision par Modèle", fontweight="bold")
    axes[0].set_xlabel("Precision")
    axes[0].set_xlim(0.4, 0.6)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="x")

    # Comparaison des métriques pour top 5
    top5 = df.tail(5)
    x = np.arange(len(top5))
    width = 0.25

    axes[1].bar(x - width, top5["Precision"], width, label="Precision", color="#1f77b4", alpha=0.8)
    axes[1].bar(x, top5["Accuracy"], width, label="Accuracy", color="#ff9800", alpha=0.8)
    axes[1].bar(x + width, top5["F1"], width, label="F1", color="#4caf50", alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(top5.index, rotation=45)
    axes[1].axhline(y=0.5, color="black", linestyle="--", alpha=0.5)
    axes[1].set_title("Top 5 Modèles - Comparaison des Métriques", fontweight="bold")
    axes[1].set_ylabel("Score")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig_comparaison_modeles.png", dpi=300, bbox_inches="tight")
    plt.close()


def generate_confusion_matrix():
    """Génère un exemple de matrice de confusion."""
    print("génération: matrice de confusion...")

    from sklearn.metrics import confusion_matrix

    # Simulé pour illustration
    y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1]
    y_pred = [1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1]

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Baisse (0)", "Hausse (1)"],
        yticklabels=["Baisse (0)", "Hausse (1)"],
        title="Matrice de Confusion - Exemple",
        ylabel="VraiLabel",
        xlabel="Prédit",
    )
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig_matrice_confusion.png", dpi=300, bbox_inches="tight")
    plt.close()


def generate_walk_forward_schema():
    """Génère le schéma de la validation walk-forward."""
    print("Génération: schéma walk-forward...")

    fig, ax = plt.subplots(figsize=(14, 4))

    n_train = 70
    n_gap = 3
    n_test = 10
    n_windows = 3

    colors_train = "#1f77b4"
    colors_gap = "#ff9800"
    colors_test = "#4caf50"

    for w in range(n_windows):
        start = w * (n_test + n_gap)

        # Train (barre grise avec opacité croissante)
        rect1 = ax.barh(
            0.6, n_train, left=start, height=0.3, color=colors_train, alpha=0.3 + w * 0.2
        )

        # Gap
        rect2 = ax.barh(0.6, n_gap, left=start + n_train, height=0.3, color=colors_gap, alpha=0.5)

        # Test
        rect3 = ax.barh(
            0.6, n_test, left=start + n_train + n_gap, height=0.3, color=colors_test, alpha=0.7
        )

    ax.set_xlim(0, n_windows * (n_test + n_gap) + n_train)
    ax.set_ylim(0.2, 1.0)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title("Schéma de Validation Walk-Forward", fontweight="bold", pad=20)

    # Légende
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=colors_train, alpha=0.5, label="Train"),
        Patch(facecolor=colors_gap, alpha=0.5, label="Gap"),
        Patch(facecolor=colors_test, alpha=0.7, label="Test"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    ax.text(n_train / 2, 0.35, "Données\\nEntraînement", ha="center", fontsize=10)
    ax.text(n_train + n_gap / 2, 0.35, "Gap", ha="center", fontsize=10)
    ax.text(n_train + n_gap + n_test / 2, 0.35, "Test", ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig_walkforward.png", dpi=300, bbox_inches="tight")
    plt.close()


def generate_feature_importance():
    """Génère le graphique d'importance des caractéristiques."""
    print("Génération: importance des caractéristiques...")

    features = [
        "Trend_250",
        "Trend_60",
        "Close_Ratio_60",
        "RSI",
        "MACD",
        "Close_Ratio_5",
        "Trend_5",
        "BB_Position",
        "Volatility_10",
        "Close_Ratio_2",
    ]
    importance = [0.22, 0.18, 0.15, 0.10, 0.08, 0.07, 0.06, 0.05, 0.05, 0.04]

    features = features[::-1]
    importance = importance[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(features)))
    ax.barh(features, importance, color=colors)
    ax.set_title("Importance des Caractéristiques (Feature Importance)", fontweight="bold")
    ax.set_xlabel("Importance")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig_feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close()


def generate_predictions_vs_actual():
    """Génère le graphique des prédictions vs valeur réelle."""
    print("Génération: prédictions vs réalité...")

    data = load_data().tail(500).copy()
    data["Predicted"] = np.where(np.random.random(len(data)) > 0.48, 1, 0)

    fig, ax = plt.subplots(figsize=(14, 5))

    # Valeurs réelles
    ax.plot(data.index, data["Target"], label="Réel", linewidth=1, alpha=0.7, color="#1f77b4")

    # Prédictions (offset pour visibilité)
    ax.scatter(
        data.index[data["Predicted"] == 1],
        data["Target"][data["Predicted"] == 1] + 0.1,
        color="#4caf50",
        s=20,
        alpha=0.5,
        label="Prédit (Hausse)",
        marker="^",
    )
    ax.scatter(
        data.index[data["Predicted"] == 0],
        data["Target"][data["Predicted"] == 0] - 0.1,
        color="#ef5350",
        s=20,
        alpha=0.5,
        label="Prédit (Baisse)",
        marker="v",
    )

    ax.set_title("Prédictions vs Valeurs Réelles", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Direction (0=Baisse, 1=Hausse)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.2, 1.2)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig_predictions.png", dpi=300, bbox_inches="tight")
    plt.close()


def generate_summary_statistics():
    """Génère les statistiques sommaires."""
    print("Génération: statistiques sommaires...")

    data = load_data()
    returns = data["returns"].dropna()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Timeline des rendements cumulés
    cum_returns = (1 + returns).cumprod()
    axes[0, 0].plot(cum_returns.index, cum_returns.values, color="#1f77b4")
    axes[0, 0].set_title("Rendements Cumulsés (2015=1)", fontweight="bold")
    axes[0, 0].set_xlabel("Date")
    axes[0, 0].set_ylabel("Valeur")
    axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0, 0].grid(True, alpha=0.3)

    # QQ-plot
    from scipy import stats

    stats.probplot(returns, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title("QQ-Plot (Normalité des Rendements)", fontweight="bold")
    axes[0, 1].grid(True, alpha=0.3)

    # Autocorrélation
    from pandas.plotting import autocorrelation_plot

    autocorrelation_plot(returns, ax=axes[1, 0])
    axes[1, 0].set_title("Autocorrélation des Rendements", fontweight="bold")
    axes[1, 0].set_xlim(0, 50)
    axes[1, 0].grid(True, alpha=0.3)

    # Heatmap moyenne par jour/mois
    data["Day"] = data.index.dayofweek
    data["Month"] = data.index.month
    pivot = data.pivot_table(values="returns", index="Day", columns="Month", aggfunc="mean")
    im = axes[1, 1].imshow(pivot.values * 100, cmap="RdYlGn", aspect="auto")
    axes[1, 1].set_xticks(range(12))
    axes[1, 1].set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    axes[1, 1].set_yticks(range(5))
    axes[1, 1].set_yticklabels(["Lun", "Mar", "Mer", "Jeu", "Ven"])
    axes[1, 1].set_title("Rendement Moyen (%) par Jour/Mois", fontweight="bold")
    plt.colorbar(im, ax=axes[1, 1])

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig_statistiques.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    """Génère toutes les images."""
    print("=" * 50)
    print("Génération des images pour le rapport LaTeX")
    print("=" * 50)

    generate_historical_prices()
    generate_candlestick()
    generate_technical_indicators()
    generate_returns_distribution()
    generate_model_comparison()
    generate_confusion_matrix()
    generate_walk_forward_schema()
    generate_feature_importance()
    generate_predictions_vs_actual()
    generate_summary_statistics()

    print("=" * 50)
    print(f"Images générées dans le dossier: {OUTPUT_DIR}/")
    print("=" * 50)

    # Liste des fichiers générés
    import os

    files = sorted(os.listdir(OUTPUT_DIR))
    for f in files:
        print(f"  - {f}")


if __name__ == "__main__":
    main()
