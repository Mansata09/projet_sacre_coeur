import os
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================

LOG_PATH      = os.path.join("..", "logs", "training_log.csv")
OUTPUT_DIR    = os.path.join("..", "logs")
SMOOTH_WINDOW = 10   # moyenne mobile pour lisser la courbe

# ============================================================
# CHARGEMENT
# ============================================================

def load_log(path=LOG_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Log introuvable : {path}\n"
            f"Lance d'abord l'entrainement (ppo_trainer.py) pour generer ce fichier."
        )
    df = pd.read_csv(path)
    return df

# ============================================================
# GRAPHIQUES
# ============================================================

def plot_training_curves(df, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ---- 1. Reward moyenne globale, avec moyenne mobile ----
    ax = axes[0, 0]
    ax.plot(df["episode"], df["reward_moyenne"], alpha=0.3, color="steelblue", label="brute")
    if len(df) >= SMOOTH_WINDOW:
        smoothed = df["reward_moyenne"].rolling(SMOOTH_WINDOW).mean()
        ax.plot(df["episode"], smoothed, color="steelblue", linewidth=2, label=f"moyenne mobile ({SMOOTH_WINDOW})")
    ax.set_title("Recompense moyenne par episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward moyenne")
    ax.legend()
    ax.grid(alpha=0.3)

    # ---- 2. Reward par branche ----
    ax = axes[0, 1]
    for col, label, color in [
        ("reward_nord",  "Nord",  "tab:blue"),
        ("reward_sud",   "Sud",   "tab:orange"),
        ("reward_est",   "Est",   "tab:green"),
        ("reward_ouest", "Ouest", "tab:red"),
    ]:
        if len(df) >= SMOOTH_WINDOW:
            smoothed = df[col].rolling(SMOOTH_WINDOW).mean()
            ax.plot(df["episode"], smoothed, label=label, color=color)
        else:
            ax.plot(df["episode"], df[col], label=label, color=color)
    ax.set_title("Recompense par branche (lissee)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward cumulee")
    ax.legend()
    ax.grid(alpha=0.3)

    # ---- 3. Accidents par episode ----
    ax = axes[1, 0]
    accident_int = df["accident"].astype(int)
    ax.bar(df["episode"], accident_int, color="crimson", alpha=0.7)
    ax.set_title("Accidents detectes par episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Accident (0/1)")
    ax.set_yticks([0, 1])
    ax.grid(alpha=0.3)

    # ---- 4. Steps completes par episode ----
    ax = axes[1, 1]
    ax.plot(df["episode"], df["step_total"], color="purple")
    ax.set_title("Duree de l'episode (steps avant fin)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.grid(alpha=0.3)

    plt.tight_layout()

    output_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(output_path, dpi=150)
    print(f"Graphique sauvegarde -> {output_path}")

    plt.show()


# ============================================================
# RESUME CHIFFRE
# ============================================================

def print_summary(df):
    print("\n=== RESUME ENTRAINEMENT ===")
    print(f"Episodes completes      : {len(df)}")
    print(f"Reward moyenne (debut)  : {df['reward_moyenne'].head(10).mean():.4f}  (10 premiers episodes)")
    print(f"Reward moyenne (fin)    : {df['reward_moyenne'].tail(10).mean():.4f}  (10 derniers episodes)")
    print(f"Accidents totaux        : {df['accident'].sum()} / {len(df)} episodes")
    print(f"Step moyen par episode  : {df['step_total'].mean():.1f}")


# ============================================================
# POINT D'ENTREE
# ============================================================

if __name__ == "__main__":
    df = load_log()
    print_summary(df)
    plot_training_curves(df)