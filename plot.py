import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    if len(sys.argv) < 2:
        print("Utilisation: python3 plot.py runs/run_xxx_xxx")
        sys.exit(1)
        
    run_dir = sys.argv[1]
    csv_file = os.path.join(run_dir, "stats.csv")
    if not os.path.exists(csv_file):
        print(f"Fichier {csv_file} introuvable. Est-ce un dossier de run valide ?")
        sys.exit(1)
        
    print(f"Chargement de {csv_file}...")
    df = pd.read_csv(csv_file)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Épisode')
    ax1.set_ylabel('Score Moyen (Cumulé)', color=color)
    ax1.plot(df['Episode'], df['AvgScore'], color=color, label='Score Moyen')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Tuile Max (Globale)', color=color)  
    ax2.plot(df['Episode'], df['MaxTile'], color=color, alpha=0.5, label='Tuile Max', linestyle='-.')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Échelle log base 2 pour rendre les puissances de 2 linéaires
    ax2.set_yscale('log', base=2)
    # Ajouter des labels en puissance de 2 propres
    import matplotlib.ticker as ticker
    ax2.yaxis.set_major_formatter(ticker.ScalarFormatter())
    
    plt.title(f"Statistiques d'Apprentissage (N-Tuple RL)\n{os.path.basename(run_dir)}")
    fig.tight_layout()
    
    out_file = os.path.join(run_dir, "learning_curve.png")
    plt.savefig(out_file, dpi=300)
    print(f"Graphique généré avec succès dans : {out_file}")

if __name__ == "__main__":
    main()
