import pandas as pd

# Fichier CSV contenant les segments
CSV_FILE = "/mnt/iridia/sehlalou/thesis/rrr_segments_all_RR.csv"

def top_longest_segments(csv_file, top_n=20):
    # Charger le fichier CSV
    df = pd.read_csv(csv_file)

    # Trier par durée décroissante
    df_sorted = df.sort_values(by="duration_seconds", ascending=False)

    # Sélectionner les top N segments les plus longs
    top_segments = df_sorted.head(top_n)

    # Afficher le résultat
    print(f"Top {top_n} des segments les plus longs :\n")
    print(top_segments.to_string(index=False))

    return top_segments

if __name__ == "__main__":
    top_segments = top_longest_segments(CSV_FILE)
