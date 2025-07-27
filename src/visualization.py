import matplotlib.pyplot as plt
import seaborn as sns

def plot_top_countries(df, top_n=15):
    """Grafica el top N de países con más colombianos registrados."""
    plt.figure(figsize=(12, 8))
    country_counts = df['País'].value_counts().head(top_n)
    sns.barplot(x=country_counts.values, y=country_counts.index, palette='viridis')
    plt.title(f'Top {top_n} Países de Residencia de Colombianos en el Exterior')
    plt.xlabel('Cantidad de Registros')
    plt.ylabel('País')
    plt.tight_layout()
    # Guardar la figura es una buena práctica
    plt.savefig('reports/figures/top_countries_distribution.png')
    plt.show()

# Puedes crear funciones similares para distribuciones de edad, nivel académico, etc.