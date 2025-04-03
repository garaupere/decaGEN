import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def descript(df):
    # Anàlisi Descriptiva
    print("Descripció de les Freqüències:")
    print(df["Freqüència"].describe())

    print("\nDescripció de la Complexitat:")
    print(df["Complexitat"].describe())

    # Gràfic de Dispersió
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="Complexitat", y="Freqüència", data=df)
    plt.title("Relació entre Complexitat i Freqüència")
    plt.xlabel("Complexitat")
    plt.ylabel("Freqüència")
    plt.show()

    # Histogrammes
    plt.figure(figsize=(10, 6))
    sns.histplot(df["Freqüència"], bins=30, kde=True)
    #plt.title("Distribució de la Freqüència")
    plt.xlabel("Freqüència")
    plt.ylabel("Recompte")
    plt.savefig("compared/Freqüència.png", dpi=600)

    plt.figure(figsize=(10, 6))
    sns.histplot(df["Complexitat"], bins=30, kde=True)
    #plt.title("Distribució de la Complexitat")
    plt.xlabel("Complexitat")
    plt.ylabel("Recompte")
    plt.savefig("compared/Complexitat.png", dpi=600)

    # Density Plots
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df["Freqüència"], shade=True)
    #plt.title("Distribució de la Freqüència")
    plt.xlabel("Freqüència")
    plt.savefig("compared/Freqüència_density.png", dpi=600)


    # Box Plots
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Freqüència", data=df)
    plt.title("Box Plot de la Freqüència")
    plt.xlabel("Freqüència")
    plt.show()

    # Dades d'exemple
    data = {
        "Coincidència": ["o1980", "o1988", "o1992b", "d2006", "o2008"],
        "Freqüència": [20, 15, 10, 5, 8]
    }

    # Crear DataFrame
    df = pd.DataFrame(data)

    # Gràfic de Barres
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Coincidència", y="Freqüència", data=df)
    plt.title("Distribució de Coincidències per Categoria")
    plt.xticks(data["Coincidència"])
    plt.xlabel("Categoria")
    plt.ylabel("Freqüència")
    plt.show()

    # Dades d'exemple
    data = {
        "Complexitat": [0, 0.2, 0.25, 0.333, 0.5, 0.666, 1, 1.5, 2, 3, 4.25, 9],
        "Freqüència": [9033, 2616, 44, 66, 1, 1, 0, 0, 24, 72, 1691, 20]
    }

    # Crear DataFrame
    df = pd.DataFrame(data)

    # Gràfic de Dispersió
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="Complexitat", y="Freqüència", data=df)
    plt.title("Relació entre Complexitat i Freqüència")
    plt.xlabel("Complexitat")
    plt.ylabel("Freqüència")
    plt.show()