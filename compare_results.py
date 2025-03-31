"""Un mòdul per a comparar els resultats dels diferents generadors i avaluadors."""
from copy import deepcopy

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tabulate import tabulate

import gen
import eval
from decaGEN import grammar


def stress(pattern):
    """Equipara el patró a valors numèrics"""
    stress_values = []
    for pos in pattern:
        if pos == 'A':
            stress_values.append(0)
        elif pos == 'T':
            stress_values.append(100)
    return stress_values


def convert_df(df):
    """Converteix els valors de text dels Exemples del DF en valors numèrics"""
    df['Exemple'] = df['Exemple'].apply(stress)
    return df


def plot_stress(df):
    """Per a cada generador del DF, obtén la mitjana de tonicitat per síl·laba i fes un gràfic amb tots els valors"""
    df_original = df.copy()

    x_values = np.arange(1, 11)
    for generator in df['Generador'].unique():
        df_gen = df[df['Generador'] == generator]
        values = df_gen['Exemple'].apply(pd.Series).mean()
        plt.plot(x_values, values, label=generator)

    plt.legend()
    plt.xticks(np.arange(0, 11))
    plt.tight_layout()
    plt.savefig("compared/stress.png", dpi=600)
    plt.close()

    #Ara calcula la mitjana de tonicitat per síl·laba de tots els generadors i fes un gràfic amb el valor mitjà de cada síl·laba i la std
    x_values = np.arange(1, 11)
    values = df['Exemple'].apply(pd.Series).mean()
    plt.plot(x_values, values, color='blue')
    plt.xticks(np.arange(0, 11))
    plt.tight_layout()
    plt.savefig("compared/stress_mean.png", dpi=600)
    plt.close()

    # Compara la mitjana de cada generador amb la mitjana real
    real = [23.93998628,	55.66301726,	32.91118477,	62.76294145,	15.94435333,	71.88584774,	22.50826433,	76.41943222,	5.379046379,	100]

    for generator in df['Generador'].unique():
        df_gen = df[df['Generador'] == generator]
        values = df_gen['Exemple'].apply(pd.Series).mean()
        plt.plot(x_values, values, label=generator)

    plt.plot(x_values, real, color='red', label='Real', linestyle='--')
    plt.legend()
    plt.xticks(np.arange(0, 11))
    plt.tight_layout()
    plt.savefig("compared/comparison.png", dpi=600)
    plt.close()

    # Desa els valors en un fitxer Excel
    npdf = pd.DataFrame()
    for generator in df['Generador'].unique():
        df_gen = df[df['Generador'] == generator]
        values = df_gen['Exemple'].apply(pd.Series).mean()
        npdf[generator] = values
    npdf = npdf.T
    real_series = pd.Series(real, name='Real')
    npdf = npdf._append(real_series)
    npdf.to_excel("compared/comparison.xlsx", index=True)




    # Ara compararam la mitjana teòrica amb la real
    plt.plot(x_values, values, color='red', label='Real')
    plt.plot(x_values, real, color='blue', label='Teòric')
    # Dibuixa la distància entre els valors teòrics i reals
    plt.fill_between(x_values, values, real, color='lightblue', alpha=0.2)
    # Escriu els valors de la distància
    for i, txt in enumerate(abs(np.subtract(real, values))):
        plt.annotate(f"{txt:.2f}", (x_values[i], values[i]), textcoords="offset points", xytext=(0, 10), ha='center')

    plt.legend()
    plt.xticks(np.arange(0, 11))
    plt.tight_layout()
    plt.savefig("compared/stress_mean_comparison.png", dpi=600)
    plt.close()


    # Ara comparam les mitjanes de cada generador amb la mitjana teòrica i en calculam la diferència. En el gràfic. L'eix Y serà la diferència. A X el valor de la mitjana real serà representat per [0, 0, 0, 0, 0, 0 ,0, 0, 0, 0] i així podrem veure la diferència.

    diffs = []
    for generator in df['Generador'].unique():
        df_gen = df[df['Generador'] == generator]
        values = df_gen['Exemple'].apply(pd.Series).mean()
        diff = [abs(x) for x in np.subtract(real, values)]
        diffs.append([generator, diff])
        plt.plot(x_values, diff, label=generator)

    plt.plot(x_values, np.zeros(10), color='red', label='Real', linestyle='--')
    plt.legend()
    plt.xticks(np.arange(0, 11))
    plt.tight_layout()
    plt.savefig("compared/stress_mean_diff.png", dpi=600)
    plt.close()


    ndiffs = []
    for diff in diffs:
        generador = diff[0]
        values = diff[1]
        diff_mean = np.mean(values)
        ndiffs.append([generador, values, diff_mean])

    # Ordena els generadors per la mitjana de la diferència (menor->major)
    diffs = sorted(ndiffs, key=lambda x: x[2])
    df = pd.DataFrame(ndiffs, columns=['Generador', 'Diferència', 'Mitjana']).drop(columns='Diferència')
    df.index = range(1, len(df) + 1)
    print(tabulate(df, headers='keys', tablefmt='psql'))

    # Calcula l'accuracy de cada generador
    accuracies = []
    for diff in diffs:
        generador = diff[0]
        values = diff[1]
        accuracy = (100 - np.mean(values)) / 100
        accuracies.append([generador, accuracy])

    # Ordena els generadors per l'accuracy (major->menor)
    accuracies = sorted(accuracies, key=lambda x: x[1], reverse=True)
    df = pd.DataFrame(accuracies, columns=['Generador', 'Accuracy'])
    df.index = range(1, len(df) + 1)
    print(tabulate(df, headers='keys', tablefmt='psql'))



def add_soroll(df):
    """Afegeix soroll als valors teòrics. Es calcula un valor de potència per a cada patró a partir del valor de complexitat.
    potència = (10-complexitat) * 100"""

    soroll = []
    for index, row in df.iterrows():
        complexitat = row['Complexitat']
        soroll.append((10 - complexitat) * 100)
    df['Soroll'] = soroll

    # Ara repeteix els patrons tantes vegades com indica la potència
    new_df = pd.DataFrame()
    for index, row in df.iterrows():
        for i in range(int(row['Soroll'])):
            new_df = new_df._append(row)
            print(row)
    new_df = new_df.drop(columns='Soroll')
    new_df.index = range(1, len(new_df) + 1)
    # Desa-ho en un fitxer Excel
    new_df.to_excel("compared/with_soroll.xlsx", index=False)
    return new_df



def get_var_name(variable):
    for name, value in globals().items():
        if value is variable:
            return name


def soroll(dnum):
    # Soroll
    print("-" * 50)
    print("Soroll")
    print("-" * 50)
    dnum = add_soroll(dnum)
    print(tabulate(dnum, headers='keys', tablefmt='psql'))

    # A partir de les dades amb soroll fes la comparativa
    plot_stress(dnum)
    print("-" * 50)

def false_positives_test(dfs):
    # Genera un DF amb els patrons de cada generador
    patterns = []
    for generator in dfs:
        for index, row in generator.iterrows():
            patterns.append(row['Exemple'])
    patterns = list(set(patterns))

    # Genera un DF amb els patrons reals
    real = pd.read_excel("compared/real.xlsx")
    real = real['patró'].tolist()

    # Genera un DF que identifica els patrons de cada generador que:
    # 1) Són generats i apareixen en les dades reals (true positives) [x] [x]
    # 2) Són generats i no apareixen en les dades reals (false positives) [x] [ ]
    # 3) No són generats i apareixen en les dades reals (false negatives) [ ] [x]

    patrons = patterns + real
    patrons = list(set(patrons))
    results = []
    for pattern in patrons: # Per a cada patró
        for generator in dfs: # Per a cada generador
            if pattern in generator['Exemple'].values: # Si el patró és generat pel generador
                if pattern in real: # Si el patró també és real
                    results.append([pattern, generator['Generador'].values[0], 'TP']) # True Positive
                else: # Si el patró no és real
                    results.append([pattern, generator['Generador'].values[0], 'FP']) # False Positive
            else: # Si el patró no és generat pel generador
                if pattern in real: # Si el patró és real
                    results.append([pattern, generator['Generador'].values[0], 'FN']) # False Negative


    results = pd.DataFrame(results, columns=['Patró', 'Generador', 'Resultat'])
    results.index = range(1, len(results) + 1)
    print(tabulate(results, headers='keys', tablefmt='psql'))

    # Ara disposa-ho en un DataFrame que tengui l'estructura següent:
    # Generador | TP | FP | FN | Precision | Recall | F1
    # On:
    # Precision = TP / (TP + FP)
    # Recall = TP / (TP + FN)
    # F1 = 2 * (Precision * Recall) / (Precision + Recall)

    # Genera un DF amb els valors de TP, FP i FN per a cada generador
    df_results = []
    for generator in dfs:
        tp = len(results[(results['Generador'] == get_var_name(generator)) & (results['Resultat'] == 'TP')])
        fp = len(results[(results['Generador'] == get_var_name(generator)) & (results['Resultat'] == 'FP')])
        fn = len(results[(results['Generador'] == get_var_name(generator)) & (results['Resultat'] == 'FN')])
        df_results.append([get_var_name(generator), tp, fp, fn])

    print(tabulate(df_results, headers=['Generador', 'True Positive', 'False Positive', 'False Negative'], tablefmt='psql'))

    # Ara calcula la precisió, recall i F1 per a cada generador
    df_results = pd.DataFrame(df_results, columns=['Generador', 'True Positive', 'False Positive', 'False Negative'])
    df_results['Precision'] = df_results['True Positive'] / (df_results['True Positive'] + df_results['False Positive'])
    df_results['Recall'] = df_results['True Positive'] / (df_results['True Positive'] + df_results['False Negative'])
    df_results['F1'] = 2 * (df_results['Precision'] * df_results['Recall']) / (df_results['Precision'] + df_results['Recall'])
    print(tabulate(df_results, headers='keys', tablefmt='psql'))
    # Desa el DataFrame a un fitxer Excel
    df_results.to_excel("compared/results.xlsx", index=False)

    # Mostra els resultats en un gràfic
    plt.figure(figsize=(10, 6))
    plt.bar(df_results['Generador'], df_results['Precision'], label='Precision')
    plt.bar(df_results['Generador'], df_results['Recall'], label='Recall')
    plt.bar(df_results['Generador'], df_results['F1'], label='F1')
    plt.legend()
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("compared/results.png", dpi=600)
    plt.close()



if __name__ == "__main__":
    model = 'WSWSWSWSWS'
    o1980 = grammar(model, gen.oliva1980, eval.oliva1980)
    o1988 = grammar(model, gen.oliva1988, eval.oliva1980)
    o1992b = grammar(model, gen.oliva1992b, eval.oliva1992b)
    d2006 = grammar(model, gen.dols2006, eval.oliva1980)
    o2008 = grammar(model, gen.oliva2008, eval.oliva1980)

    # Substitueix els valors de complexitat de d2006 per 0
    d2006['Complexitat'] = 0

    # Per a o1992b substitueix en els exemples els valors de 't' per 'T'.
    o1992b['Exemple'] = o1992b['Exemple'].apply(lambda x: x.replace('t', 'T'))

    # Per a o1992b calcula la complexitat mitjana dels exemples duplicats i després elimina els duplicats.
    o1992b['Complexitat'] = o1992b.groupby('Exemple')['Complexitat'].transform('mean')
    o1992b = o1992b.drop_duplicates(subset='Exemple')

    # Inclou tots els DF en una llista
    dfs = [o1980, o1988, o1992b, d2006, o2008]

    # Per a cada DF, crea una columna amb el nom del generador
    for df in dfs:
        df['Generador'] = get_var_name(df)

    # Concatena tots els DFs en un de sol
    d = pd.concat(dfs, axis=0).fillna(0)

    # Mostra en una nova columna ('Coincidències') si el resultat d'un generador és generat també per altres generadors. Indica-hi quins.
    d['Coincidències'] = d['Exemple'].apply(lambda x: ', '.join(d[d['Exemple'] == x]['Generador'].unique()))

    # Elimina les columnes supèrflues (conserva només: índex, generador, exemple, complexitat, coincidències)
    d = d[['Generador', 'Exemple', 'Complexitat', 'Coincidències']]

    print(tabulate(d, headers='keys', tablefmt='psql'))

    # Mostra la diferència de mida entre els resultats de cada generador
    print("-" * 50)
    print(d.groupby('Generador').size())
    print("-" * 50)

    # Converteix els valors de text dels Exemples del DF en valors numèrics
    dnum = convert_df(d)
    print(tabulate(dnum, headers='keys', tablefmt='psql'))

    # Per a cada generador del DF, obtén la mitjana de tonicitat per síl·laba i fes un gràfic amb tots els valors
    plot_stress(dnum)


    # De tot el DF mostra només els exemples únics, calcula la complexitat mitjana per exemple i ordena per complexitat
    d_unique = d.drop_duplicates(subset='Exemple').sort_values(by='Complexitat')
    d_unique.index = range(1, len(d_unique) + 1)
    print(tabulate(d_unique, headers='keys', tablefmt='psql'))

    # Desa el DataFrame a un fitxer Excel
    d_unique.to_excel("compared/compare_results.xlsx", index=True)

    # Ara converteix els valors numèrics dels exemples en valors de text:
    # [0, 100...] = AT, on 0 = A i 100 = T
    d_unique['Exemple'] = d_unique['Exemple'].apply(lambda x: ''.join(['A' if y == 0 else 'T' for y in x]))

    # Del conjunt, calcula un valor de complexitat dividint el valor actual entre el nombre de generadors que han generat l'exemple
    d_unique['Complexitat'] = d_unique['Complexitat'] / d_unique['Coincidències'].apply(lambda x: len(x.split(', ')))
    d_unique = d_unique.sort_values(by='Complexitat')
    d_unique.index = range(1, len(d_unique) + 1)
    print(tabulate(d_unique, headers='keys', tablefmt='psql'))
    # Desa el DataFrame a un fitxer Excel
    d_unique.to_excel("compared/compare_results_normalized.xlsx", index=True)
    ##############################################################################
    false_positives_test(dfs)
    ##############################################################################
    # Ara investiga si hi ha una correlació entre la complexitat dels patrons generats i la freqüència real dels patrons
    real = pd.read_excel("compared/real.xlsx")

    # Per a cada patró generat registra la freqüència real del patró
    freqs = []
    for index, row in d_unique.iterrows():
        freq = real[real['patró'] == row['Exemple']]['freq'].values
        if len(freq) == 0:
            freqs.append(0)
        else:
            freqs.append(freq[0])
    d_unique['Freqüència'] = freqs

    print(tabulate(d_unique, headers='keys', tablefmt='psql'))
    d_unique.to_excel("compared/compare_results_normalized_freq.xlsx", index=True)


    # Calcula la correlació entre la complexitat i la freqüència
    correlation = d_unique['Complexitat'].corr(d_unique['Freqüència'], method='pearson')
    print(f"Correlació entre complexitat i freqüència: {correlation:.4f}")

    # Mostra la correlació en un gràfic amb la línia de regressió i el valor de R
    plt.scatter(d_unique['Complexitat'], d_unique['Freqüència'])
    plt.plot(np.unique(d_unique['Complexitat']), np.poly1d(np.polyfit(d_unique['Complexitat'], d_unique['Freqüència'], 1))(np.unique(d_unique['Complexitat'])), color='red')
    plt.text(0.1, 0.9, f"R= {correlation:.2f}", transform=plt.gca().transAxes)
    plt.xlabel('Complexitat')
    plt.ylabel('Freqüència')
    plt.tight_layout()
    plt.savefig("compared/correlation.png", dpi=600)
    plt.close()







