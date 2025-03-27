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
    freq_real = {
        "ATATATATAT": 35.88083416, "AATAATATAT": 14.20456802, "TAATATATAT": 10.39126117, "TATAATATAT": 6.716981132,
        "ATATAATAAT": 5.167825223, "TAATAATAAT": 2.387288977, "AATATATAAT": 1.982125124, "ATAATAATAT": 1.882820258,
        "ATTAATATAT": 1.620655412, "ATAATATAAT": 1.557100298, "AATTATATAT": 1.441906653, "ATATAATTAT": 1.410129096,
        "TATATATAAT": 1.374379345, "ATATATAATT": 1.279046673, "AATATAATAT": 1.203574975, "TATATAATAT": 1.024826216,
        "ATAAATATAT": 0.691161867, "AATAATAATT": 0.683217478, "TATTATATAT": 0.623634558, "TAATAATTAT": 0.512413108,
        "ATATAATATT": 0.476663357, "TATAATAATT": 0.413108242, "ATAATTATAT": 0.381330685, "ATATATAAAT": 0.309831182,
        "ATAATATATT": 0.297914598, "TAATATAATT": 0.293942403, "AATAATTAAT": 0.285998014, "AATTAATAAT": 0.278053625,
        "ATATATTAAT": 0.262164846, "ATATAAATAT": 0.250248262, "TATATATATT": 0.238331678, "ATTATATAAT": 0.2224429,
        "AATAATAAAT": 0.206554121, "AATATATATT": 0.186693148, "TAATAATATT": 0.178748759, "AAATATATAT": 0.174776564,
        "TATTAATAAT": 0.158887786, "ATATATATTT": 0.158887786, "AATATTATAT": 0.142999007, "ATATTAATAT": 0.139026812,
        "TATATTATAT": 0.135054618, "ATTATAATAT": 0.119165839, "AATATATTAT": 0.107249255, "AATTAATTAT": 0.103277061,
        "ATTTATATAT": 0.099304866, "TATAATTAAT": 0.095332671, "ATATTATAAT": 0.095332671, "ATTAATAATT": 0.091360477,
        "AATAAATAAT": 0.087388282, "TATAATAAAT": 0.079443893, "ATAATATTAT": 0.079443893, "TATATATTAT": 0.075471698,
        "TAATAAATAT": 0.063555114, "TAATATATTT": 0.05958292, "TATAAATAAT": 0.05958292, "TAATTAATAT": 0.055610725,
        "TAAAATATAT": 0.047666336, "TAATATAAAT": 0.047666336, "AATTATAATT": 0.047666336, "AAATAATAAT": 0.043694141,
        "TATTAATTAT": 0.043694141, "TAATATTAAT": 0.043694141, "ATTTAATAAT": 0.043694141, "ATTAATAAAT": 0.043694141,
        "TAAATATAAT": 0.039721946, "AATTAATATT": 0.039721946, "ATTATATATT": 0.039721946, "AATAATTATT": 0.039721946,
        "ATAAATAATT": 0.039721946, "ATATATTTAT": 0.035749752, "ATATTATATT": 0.031777557, "TATAATATTT": 0.031777557,
        "TATTATAATT": 0.027805362, "ATAATAATTT": 0.027805362, "TAATTATAAT": 0.027805362, "TAAATAATAT": 0.027805362,
        "ATTAAATAAT": 0.023833168, "AATAATATTT": 0.023833168, "ATTTAATTAT": 0.023833168, "ATTATTATAT": 0.023833168,
        "ATTATATTAT": 0.023833168, "AATATTAATT": 0.019860973, "TATATAATTT": 0.019860973, "ATATATTATT": 0.019860973,
        "TATAATTATT": 0.019860973, "ATAATTAATT": 0.019860973, "AATTAAATAT": 0.015888779, "ATAAATTAAT": 0.015888779,
        "TAAATATATT": 0.011916584, "AATTATATTT": 0.011916584, "TAATTTATAT": 0.011916584, "ATAATTAAAT": 0.011916584,
        "AAATAATATT": 0.011916584, "ATAATAAATT": 0.011916584, "ATTAATTAAT": 0.011916584, "TATTAATATT": 0.011916584,
        "AATTTAATAT": 0.011916584, "TATAATTTAT": 0.011916584, "ATTAATATTT": 0.011916584, "AATATAATTT": 0.011916584,
        "ATTTATAATT": 0.011916584, "ATATTTATAT": 0.011916584, "TAAATTATAT": 0.007944389, "AAATATTAAT": 0.007944389,
        "ATTAATTATT": 0.007944389, "AATATAAATT": 0.007944389, "AATATATTTT": 0.007944389, "AATTAATTTT": 0.007944389,
        "AATAAATTAT": 0.007944389, "ATATAAAATT": 0.007944389, "TATAAATATT": 0.007944389, "ATATTATTAT": 0.007944389,
        "ATAAAATAAT": 0.007944389, "TATATAAAAT": 0.007944389, "ATATTAATTT": 0.007944389, "AAATAATTAT": 0.007944389,
        "ATATTTAAAT": 0.003972195, "ATAATAAAAT": 0.003972195, "AATATTAAAT": 0.003972195, "AATATTTATT": 0.003972195,
        "ATTTAATATT": 0.003972195, "ATTTTATAAT": 0.003972195, "TATATATTTT": 0.003972195, "ATATTTTAAT": 0.003972195,
        "ATAAATAAAT": 0.003972195, "TAATTATTAT": 0.003972195, "TAATAAATTT": 0.003972195, "ATTTTTATAT": 0.003972195,
        "TAATTAAATT": 0.003972195, "ATATTTATTT": 0.003972195, "AATATAAAAT": 0.003972195, "ATTATTTAAT": 0.003972195,
        "AATTATAAAT": 0.003972195, "ATAAATATTT": 0.003972195, "AAATTATAAT": 0.003972195, "AATAATTTAT": 0.003972195,
        "AAATATTATT": 0.003972195, "AAATATAATT": 0.003972195, "ATTTTAATTT": 0.003972195, "ATTTATTATT": 0.003972195,
        "TATATAAATT": 0.003972195, "AATTTATAAT": 0.003972195, "TAATAATTTT": 0.003972195, "TATTTATAAT": 0.003972195,
        "AATTATTATT": 0.003972195, "ATATAATTTT": 0.003972195, "AAATAAATAT": 0.003972195, "ATTAATTTTT": 0.003972195,
        "AATTTTATAT": 0.003972195, "TAATTAATTT": 0.003972195, "TATATTTAAT": 0.003972195, "TAAATTATTT": 0.003972195,
        "ATATTTAATT": 0.003972195, "TATTATTATT": 0.003972195, "ATTAATTTAT": 0.003972195, "TAATTATATT": 0.003972195,
        "TAATATTATT": 0.003972195, "TATTATTAAT": 0.003972195, "AAATATAAAT": 0.003972195, "TATTATATTT": 0.003972195,
        "AAATTAATAT": 0.003972195, "ATAATTTTAT": 0.003972195, "TATTTAATTT": 0.003972195, "AATTTATATT": 0.003972195,
        "ATTTATTAAT": 0.003972195, "ATAATTTAAT": 0.003972195, "TATATTAATT": 0.003972195
    }

    # Converteix els patrons
    freq_real_nou = []
    for key in freq_real.keys():
        freq_real_nou.append([stress(key), freq_real[key]])

    freq_real = freq_real_nou

    print(freq_real)

    print(df_original['Exemple'])
    # Identifica els patrons generats dins dels resultats reals
    real = [0] * 10
    for example in df_original['Exemple']:
        for i in range(10):
            real[i] += example[i]

    # Normalitza els valors reals
    real = [x / sum(real) * 100 for x in real]



    for generator in df['Generador'].unique():
        df_gen = df[df['Generador'] == generator]
        values = df_gen['Exemple'].apply(pd.Series).mean()
        plt.plot(x_values, values, label=generator)

    plt.plot(x_values, real, color='red', label='Real')
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
    plt.plot(x_values, values, color='blue', label='Real')
    plt.plot(x_values, real, color='red', label='Teòric')
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

    plt.plot(x_values, np.zeros(10), color='red', label='Real')
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
        accuracy = 100 - np.mean(values)
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
    d_original = deepcopy(d)

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

    # Ara calcula la probabilitat de cada generador de generar un exemple.
    """## Fórmula per calcular la probabilitat ajustada
        La fórmula usada per calcular la probabilitat ajustada \( P'(i) \) de cada exemple \( i \) és:
        \[ P'(i) = \frac{C(i) \times \frac{1}{1 + \text{Complexitat}(i)}}{N'} \]
        ### On:
        - \( P'(i) \) és la probabilitat ajustada de l'exemple \( i \).
        - \( C(i) \) és el nombre de coincidències per l'exemple \( i \).
        - \( \text{Complexitat}(i) \) és la complexitat de l'exemple \( i \).
        - \( N' \) és la suma de totes les coincidències ajustades del conjunt de dades."""
    # Per a cada exemple compta el nombre de generadors que l'han generat
    d_unique['Coincidències'] = d_unique['Coincidències'].apply(lambda x: len(x.split(', ')))
    # Ajusta la complexitat sumant-hi 1
    d_unique['Complexitat'] = d_unique['Complexitat'] + 1
    # Calcula la probabilitat: (nombre de generadors de l'exemple  / complexitat de l'exemple) / suma total de coincidències
    d_unique['Probabilitat'] = d_unique['Coincidències'] / d_unique['Complexitat'] / d_unique['Coincidències'].sum() * 1000
    d_unique = d_unique.sort_values(by='Probabilitat', ascending=False)
    d_unique.index = range(1, len(d_unique) + 1)
    print(tabulate(d_unique, headers='keys', tablefmt='psql'))
    d_unique.to_excel("compared/compare_results_normalized_prob.xlsx", index=True)

    # Ara compara la probabilitat de cada patró amb la freqüència real d'aparició de cada patró.
    # Obté els valors reals de l'excel 'real.xlsx'
    real = pd.read_excel("compared/real.xlsx")
    real = real.set_index('patró')

    # Ara cerca els valors de d_unique a real i calcula la diferència entre la probabilitat real i la calculada. Pot ser que els valors no hi siguin, si no hi són defineix la freqüència com el valor NaN.
    d_unique['Freqüència'] = d_unique['Exemple'].apply(lambda x: real.loc[x, 'freq'] if x in real.index else np.nan)
    d_unique['Diferència'] =  d_unique['Probabilitat'] - d_unique['Freqüència']
    #d_unique = d_unique.sort_values(by='Diferència')
    #d_unique.index = range(1, len(d_unique) + 1)
    print(tabulate(d_unique, headers='keys', tablefmt='psql'))

    # Desa el DataFrame a un fitxer Excel
    d_unique.to_excel("compared/compare_results_normalized_prob_freq.xlsx", index=True)

    # A partir dels valors reals i els de d compta quins casos de d no apareixen a real.
    d = deepcopy(d)
    dn = d.drop_duplicates(subset='Exemple')
    dn.index = range(1, len(dn) + 1)
    dn['Exemple'] = dn['Exemple'].apply(lambda x: ''.join(['A' if y == 0 else 'T' for y in x]))
    # Afegeix la columna 'Apareix al real' amb True si l'exemple apareix a real i False si no.
    dn['Apareix al real'] = dn['Exemple'].apply(lambda x: x in real.index)

    # Retorna el nombre de casos que apareixen a real i el nombre de casos que no.
    print("-" * 50)
    print(dn['Apareix al real'].value_counts())
    print("=" * 50)
    print(tabulate(dn, headers='keys', tablefmt='psql'))

    # Per a cada generador, calcula quants dels seus exemples apareixen a real
    print("-" * 50)
    d['Exemple'] = d['Exemple'].apply(lambda x: ''.join(['A' if y == 0 else 'T' for y in x]))
    d['Apareix al real'] = d['Exemple'].apply(lambda x: x in real.index)
    print(d.groupby('Generador')['Apareix al real'].value_counts())
    # Percentatge de precisió (nombre de casos que apareixen a real / nombre de casos generats pel generador)
    print("-" * 50)
    print(d.groupby('Generador')['Apareix al real'].value_counts(normalize=True))

    print("=" * 50)




