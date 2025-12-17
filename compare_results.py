"""Un mòdul per a comparar els resultats dels diferents generadors i avaluadors."""
import re
from copy import deepcopy

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tabulate import tabulate

import gen
import eval
import proves
from decaGEN import grammar


def obtain_real_patterns():
    #corpus = pd.read_excel("compared/corpus.xlsx")

    corpus = pd.read_csv("D:/corpora/actual/etiquetat/actual.csv", encoding='utf-8', delimiter=';')

    # el corpus són només les mostres amb subdivisió 'Cap'
    corpus = corpus[corpus['subdivisió'] == 'Cap']

    # Genera un patró numèric per cada patró comparatiu del corpus, A=0, T=1
    corpus['npattern'] = corpus['patró comparatiu']#.apply(lambda x: ''.join(['0' if y == 'A' else '1' for y in x]))
    corpus['patró comparatiu'] = corpus['patró comparatiu'].apply(lambda x: x.replace('0', 'A').replace('1', 'T'))

    df_freq = corpus[['patró comparatiu', 'npattern', 'autor']]

    print('-' * 20, 'Patrons del corpus', '-' * 20)
    print(tabulate(df_freq, headers='keys', tablefmt='psql'))

    # Afegeix una columna amb una llista dels autors que generen cada patró i elimina els duplicats
    df_freq['Autors'] = df_freq['patró comparatiu'].apply(
        lambda x: ', '.join(df_freq[df_freq['patró comparatiu'] == x]['autor'].unique()))

    # Ara compta la freqüència de cada patró comparatiu
    df_freq['Freqüència'] = df_freq.groupby('patró comparatiu')['patró comparatiu'].transform('count')

    df_freq.drop_duplicates(subset='patró comparatiu', inplace=True)
    df_freq.index = range(1, len(df_freq) + 1)

    # Elimina la columna 'autor'
    df_freq.drop(columns='autor', inplace=True)

    # Afegeix una columna amb el recompte d'autors
    df_freq["Nombre d'autors"] = df_freq['Autors'].apply(lambda x: len(x.split(', ')))

    print('=' * 20, 'Corpus de freqüències', '=' * 20)
    print(tabulate(df_freq, headers='keys', tablefmt='psql'))
    # Desa-ho en un fitxer Excel
    df_freq.to_excel("compared/corpus_patterns.xlsx", index=True)

    # Calcula el valor mitjà de tonicitat per síl·laba dels patrons del corpus
    mitjana_corpus = list(corpus['npattern'].apply(lambda x: [0 if y == '0' else 1 for y in x]).apply(pd.Series).mean())
    print(mitjana_corpus)
    return df_freq, mitjana_corpus, corpus


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


def plot_stress(df, mitjana_real, versio):
    """Per a cada generador del DF, obtén la mitjana de tonicitat per síl·laba i fes un gràfic amb tots els valors"""

    x_values = np.arange(1, 11)
    for generator in df['Generador'].unique():
        df_gen = df[df['Generador'] == generator]
        values = df_gen['Exemple'].apply(pd.Series).mean()
        plt.plot(x_values, values, label=generator)

    plt.legend()
    plt.xticks(np.arange(0, 11))
    plt.tight_layout()
    plt.savefig(f"compared/stress{versio}.png", dpi=600)
    plt.close()

    # Ara calcula la mitjana de tonicitat per síl·laba de tots els generadors i fes un gràfic amb el valor mitjà de cada síl·laba i la std
    x_values = np.arange(1, 11)
    values = df['Exemple'].apply(pd.Series).count() / len(df)
    plt.plot(x_values, values, color='blue')
    plt.xticks(np.arange(0, 11))
    plt.tight_layout()
    plt.savefig(f"compared/stress_mean{versio}.png", dpi=600)
    plt.close()

    # Compara la mitjana de cada generador amb la mitjana real
    for generator in df['Generador'].unique():
        df_gen = df[df['Generador'] == generator]
        values = df_gen['Exemple'].apply(pd.Series).mean()
        plt.plot(x_values, values, label=generator)

    plt.plot(x_values, mitjana_real, color='red', label='Real', linestyle='--')
    plt.legend()
    plt.xticks(np.arange(0, 11))
    plt.tight_layout()
    plt.savefig(f"compared/comparison{versio}.png", dpi=600)
    plt.close()

    # Desa els valors en un fitxer Excel
    npdf = pd.DataFrame()
    for generator in df['Generador'].unique():
        df_gen = df[df['Generador'] == generator]
        values = df_gen['Exemple'].apply(pd.Series).mean()
        npdf[generator] = values
    npdf = npdf.T
    real_series = pd.Series(mitjana_real, name='Real')
    npdf = npdf._append(real_series)
    npdf.to_excel(f"compared/comparison{versio}.xlsx", index=True)

    # Ara compararam la mitjana teòrica amb la real
    plt.plot(x_values, values, color='red', label='Teòric')
    plt.plot(x_values, mitjana_real, color='blue', label='Real')
    # Dibuixa la distància entre els valors teòrics i reals
    plt.fill_between(x_values, values, mitjana_real, color='lightblue', alpha=0.2)
    # Escriu els valors de la distància
    # for i, txt in enumerate(abs(np.subtract(mitjana_real, values))):
    #    plt.annotate(f"{txt:.2f}", (x_values[i], values[i]), textcoords="offset points", xytext=(0, 10), ha='center')

    plt.legend()
    plt.xticks(np.arange(0, 11))
    plt.tight_layout()
    plt.savefig(f"compared/stress_mean_comparison{versio}.png", dpi=600)
    plt.close()

    # Ara comparam les mitjanes de cada generador amb la mitjana teòrica i en calculam la diferència. En el gràfic. L'eix Y serà la diferència. A X el valor de la mitjana real serà representat per [0, 0, 0, 0, 0, 0 ,0, 0, 0, 0] i així podrem veure la diferència.

    diffs = []
    for generator in df['Generador'].unique():
        df_gen = df[df['Generador'] == generator]
        values = df_gen['Exemple'].apply(pd.Series).mean()
        diff = [abs(x) for x in np.subtract(mitjana_real, values)]
        diffs.append([generator, diff])
        plt.plot(x_values, diff, label=generator)

    plt.plot(x_values, np.zeros(10), color='red', label='Real', linestyle='--')
    plt.legend()
    plt.xticks(np.arange(0, 11))
    plt.tight_layout()
    plt.savefig(f"compared/stress_mean_diff{versio}.png", dpi=600)
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
    print('-' * 20, f'Diferència{versio}', '-' * 20)
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
    print('-' * 20, f'Accuracy{versio}', '-' * 20)
    print(tabulate(df, headers='keys', tablefmt='psql'))


def get_var_name(variable):
    for name, value in globals().items():
        if value is variable:
            return name


def false_positives_test(dfs, real, versio):
    # Genera un DF amb els patrons de cada generador
    patterns = []
    for generator in dfs:
        for index, row in generator.iterrows():
            patterns.append(row['Exemple'])
    patterns = list(set(patterns))

    # Genera un DF amb els patrons reals
    real = real['patró comparatiu'].tolist()

    # Genera un DF que identifica els patrons de cada generador que:
    # 1) Són generats i apareixen en les dades reals (true positives) [x] [x]
    # 2) Són generats i no apareixen en les dades reals (false positives) [x] [ ]
    # 3) No són generats i apareixen en les dades reals (false negatives) [ ] [x]

    patrons = patterns + real
    patrons = list(set(patrons))
    results = []
    for pattern in patrons:  # Per a cada patró
        for generator in dfs:  # Per a cada generador
            if pattern in generator['Exemple'].values:  # Si el patró és generat pel generador
                if pattern in real:  # Si el patró també és real
                    results.append([pattern, generator['Generador'].values[0], 'TP'])  # True Positive
                else:  # Si el patró no és real
                    results.append([pattern, generator['Generador'].values[0], 'FP'])  # False Positive
            else:  # Si el patró no és generat pel generador
                if pattern in real:  # Si el patró és real
                    results.append([pattern, generator['Generador'].values[0], 'FN'])  # False Negative

    results = pd.DataFrame(results, columns=['Patró', 'Generador', 'Resultat'])
    results.index = range(1, len(results) + 1)

    # Desa el DataFrame a un fitxer Excel
    results.to_excel(f"compared/results_full{versio}.xlsx", index=False)
    test_results = deepcopy(results)

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

    print('-' * 20, f'Results{versio}', '-' * 20)
    print(tabulate(df_results, headers=['Generador', 'True Positive', 'False Positive', 'False Negative'],
                   tablefmt='psql'))

    # Ara calcula la precisió, recall i F1 per a cada generador
    df_results = pd.DataFrame(df_results, columns=['Generador', 'True Positive', 'False Positive', 'False Negative'])
    df_results['Precision'] = df_results['True Positive'] / (df_results['True Positive'] + df_results['False Positive'])
    df_results['Recall'] = df_results['True Positive'] / (df_results['True Positive'] + df_results['False Negative'])
    df_results['F1'] = 2 * (df_results['Precision'] * df_results['Recall']) / (
                df_results['Precision'] + df_results['Recall'])
    print(tabulate(df_results, headers='keys', tablefmt='psql'))
    # Desa el DataFrame a un fitxer Excel
    df_results.to_excel(f"compared/results{versio}.xlsx", index=False)

    # Mostra els resultats en un gràfic
    plt.figure(figsize=(10, 6))
    plt.bar(df_results['Generador'], df_results['Precision'], label='Precision')
    plt.bar(df_results['Generador'], df_results['Recall'], label='Recall')
    plt.bar(df_results['Generador'], df_results['F1'], label='F1')
    plt.legend()
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"compared/results{versio}.png", dpi=600)
    plt.close()

    detall(test_results, versio)

    return test_results


def detall(fp, versio):
    # Per a cada Patró identifica els generadors que han generat el patró
    # Mostra en una nova columna ('Coincidències') si el resultat d'un generador és generat també per altres generadors. Indica-hi quins.
    fp['Generadors'] = fp['Patró'].apply(lambda x: ', '.join(fp[fp['Patró'] == x]['Generador'].unique()))

    # Ara elimina els patrons duplicats
    fp = fp.drop_duplicates(subset='Patró')
    fp = fp.drop(columns='Generador')

    # Per a cada FP compta les seqüències de 'T' i 'A' consecutives
    # Fes servir re.findall per a trobar totes les seqüències de 'T' i 'A' consecutives
    # Afegeix dues columnes al DF: 'T' i 'A' amb el recompte de seqüències de 'T' i 'A' respectivament
    for index, row in fp.iterrows():
        xocs = len(re.findall('TT', row['Patró'])) + len(re.findall('TTT', row['Patró']))
        valls = len(re.findall('AAA', row['Patró'])) + len(re.findall('AAAA', row['Patró']))
        posicions = []
        for i in range(0, len(row['Patró']) - 1):
            if row['Patró'][i] == 'T':
                posicions.append(i + 1)
        posicions = ', '.join(map(str, posicions))
        fp.loc[index, 'Xocs'] = xocs
        fp.loc[index, 'Valls'] = valls
        fp.loc[index, 'Posicions'] = posicions

    fp.index = range(1, len(fp) + 1)
    print(tabulate(fp, headers='keys', tablefmt='psql'))
    print(f'Mitjana de xocs: {fp["Xocs"].mean()}')
    print(f'Mitjana de valls: {fp["Valls"].mean()}')

    # Desa el DataFrame a un fitxer Excel
    fp.to_excel(f"compared/detall{versio}.xlsx", index=False)


def correlacions(d_unique, real, versio):
    # Investiga si hi ha una correlació entre la complexitat dels patrons generats i la freqüència real dels patrons
    # Per a cada patró generat registra la freqüència real del patró
    freqs = []
    for index, row in d_unique.iterrows():
        freq = real[real['patró comparatiu'] == row['Exemple']]['Freqüència'].values
        if len(freq) == 0:
            freqs.append(0)
        else:
            freqs.append(freq[0])
    d_unique['Freqüència'] = freqs

    print(tabulate(d_unique, headers='keys', tablefmt='psql'))
    d_unique.to_excel(f"compared/compare_results_normalized_freq{versio}.xlsx", index=True)

    # Calcula la correlació entre la complexitat i la freqüència
    correlation = d_unique['Complexitat'].corr(d_unique['Freqüència'], method='pearson')
    print(f"Correlació entre complexitat i freqüència: {correlation:.4f}")

    # Mostra la correlació en un gràfic amb la línia de regressió i el valor de R
    plt.scatter(d_unique['Complexitat'], d_unique['Freqüència'])
    plt.plot(np.unique(d_unique['Complexitat']),
             np.poly1d(np.polyfit(d_unique['Complexitat'], d_unique['Freqüència'], 1))(
                 np.unique(d_unique['Complexitat'])), color='red')
    plt.text(0.1, 0.9, f"R= {correlation:.2f}", transform=plt.gca().transAxes)
    plt.xlabel('Complexitat')
    plt.ylabel('Freqüència')
    plt.tight_layout()
    plt.savefig(f"compared/correlation{versio}.png", dpi=600)
    plt.close()

    # Ara investiga si hi ha una correlació entre el nombre de generadors que han generat un patró i la freqüència real dels patrons
    # Per a cada patró generat registra el nombre de generadors que han generat el patró
    freqs = []
    for index, row in d_unique.iterrows():
        freq = real[real['patró comparatiu'] == row['Exemple']]['Freqüència'].values
        if len(freq) == 0:
            freqs.append(0)
        else:
            freqs.append(freq[0])
    d_unique['Freqüència'] = freqs

    # Calcula la correlació entre el nombre de generadors i la freqüència
    correlation = d_unique['Coincidències'].apply(lambda x: len(x.split(', '))).corr(d_unique['Freqüència'],
                                                                                     method='pearson')
    print(f"Correlació entre nombre de generadors i freqüència: {correlation:.4f}")

    # Mostra la correlació en un gràfic amb la línia de regressió i el valor de R
    plt.scatter(d_unique['Coincidències'].apply(lambda x: len(x.split(', '))), d_unique['Freqüència'])
    x_values = np.unique(d_unique['Coincidències'].apply(lambda x: len(x.split(', '))))
    y_values = np.poly1d(
        np.polyfit(d_unique['Coincidències'].apply(lambda x: len(x.split(', '))), d_unique['Freqüència'], 1))(x_values)
    plt.plot(x_values, y_values, color='red')
    plt.text(0.1, 0.9, f"R= {correlation:.2f}", transform=plt.gca().transAxes)
    plt.xlabel('Nombre de Generadors')
    plt.ylabel('Freqüència')
    plt.tight_layout()
    plt.savefig(f"compared/correlation_generators{versio}.png", dpi=600)
    plt.close()


if __name__ == "__main__":
    model = 'WSWSWSWSWS'
    o1980 = grammar(model, gen.oliva1980, eval.oliva1980)
    o1988 = grammar(model, gen.oliva1988, eval.oliva1980)
    o1992b = grammar(model, gen.oliva1992b, eval.oliva1992b)
    #d2006 = grammar(model, gen.dols2006, eval.oliva1980)
    o2008 = grammar(model, gen.oliva2008, eval.oliva1980)
    #g2025 = grammar(model, gen.garau2025, eval.oliva1980)

    # Substitueix els valors de complexitat de d2006 per 0
    #d2006['Complexitat'] = 0
    # Per a o1992b substitueix en els exemples els valors de 't' per 'T'.
    o1992b['Exemple'] = o1992b['Exemple'].apply(lambda x: x.replace('t', 'T'))
    # Per a o1992b calcula la complexitat mitjana dels exemples duplicats i després elimina els duplicats.
    o1992b['Complexitat'] = o1992b.groupby('Exemple')['Complexitat'].transform('mean')
    o1992b = o1992b.drop_duplicates(subset='Exemple')

    # Inclou tots els DF en una llista
    dfs = [o1980, o1988, o1992b, o2008]

    # Per a cada DF, crea una columna amb el nom del generador
    for df in dfs:
        df['Generador'] = get_var_name(df)

    # Concatena tots els DFs en un de sol
    d = pd.concat(dfs, axis=0).fillna(0)
    d_original = deepcopy(d)
    print('-' * 20, 'Dades originals', '-' * 20)
    print(tabulate(d, headers='keys', tablefmt='psql'))

    # Mostra en una nova columna ('Coincidències') si el resultat d'un generador és generat també per altres generadors. Indica-hi quins.
    d['Coincidències'] = d['Exemple'].apply(lambda x: ', '.join(d[d['Exemple'] == x]['Generador'].unique()))

    # Elimina les columnes supèrflues (conserva només: índex, exemple, complexitat, coincidències)
    d = d[['Generador', 'Exemple', 'Complexitat', 'Coincidències']]

    print(tabulate(d, headers='keys', tablefmt='psql'))

    # Mostra la diferència de mida entre els resultats de cada generador
    print("-" * 50)
    print(d.groupby('Generador').size())
    print("-" * 50)

    corpus_raw = obtain_real_patterns()
    corpus = corpus_raw[0]
    mitjana_real = corpus_raw[1]
    corpus_raw = corpus_raw[2]


    # Mostra els percentils de freqüència real
    percentils = corpus['Freqüència'].quantile([0.25, 0.5, 0.75])
    print('Percentils de freqüència real:\n', percentils)

    # Elimina els casos que tenguin una freqüència inferior al tercer percentil
    corpus_drop = corpus[corpus['Freqüència'] >= percentils[0.75]]

    # Mostra la diferència de mida entre corpus i corpus_drop
    print("-" * 20, 'Diferència de mida entre corpus i corpus_drop', "-" * 20)
    print(len(corpus), len(corpus_drop), 'Diferència:', len(corpus) - len(corpus_drop))
    # Desa-ho en un txt
    with open("compared/compared_lenghts.txt", "w") as f:
        f.write(f"Corpus (patrons): {len(corpus)}\n")
        f.write(f"Corpus drop (patrons): {len(corpus_drop)}\n")
        f.write(f"Diferència de patrons: {len(corpus) - len(corpus_drop)}\n")
        f.write(f'Percentils de freqüència real:\n{percentils}')
        # Escriu la suma de la freqüència de corpus i corpus_drop
        f.write(f"\nSuma freqüència total del corpus: {corpus['Freqüència'].sum()}\n")
        f.write(f"Suma freqüència valors per sobre del segon percentil: {corpus_drop['Freqüència'].sum()}\n")
        # Escriu la diferència de la suma de la freqüència entre corpus i corpus_drop
        f.write(f"Suma freqüència valors residuals: {corpus['Freqüència'].sum() - corpus_drop['Freqüència'].sum()}\n")
        f.write(f"Percentatge dels valors per sobre del segon percentil: {corpus_drop['Freqüència'].sum() / corpus['Freqüència'].sum() * 100:.2f}%\n")
        f.write(f"Percentatge dels valors residuals: {(corpus['Freqüència'].sum() - corpus_drop['Freqüència'].sum()) / corpus['Freqüència'].sum() * 100:.2f}%\n")
    f.close()
    print("-" * 50)

    mitjana_corpus_drop = list(
        corpus_drop['npattern'].apply(lambda x: [0 if y == '0' else 1 for y in x]).apply(pd.Series).mean())
    print(mitjana_corpus_drop)


    # Converteix els valors de text dels Exemples del DF en valors numèrics
    #dnum = convert_df(d)
    # print(tabulate(dnum, headers='keys', tablefmt='psql'))

    #plot_stress(dnum, mitjana_real, versio='')
    #plot_stress(dnum, mitjana_corpus_drop, versio='_drop')

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
    # d_unique['Complexitat'] = d_unique['Complexitat'] / d_unique['Coincidències'].apply(lambda x: len(x.split(', ')))
    d_unique = d_unique.sort_values(by='Complexitat')
    d_unique.index = range(1, len(d_unique) + 1)
    print(tabulate(d_unique, headers='keys', tablefmt='psql'))
    # Desa el DataFrame a un fitxer Excel
    d_unique.to_excel("compared/compare_results_normalized.xlsx", index=True)

    ##############################################################################
    #results = false_positives_test(dfs, corpus, '')
    ##############################################################################
    #correlacions(d_unique, corpus, versio='')
    ##############################################################################
    # proves.descript(d_unique)
    ##############################################################################

    ##############################################################################
    #results_dos = false_positives_test(dfs, corpus_drop, '_drop')
    ##############################################################################
    #correlacions(d_unique, corpus_drop, versio='_drop')
    ##############################################################################

    # A partir de corpus, mostra la correlació entre:
    # El nombre d'autors que generen un patró i la freqüència del patró

    correlation = corpus['Nombre d\'autors'].corr(corpus['Freqüència'], method='pearson')
    print(f"Correlació entre nombre d'autors i freqüència: {correlation:.4f}")
    # Mostra la correlació amb un gràfic amb la línia de regressió i el valor de R
    plt.scatter(corpus['Nombre d\'autors'], corpus['Freqüència'])
    x_values = np.unique(corpus['Nombre d\'autors'])
    y_values = np.poly1d(np.polyfit(corpus['Nombre d\'autors'], corpus['Freqüència'], 1))(x_values)
    plt.plot(x_values, y_values, color='red')
    plt.text(0.1, 0.9, f"R= {correlation:.2f}", transform=plt.gca().transAxes)
    plt.xlabel('Nombre d\'autors')
    plt.ylabel('Freqüència')
    plt.tight_layout()
    plt.savefig("compared/correlation_authors.png", dpi=600)
    plt.close()


    ###############################################################################

    # Del df d_original conserva les columnes 'Exemple', 'Complexitat' i 'Generador'
    d_original = d_original[['Exemple', 'Complexitat', 'Generador']]
    # Per a cada patró de d_original assigna-hi la freqüència del patró en corpus
    for index, row in d_original.iterrows():
        freq = corpus[corpus['patró comparatiu'] == row['Exemple']]['Freqüència'].values
        if len(freq) == 0:
            d_original.at[index, 'Freqüència'] = 0
        else:
            d_original.at[index, 'Freqüència'] = freq[0]

    print('-' * 20, 'Dades originals amb freqüències', '-' * 20)
    print(tabulate(d_original, headers='keys', tablefmt='psql'))

    # Compta quants de patrons genera cada generador i desa-ho en un nou DataFrame
    gen_counts = d_original.groupby('Generador').size().reset_index(name='Counts')
    gen_counts.index = range(1, len(gen_counts) + 1)
    print('-' * 20, 'Counts', '-' * 20)
    print(tabulate(gen_counts, headers='keys', tablefmt='psql'))

    # Al DF d_original marca com a 'restrictius' els patrons generats pels generadors que generen menys patrons (i.e. < a la mitjana de patrons generats) i com a 'no restrictius' els altres
    mean_counts = gen_counts['Counts'].mean()
    d_original['Restrictiu'] = d_original['Generador'].apply(lambda x: 1 if gen_counts[gen_counts['Generador'] == x]['Counts'].values[0] < mean_counts else 0)
    print('-' * 20, 'Dades originals amb freqüències i restrictius', '-' * 20)
    # Ordena per freqüència
    d_original = d_original.sort_values(by='Freqüència', ascending=False)
    print(tabulate(d_original, headers='keys', tablefmt='psql'))
    # Desa-ho en un fitxer Excel
    d_original.to_excel("compared/compare_results_restrictius.xlsx", index=True)

    # Mostra per a cada generador la suma de la freqüència dels patrons generats
    gen_freq = d_original.groupby('Generador')['Freqüència'].sum().reset_index()
    gen_freq.index = range(1, len(gen_freq) + 1)
    print('-' * 20, 'Freqüència per generador', '-' * 20)
    print(tabulate(gen_freq, headers='keys', tablefmt='psql'))


    # Comprova si els generadors més restrictius generen patrons més freqüents i si els menys restrictius generen patrons menys freqüents
    correlation = d_original['Restrictiu'].corr(d_original['Freqüència'], method='pearson')
    print(f"Correlació entre restrictius i freqüència: {correlation:.4f}")
    # Mostra la correlació amb un gràfic amb la línia de regressió i el valor de R
    plt.scatter(d_original['Restrictiu'], d_original['Freqüència'])
    x_values = np.unique(d_original['Restrictiu'])
    y_values = np.poly1d(np.polyfit(d_original['Restrictiu'], d_original['Freqüència'], 1))(x_values)
    plt.plot(x_values, y_values, color='red')
    plt.text(0.1, 0.9, f"R= {correlation:.2f}", transform=plt.gca().transAxes)
    plt.xlabel('Restrictius')
    plt.ylabel('Freqüència')
    plt.tight_layout()
    plt.savefig("compared/correlation_restrictius.png", dpi=600)
    plt.close()

    # Calcula la freqüència percentual de cada patró generat
    d_original['Freqüència (%)'] = corpus['Freqüència'] / corpus['Freqüència'].sum() * 100

    dataframes = []
    for generator in d_original['Generador'].unique():
        df_gen = d_original[d_original['Generador'] == generator]
        df_gen = df_gen[['Generador', 'Exemple', 'Complexitat']]
        dataframes.append(df_gen)

    for df in dataframes:
        df['Freqüència (%)'] = corpus[corpus['patró comparatiu'] == df['Exemple']]['Freqüència'].values

    fulldf = pd.concat(dataframes, axis=0)
    fulldf = fulldf[['Generador', 'Exemple', 'Complexitat', 'Freqüència (%)']]
    fulldf = fulldf.sort_values(by='Freqüència (%)', ascending=False)
    print('-' * 20, 'Dades originals amb freqüències percentuals', '-' * 20)
    print(tabulate(fulldf, headers='keys', tablefmt='psql'))




    # Per a cada generador fes un gràfic de dispersió en què:
    # 1) L'eix Y representa la freqüència i complexitat de cada patró generat
    # 2) L'eix X conté els patrons ordenats per freqüència (ascendent).
    # 3) Cada generador té un gràfic propi
    for generator in d_original['Generador'].unique():
        df_gen = d_original[d_original['Generador'] == generator]
        x_values = df_gen['Exemple'][df_gen['Freqüència (%)'].sort_values(ascending=False).index]
        y_values = df_gen['Freqüència (%)'].sort_values(ascending=False)
        # Escalam els valors de freqüència entre 0 i 1
        y_values = (y_values - y_values.min()) / (y_values.max() - y_values.min())
        plt.plot(x_values, y_values, label='Freqüència (%)')
        # Afegeix els valors de complexitat de cada patró
        y2_values = df_gen['Complexitat'][df_gen['Freqüència (%)'].sort_values(ascending=True).index]
        # Escalam els valors de complexitat entre 0 i 1
        y2_values = (y2_values - y2_values.min()) / (y2_values.max() - y2_values.min())
        plt.plot(x_values, y2_values, label='Complexitat', linestyle='--')
        plt.xticks(rotation=90)
        plt.title(generator)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"compared/{generator}_complexity_frequency.png", dpi=600)
        plt.close()


    plt.figure(figsize=(10, 6))
    # Crea un plot amb subplots per a cada generador en dues columnes
    fig, axs = plt.subplots(2, 3, figsize=(10, 7))
    axs = axs.flatten()
    for i, generator in enumerate(d_original['Generador'].unique()):
        df_gen = d_original[d_original['Generador'] == generator]
        x_values = df_gen['Exemple'][df_gen['Freqüència (%)'].sort_values(ascending=False).index]
        y_values = df_gen['Freqüència (%)'].sort_values(ascending=False)
        y_values = (y_values - y_values.min()) / (y_values.max() - y_values.min())
        y2_values = df_gen['Complexitat'][df_gen['Freqüència (%)'].sort_values(ascending=True).index]
        y2_values = (y2_values - y2_values.min()) / (y2_values.max() - y2_values.min())
        axs[i].plot(x_values, y_values, label='Freqüència (%)')
        axs[i].plot(x_values, y2_values, linestyle='--', label='Complexitat')
        axs[i].set_xticklabels(x_values, rotation=90)
        axs[i].set_title(generator)
        #axs[i].set_xlabel('Patrons')
        #axs[i].set_ylabel('Freqüència (%) i complexitat')
        axs[i].legend()
    plt.tight_layout()
    plt.savefig("compared/complexity_frequency.png", dpi=600)
    plt.close()


    # Representa la freqüència real que acumula cada generador (d_original)
    # Ordena'ls de menor a major
    plt.figure(figsize=(10, 6))
    plt.bar(d_original['Generador'], d_original['Freqüència (%)'])
    plt.xlabel('Generador')
    plt.ylabel('Freqüència (%)')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("compared/frequency.png", dpi=600)
    plt.close()
