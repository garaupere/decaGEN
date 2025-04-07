from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import gen
import eval
from decaGEN import grammar
import pandas as pd
from tabulate import tabulate


def get_var_name(variable):
    for name, value in globals().items():
        if value is variable:
            return name


def obtain_real_patterns():
    corpus = pd.read_csv("compared/corpus.csv", encoding='utf-8', sep=';', header=0)

    # Genera un patró numèric per cada patró comparatiu del corpus, A=0, T=1
    corpus['npattern'] = corpus['patró comparatiu'].apply(lambda x: ''.join(['0' if y == 'A' else '1' for y in x]))

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


def compile_corpora():
    model = 'WSWSWSWSWS'
    o1980 = grammar(model, gen.oliva1980, eval.oliva1980)
    o1988 = grammar(model, gen.oliva1988, eval.oliva1980)
    o1992 = grammar(model, gen.oliva1992b, eval.oliva1980)
    d2016 = grammar(model, gen.dols2006, eval.oliva1980)
    o2008 = grammar(model, gen.oliva2008, eval.oliva1980)
    g2025 = grammar(model, gen.garau2025, eval.oliva1980)

    # Substitueix els valors de complexitat de d2006 per 0
    # d2006['Complexitat'] = 0
    # Per a o1992b substitueix en els exemples els valors de 't' per 'T'.
    o1992['Exemple'] = o1992['Exemple'].apply(lambda x: x.replace('t', 'T'))
    o1992.drop_duplicates(subset='Exemple', inplace=True)
    o1992.index = range(1, len(o1992) + 1)
    # Inclou tots els DF en una llista
    dfs = [o1980, o1988, o1992, o2008, d2016]

    # Per a cada DF, crea una columna amb el nom del generador
    for df in dfs:
        df['Generador'] = get_var_name(df)

    # Concatena tots els DFs en un de sol
    dataframe = pd.concat(dfs, axis=0).fillna(0)
    # Conserva les columnes 'Exemple', 'Complexitat', 'RC2', 'RC3' i 'Generador'
    dataframe = dataframe[['Exemple', 'Complexitat', 'RC2', 'RC3', 'Generador']]
    print('-' * 20, 'Dades processades', '-' * 20)
    print(tabulate(dataframe, headers='keys', tablefmt='psql'))
    data_teoric = dataframe
    data_teoric.rename(columns={'Exemple': 'Patró'}, inplace=True)

    # Obre el corpus empíric
    corpus_raw = obtain_real_patterns()
    corpus = corpus_raw[0]
    mitjana_real = corpus_raw[1]
    corpus_raw = corpus_raw[2]

    data_empiric = corpus[['patró comparatiu', 'Freqüència', 'Nombre d\'autors', 'Autors']]
    # Canvia el nom de 'patró comparatiu' a 'Patró'
    data_empiric.rename(columns={'patró comparatiu': 'Patró'}, inplace=True)
    data_empiric['Freqüència (%)'] = round(data_empiric['Freqüència'] / data_empiric['Freqüència'].sum(), 4)
    data_empiric.to_csv("net/data_corpus.csv", index=True, sep=';', encoding='utf-8')

    # Per a cada patró de 'data_teoric' assigna el valors de Freqüència de 'data_empiric'
    # Si el patró no existeix a 'data_empiric' assigna 0
    data_teoric['Freqüència (%)'] = data_teoric['Patró'].apply(
        lambda x: data_empiric[data_empiric['Patró'] == x]['Freqüència (%)'].values[0] if x in data_empiric[
            'Patró'].values else 0)
    data_teoric.to_csv('net/data_teoric.csv', index=True, sep=';', encoding='utf-8')

    print('=' * 20, 'Corpus', '=' * 20)
    print(tabulate(data_teoric, headers='keys', tablefmt='psql'))
    print('=' * 50)

    # # TEST
    # print(data_teoric[(data_teoric['Patró'] == 'ATATATATAT') & (data_teoric['Generador'] == 'o1980')])
    # print(data_teoric[(data_teoric['Patró'] == 'ATATATATAT') & (data_teoric['Generador'] == 'o1988')])
    # print(data_teoric[(data_teoric['Patró'] == 'ATATATATAT') & (data_teoric['Generador'] == 'o1992b')])
    # print(data_teoric[(data_teoric['Patró'] == 'ATATATATAT') & (data_teoric['Generador'] == 'o2008')])
    # print(data_teoric[(data_teoric['Patró'] == 'ATATATATAT') & (data_teoric['Generador'] == 'd2006')])

    return data_teoric, data_empiric


def plot_freq_compl(df_teoric, df_real):
    plt.figure(figsize=(10, 6))
    for generador in data_teoric['Generador'].unique():
        data = data_teoric[data_teoric['Generador'] == generador].sort_values(by='Freqüència (%)', ascending=False)
        data.drop_duplicates(subset='Patró', inplace=True)
        x_values = np.arange(len(data))
        y_values = data['Freqüència (%)']
        y2 = plt.twinx()
        y2_values = data['Complexitat']
        plt.plot(x_values, y_values, label='Freqüència (%)', color='blue')
        y2.plot(x_values, y2_values, label='Complexitat', color='orange')
        #plt.xticks(rotation=90)
        plt.xticks(x_values, data['Patró'], rotation=90)
        plt.xlabel('Patró')
        plt.ylabel('Freqüència (%)')
        y2.set_ylabel('Complexitat')
        plt.legend()
        plt.tight_layout()
        plt.title(generador)
        plt.savefig(f'net/{generador}.png', dpi=600)
        plt.close()

    # Fes els mateixos gràfics però com a subplots d'un únic gràfic
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.flatten()
    for i, generador in enumerate(data_teoric['Generador'].unique()):
        data = data_teoric[data_teoric['Generador'] == generador].sort_values(by='Freqüència (%)', ascending=False)
        data.drop_duplicates(subset='Patró', inplace=True)
        x_values = np.arange(len(data))
        y_values = data['Freqüència (%)']
        ax2 = axs[i].twinx()
        y2_values = data['Complexitat']
        axs[i].plot(x_values, y_values, label='Freqüència (%)', color='blue')
        axs[i].plot(0, 0, label='Complexitat', color='orange')  # Dummy plot for legend
        ax2.plot(x_values, y2_values, label='Complexitat', color='orange')
        repr = data['Freqüència (%)'].sum() * 100
        axs[i].set_title(f"{generador} ({round(repr, 2)}%)")
        axs[i].set_xlabel('Patró')
        axs[i].set_ylabel('Freqüència (%)')
        ax2.set_ylabel('Complexitat')
    axs[5].set_visible(False)
    # Situa la llegenda a la dreta del penúltim plot (axs[4]) i fora del gràfic
    axs[4].legend(loc='upper left', bbox_to_anchor=(1.25, 1))
    plt.tight_layout()
    plt.savefig('net/subplots.png', dpi=600)
    plt.close()


def confusion_matrix(data_teoric, data_empiric):
    # # Del df data_teoric fes un df per cada generador amb els seus patrons
    dfs = []
    for generador in data_teoric['Generador'].unique():
        df = data_teoric[data_teoric['Generador'] == generador]
        dfs.append(df)
    # Concatena tots els DFs en un de sol
    data_teoric_gen = pd.concat(dfs, axis=0).fillna(0)

    # Genera un DF que identifica els patrons de cada generador que:
    # 1) Són generats i apareixen en les dades reals (true positives) [x] [x]
    # 2) Són generats i no apareixen en les dades reals (false positives) [x] [ ]
    # 3) No són generats i apareixen en les dades reals (false negatives) [ ] [x]
    # 4) No són generats i no apareixen en les dades reals (true negatives) [ ] [ ]

    patterns = []
    for generator in dfs:
        for index, row in generator.iterrows():
            patterns.append(row['Patró'])
    patterns = list(set(patterns))

    # Genera un DF amb els patrons reals
    real = data_empiric['Patró'].tolist()

    patrons = patterns + real
    patrons = list(set(patrons))
    results = []
    for pattern in patrons:  # Per a cada patró
        for generator in dfs:  # Per a cada generador
            if pattern in generator['Patró'].values:  # Si el patró és generat pel generador
                if pattern in real:  # Si el patró també és real
                    results.append([pattern, generator['Generador'].values[0], 'TP'])  # True Positive
                else:  # Si el patró no és real
                    results.append([pattern, generator['Generador'].values[0], 'FP'])  # False Positive
            else:  # Si el patró no és generat pel generador
                if pattern in real:  # Si el patró és real
                    results.append([pattern, generator['Generador'].values[0], 'FN'])  # False Negative

    results = pd.DataFrame(results, columns=['Patró', 'Generador', 'Resultat'])
    # Afegeix els valors de complexitat i Freqüència (%) de data_teoric_gen
    results = results.merge(data_teoric_gen[['Patró', 'Complexitat', 'Freqüència (%)']], on='Patró', how='left')

    results.index = range(1, len(results) + 1)
    results.to_excel(f"net/confusion.xlsx", index=False)
    df_results = []
    for generator in data_teoric_gen['Generador'].unique():
        tp = len(results[(results['Generador'] == generator) & (results['Resultat'] == 'TP')])
        fp = len(results[(results['Generador'] == generator) & (results['Resultat'] == 'FP')])
        fn = len(results[(results['Generador'] == generator) & (results['Resultat'] == 'FN')])
        df_results.append([generator, tp, fp, fn])

    df_results = pd.DataFrame(df_results, columns=['Generador', 'True Positive', 'False Positive', 'False Negative'])

    # Ara calcula la precisió, recall i F1 per a cada generador
    df_results['Precision'] = df_results['True Positive'] / (df_results['True Positive'] + df_results['False Positive'])
    df_results['Recall'] = df_results['True Positive'] / (df_results['True Positive'] + df_results['False Negative'])
    df_results['F1'] = 2 * (df_results['Precision'] * df_results['Recall']) / (
            df_results['Precision'] + df_results['Recall'])
    print(tabulate(df_results, headers='keys', tablefmt='psql'))
    df_results.to_csv(f"net/confusion_matrix.csv", index=False, sep=';', encoding='utf-8')

    return df_results, results


if __name__ == '__main__':
    # Obté els dataframe dels csv
    data_teoric = pd.read_csv("net/data_teoric.csv", encoding='utf-8', sep=';', header=0)
    data_empiric = pd.read_csv("net/data_corpus.csv", encoding='utf-8', sep=';', header=0)

    # Mostra la distribució de data_empiric ordena per nombre de patrons
    plt.figure(figsize=(6, 6))
    data_empiric = data_empiric.sort_values(by='Freqüència (%)', ascending=False)
    x_values = np.arange(len(data_empiric))
    y_values = data_empiric['Freqüència (%)']
    plt.bar(x_values, y_values, color='blue', alpha=0.7)
    plt.xlabel('Patró')
    plt.ylabel('Freqüència (%)')
    plt.tight_layout()
    plt.savefig('net/distribucio_freq.png', dpi=600)
    plt.close()

    #data_empiric_filtered = data_empiric.sort_values(by='Freqüència (%)', ascending=False).head(52)

    # Conserva només els patrons generats per més de 9 autors
    data_empiric_filtered = data_empiric[data_empiric['Nombre d\'autors'] > 9].sort_values(by='Freqüència (%)',
                                                                                           ascending=False)

    print('-' * 20, 'Distribució de freqüències', '-' * 20)
    print(tabulate(data_empiric_filtered, headers='keys', tablefmt='psql'))
    print(len(data_empiric_filtered))

    plt.figure(figsize=(6, 6))
    data_empiric_filtered = data_empiric_filtered.sort_values(by='Freqüència (%)', ascending=False)
    x_values = np.arange(len(data_empiric_filtered))
    y_values = data_empiric_filtered['Freqüència (%)']
    plt.bar(x_values, y_values, color='blue', alpha=0.7)
    plt.xlabel('Patró')
    plt.ylabel('Freqüència (%)')
    plt.tight_layout()
    plt.savefig('net/distribucio_freq_filtered.png', dpi=600)
    plt.close()

    cmatrix = confusion_matrix(data_teoric, data_empiric)[1]

    # Per a cada patró fes una llista dels generadors que el generen (= TP)

    cmatrix['Generadors'] = cmatrix.apply(
        lambda x: ', '.join(
            cmatrix[(cmatrix['Patró'] == x['Patró']) & (cmatrix['Resultat'] == 'TP')]['Generador'].unique()), axis=1)

    cmatrix.drop(columns='Generador', inplace=True)
    cmatrix.drop_duplicates(subset='Patró', inplace=True)
    cmatrix.index = range(1, len(cmatrix) + 1)
    # Afegeix una columna amb el recompte de generadors
    cmatrix["Nombre de generadors"] = cmatrix['Generadors'].apply(lambda x: len(x.split(', ')))
    print('-' * 20, 'Patrons generats', '-' * 20)
    print(tabulate(cmatrix, headers='keys', tablefmt='psql'))
    cmatrix.to_csv("net/confusion.csv", encoding='utf-8', sep=';', index=True, header=True)

    print('=' * 20, 'True Positives', '=' * 20)
    print(tabulate(cmatrix[cmatrix['Resultat'] == 'TP'], headers='keys', tablefmt='psql'))

    # Examina la correlació entre la freqüència dels TP i el nombre de generadors que els generen

    correlation = cmatrix[cmatrix['Resultat'] == 'TP'][['Freqüència (%)', 'Nombre de generadors']].corr().iloc[0, 1]
    print(f"Correlació entre la freqüència dels TP i el nombre de generadors que els generen: {correlation}")

    plt.scatter(cmatrix[cmatrix['Resultat'] == 'TP']['Freqüència (%)'],
                cmatrix[cmatrix['Resultat'] == 'TP']['Nombre de generadors'], alpha=0.7)
    plt.xlabel('Freqüència (%)')
    plt.ylabel('Nombre de generadors')
    plt.savefig('net/correlacio_TP-Gen.png', dpi=600)
    plt.close()

    # Agafa els 23 patrons més freqüents del corpus i compta quins generadors els generen
    # i quins no

    mes_fq = data_empiric.sort_values(by='Freqüència (%)', ascending=False).head(23)
    # Fes un dataframe amb els patrons i els generadors que els generen
    # i els que no
    # Indica en cada cas quins generadors generen el patró
    df = []
    for index, row in mes_fq.iterrows():
        # Si el patró és generat pel generador
        if row['Patró'] in cmatrix['Patró'].values:
            generadors = cmatrix[cmatrix['Patró'] == row['Patró']]['Generadors'].values[0]
            df.append([row['Patró'], row['Freqüència (%)'], row['Nombre d\'autors'], generadors])
        else:
            df.append([row['Patró'], row['Freqüència (%)'], row['Nombre d\'autors'], 'No generat'])
    df = pd.DataFrame(df, columns=['Patró', 'Freqüència (%)', 'Nombre d\'autors', 'Generadors'])
    df.index = range(1, len(df) + 1)
    # Afegeix una columna amb el recompte de generadors
    df["Nombre de generadors"] = df['Generadors'].apply(lambda x: len(x.split(', ')) if x != 'No generat' else 0)

    print('-' * 20, 'Patrons més freqüents', '-' * 20)
    print(tabulate(df, headers='keys', tablefmt='psql'))
    df.to_csv("net/patrons_mes_fq.csv", encoding='utf-8', sep=';', index=True, header=True)

    ###############################################################################
    # En un nou DataFrame, converteix cada patró en una llista dels índexs de les síl·labes tòniques (T).
    # Desa-hi també la freqüència de cada patró i el nombre d'autors

    df_index = []
    for index, row in data_empiric.iterrows():
        # Converteix el patró en una llista dels índexs de les síl·labes tòniques (T)
        pattern = row['Patró']
        indexes = [i+1 for i, x in enumerate(pattern) if x == 'T']
        df_index.append([pattern, row['Freqüència (%)'], row['Nombre d\'autors'], indexes])
    df_index = pd.DataFrame(df_index, columns=['Patró', 'Freqüència (%)', 'Nombre d\'autors', 'Índexs'])
    df_index.index = range(1, len(df_index) + 1)
    df_index["Nombre d'índexs"] = df_index['Índexs'].apply(lambda x: len(x))
    print('-' * 20, 'Patrons amb índexs', '-' * 20)
    print(tabulate(df_index, headers='keys', tablefmt='psql'))


    # Fes un scatterplot de la freqüència (%) i els índexs tònics
    # L'eix Y són els valors dels índexs (1-10)
    # L'eix X és la freqüència (%)
    # Fes que cada patró sigui d'un color
    # Selecciona els 5 patrons més freqüents
    mes_fq = df_index.sort_values(by='Freqüència (%)', ascending=False).head(5)
    plt.figure(figsize=(6, 6))
    for index, row in df_index.iterrows():
        indexes = row['Índexs']
        freq = row['Freqüència (%)']
        if index in mes_fq.index:
            # Si el patró és un dels 5 més freqüents
            plt.scatter([freq] * len(indexes), indexes, alpha=0.7, label=row['Índexs'])
        else:
            plt.scatter([freq] * len(indexes), indexes, alpha=0.7, label=None)
    plt.xlabel('Freqüència (%)')
    plt.ylabel('Índexs tònics')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.savefig('net/freq_index_patro.png', dpi=600, bbox_inches='tight')
    plt.close()

    # Desa-ho en un fitxer Excel
    df_index.to_excel("net/patrons_indexs.xlsx", index=True)


    # Fes un DataFrame (df_fq_indexs) amb una fila per a cada índex (1-10) i una columna (Freqüència (%)) que contengui la suma de freqüències que registra cada índex.
    # Exemple:
    # Índex 1: 0.25 + 0.15 + 0.10 = 0.50
    # Índex 2: 0.25 + 0.15 = 0.40
    # etc.
    df_fq_indexs = []
    for i in range(1, 11):
        freq = df_index[df_index['Índexs'].apply(lambda x: i in x)]['Freqüència (%)'].sum()
        df_fq_indexs.append([i, freq])
    df_fq_indexs = pd.DataFrame(df_fq_indexs, columns=['Índex', 'Freqüència (%)'])
    df_fq_indexs.index = range(1, len(df_fq_indexs) + 1)
    print('-' * 20, 'Freqüència (%) per índex', '-' * 20)
    print(tabulate(df_fq_indexs, headers='keys', tablefmt='psql'))

    # Ara fes un plot de la freqüència per índex
    plt.figure(figsize=(6, 6))
    x_values = np.arange(len(df_fq_indexs))
    y_values = df_fq_indexs['Freqüència (%)']
    plt.bar(x_values, y_values, color='blue', alpha=0.7)
    plt.xlabel('Índex')
    plt.ylabel('Freqüència (%)')
    plt.xticks(x_values, df_fq_indexs['Índex'], rotation=0)
    plt.tight_layout()
    plt.savefig('net/freq_index.png', dpi=600)
    plt.close()

    # Ara fes un scatter plot de la freqüència per índex
    plt.figure(figsize=(6, 6))
    plt.scatter(df_fq_indexs['Índex'], df_fq_indexs['Freqüència (%)'], linestyle='-', marker='o', color='blue', alpha=0.7)
    plt.xlabel('Índex')
    plt.ylabel('Freqüència (%)')
    plt.xticks(df_fq_indexs['Índex'], rotation=0)
    plt.tight_layout()
    plt.savefig('net/freq_index_scatter.png', dpi=600)
    plt.close()
