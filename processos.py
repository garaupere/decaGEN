from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

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




if __name__ == '__main__':
    model = 'WSWSWSWSWS'
    o1980 = grammar(model, gen.oliva1980, eval.oliva1980)
    o1988 = grammar(model, gen.oliva1988, eval.oliva1980)
    o1992 = grammar(model, gen.oliva1992b, eval.oliva1980)
    d2016 = grammar(model, gen.dols2006, eval.oliva1980)
    o2008 = grammar(model, gen.oliva2008, eval.oliva1980)
    g2025 = grammar(model, gen.garau2025, eval.oliva1980)

    # Substitueix els valors de complexitat de d2006 per 0
    #d2006['Complexitat'] = 0
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
        lambda x: data_empiric[data_empiric['Patró'] == x]['Freqüència (%)'].values[0] if x in data_empiric['Patró'].values else 0)
    data_teoric.to_csv('net/data_teoric.csv', index=True, sep=';', encoding='utf-8')

    print('=' * 20, 'Corpus', '=' * 20)
    print(tabulate(data_teoric, headers='keys', tablefmt='psql'))
    print('='*50)

    # # Imprimeix la freqüència d'un patró concret generat per un generador concret
    # print(data_teoric[(data_teoric['Patró'] == 'ATATATATAT') & (data_teoric['Generador'] == 'o1980')])
    # print(data_teoric[(data_teoric['Patró'] == 'ATATATATAT') & (data_teoric['Generador'] == 'o1988')])
    # print(data_teoric[(data_teoric['Patró'] == 'ATATATATAT') & (data_teoric['Generador'] == 'o1992b')])
    # print(data_teoric[(data_teoric['Patró'] == 'ATATATATAT') & (data_teoric['Generador'] == 'o2008')])
    # print(data_teoric[(data_teoric['Patró'] == 'ATATATATAT') & (data_teoric['Generador'] == 'd2006')])

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
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
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
    axs[4].legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.savefig('net/subplots.png', dpi=600)
    plt.close()