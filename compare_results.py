"""Un mòdul per a comparar els resultats dels diferents generadors i avaluadors."""

import pandas as pd
from matplotlib import pyplot as plt
from tabulate import tabulate

import gen
import eval
from decaGEN import grammar


def get_var_name(variable):
    for name, value in globals().items():
        if value is variable:
            return name

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


    # De tot el DF mostra només els exemples únics, calcula la complexitat mitjana per exemple i ordena per complexitat
    d_unique = d.drop_duplicates(subset='Exemple').sort_values(by='Complexitat')
    d_unique.index = range(1, len(d_unique) + 1)
    print(tabulate(d_unique, headers='keys', tablefmt='psql'))

    # Desa el DataFrame a un fitxer Excel
    d_unique.to_excel("generated/compare_results.xlsx", index=True)

    # Del conjunt, calcula un valor de complexitat dividint el valor actual entre el nombre de generadors que han generat l'exemple
    d_unique['Complexitat'] = d_unique['Complexitat'] / d_unique['Coincidències'].apply(lambda x: len(x.split(', ')))
    d_unique = d_unique.sort_values(by='Complexitat')
    d_unique.index = range(1, len(d_unique) + 1)
    print(tabulate(d_unique, headers='keys', tablefmt='psql'))

    # Desa el DataFrame a un fitxer Excel
    d_unique.to_excel("generated/compare_results_normalized.xlsx", index=True)



