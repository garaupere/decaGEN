"""Possibles decasíl·labs segons Oliva"""
import pandas as pd
from pattern_parser import read_pattern as gen



def _1980_():
    """
    A.	El nombre de síl·labes —des de la primera de totes fins a l’última tònica— ha de ser 10.
    B.	El nombre de síl·labes àtones posteriors a l’última tònica pot oscil·lar de zero a dues [i esporàdicament més].
    C.	La síl·laba desena ha de portar un accent màxim.
    D.	El nombre d’accents màxims no pot ser superior a cinc.
    E.	Les síl·labes parelles poden portar accents màxims segons la fórmula següent: (2)+(4)+(6)+(8)+10. (Els números indiquen la posició de la síl·laba en la seqüència, i els parèntesis indiquen opcionalitat).
    F.	Les síl·labes imparelles poden portar accent màxim segons les tres fórmules següents:
        i.	Si la síl·laba 4 porta accent màxim, també el poden portar les síl·labes (1) + (5) + (7).
        ii.	Si la síl·laba 6 porta accent màxim, també el poden portar les síl·labes (1) + (3) + (7).
        iii.	Si les síl·labes 2 i 8 porten accent màxim, també el poden portar les síl·labes (3) + (5).
"""
    gen_patterns = ['WXWXWXWXWS', 'SWWXWXWXWS', 'WYSWWXWYWS', 'WYWXSWWYWS', 'WXWXWXSWWS']
    patterns = []
    for pattern in gen_patterns:
        result = gen(pattern)
        patterns.extend(result)

    # elimina patrons duplicats
    patterns = list(set(patterns))
    # Desa-ho en un fitxer Excel
    df = pd.DataFrame(patterns, columns=['pattern'])
    df.to_excel('../generated/oliva1980.xlsx', index= False)

    return patterns


def _1988_():
    """
    A.	La síl·laba 10 ha de ser obligatòriament un accent màxim.
    B.	Les síl·labes 2, 4, 6 i 8 poden ser T o bé A.
    C.	Les síl·labes 1, 3, 5 i 7 (la 9, fins i tot si és tònica, queda desaccentuada) són les que presenten les limitacions, que podríem resumir així:
        i.	Si la 1 és T, la 4 o la 6 també ho han de ser (no la 5, ni la 7).
        ii.	Si la 3 és T, la 6 també ho ha de ser (no la 5, ni la 7).
        iii.	Si la 5 és T, la 4 o la 8 també ho han de ser (no la 3, ni la 7).
        iv.	Si la 7 és T, la 4 també ho ha de ser (no la 3, ni la 5)
    """
    gen_patterns = ['WXWXWXWXWS', 'SWWYWYWXWS', 'WXSWWSWXWS', 'WXWYSWWYWS', 'WXWSWXSWWS']
    patterns = []
    for pattern in gen_patterns:
        result = gen(pattern)
        patterns.extend(result)

    # elimina patrons duplicats
    patterns = list(set(patterns))

    # Desa-ho en un fitxer Excel
    df = pd.DataFrame(patterns, columns=['pattern'])
    df.to_excel('../generated/oliva1988.xlsx', index= False)
    return patterns


def _1992_():
    """"""
    gen_patterns = ['WXWSWXWXWS', 'SWWSWXWXWS', 'WXWSWWSWWS', 'WXWXWSWXWS', 'SWWXWSWXWS', 'SWSWWSWXWS', 'WWXWWSWXWS', 'SWSWSSWSWS']
    patterns = []
    for pattern in gen_patterns:
        result = gen(pattern)
        patterns.extend(result)

    # elimina patrons duplicats
    patterns = list(set(patterns))
    # Desa-ho en un fitxer Excel
    df = pd.DataFrame(patterns, columns=['pattern'])
    df.to_excel('../generated/oliva1992.xlsx', index= False)
    return patterns



if __name__ == '__main__':
    o1980 = _1980_()
    print("-" * 20, "Oliva 1980", f'({len(o1980)})', "-" * 20)
    print(o1980)
    print("-" * 50)
    o1988 = _1988_()
    print("-" * 20, "Oliva 1988", f'({len(o1988)})', "-" * 20)
    print(o1988)
    print("-" * 50)
    o1992 = _1992_()
    print("-" * 20, "Oliva 1992", f'({len(o1992)})', "-" * 20)
    print(o1992)
    print("-" * 50)
