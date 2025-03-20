"""Possibles decasíl·labs segons Oliva"""
from pattern_parser import read_pattern as gen, parse_patter_hierarchy as gen2


def oliva1980():
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
    __name__ = 'Oliva 1980'
    gen_patterns = ['WXWXWXWXWS', 'SWWXWXWXWS', 'WYSWWXWYWS', 'WYWXSWWYWS', 'WXWXWXSWWS']
    patterns = []
    for pattern in gen_patterns:
        result = gen(pattern)
        patterns.extend(result)
    return list(set(patterns))


def oliva1988():
    """
    A.	La síl·laba 10 ha de ser obligatòriament un accent màxim.
    B.	Les síl·labes 2, 4, 6 i 8 poden ser T o bé A.
    C.	Les síl·labes 1, 3, 5 i 7 (la 9, fins i tot si és tònica, queda desaccentuada) són les que presenten les limitacions, que podríem resumir així:
        i.	Si la 1 és T, la 4 o la 6 també ho han de ser (no la 5, ni la 7).
        ii.	Si la 3 és T, la 6 també ho ha de ser (no la 5, ni la 7).
        iii.	Si la 5 és T, la 4 o la 8 també ho han de ser (no la 3, ni la 7).
        iv.	Si la 7 és T, la 4 també ho ha de ser (no la 3, ni la 5)
    """
    __name__ = 'Oliva 1988'
    gen_patterns = ['WXWXWXWXWS', 'SWWYWYWXWS', 'WXSWWSWXWS', 'WXWYSWWYWS', 'WXWSWXSWWS']
    patterns = []
    for pattern in gen_patterns:
        result = gen(pattern)
        patterns.extend(result)
    return list(set(patterns))


def oliva1992():
    """"""
    __name__ = 'Oliva 1992'
    gen_patterns = ['WXWSWXWXWS', 'SWWSWXWXWS', 'WXWSWWSWWS', 'WXWXWSWXWS', 'SWWXWSWXWS', 'SWSWWSWXWS', 'WWXWWSWXWS']
    patterns = []
    for pattern in gen_patterns:
        result = gen2(pattern)
        patterns.extend(result)
    return list(set(patterns))


def oliva1992b():
    """"""
    __name__ = 'Oliva 1992b'
    gen_patterns = ['WXWSWXWXWS', 'SWWSWXWXWS', 'WXWSWWSWWS', 'WXWXWSWXWS', 'SWWXWSWXWS', 'SWSWWSWXWS', 'WWXWWSWXWS']
    patterns = []
    for pattern in gen_patterns:
        result = gen(pattern)
        patterns.extend(result)
    return list(set(patterns))

def oliva2008():
    """"""
    __name__ = 'Oliva 2008'
    gen_patterns = []
    c1 = ['WSWS', 'SWWS']
    c2 = ['WSWSWS', 'WWSWWS', 'SWWSWS', 'WSSWWS', 'SWSWWS']

    for i in c1:
        for j in c2:
            gen_patterns.append(i + j)
            gen_patterns.append(j + i)

    patterns = []
    for pattern in gen_patterns:
        result = gen(pattern)
        patterns.extend(result)
    return patterns


def dols2006():
    """"""
    gen_patterns = ['WXWXWSWXWS', 'XWWXWSWXWS']
    patterns = []
    for pattern in gen_patterns:
        result = gen(pattern)
        patterns.extend(result)
    return list(set(patterns))


if __name__ == '__main__':
    o1980 = oliva1980()
    print("-" * 20, "Oliva 1980", f'({len(o1980)})', "-" * 20)
    print(o1980)
    print("-" * 50)
    o1988 = oliva1988()
    print("-" * 20, "Oliva 1988", f'({len(o1988)})', "-" * 20)
    print(o1988)
    print("-" * 50)
    o1992 = oliva1992()
    print("-" * 20, "Oliva 1992", f'({len(o1992)})', "-" * 20)
    print(o1992)
    print("-" * 50)
    o1992b = oliva1992b()
    print("-" * 20, "Oliva 1992b", f'({len(o1992b)})', "-" * 20)
    print(o1992b)
    print("-" * 50)
    d2006 = dols2006()
    print("-" * 20, "Dols 2006", f'({len(d2006)})', "-" * 20)
    print(d2006)
    print("-" * 50)
    o2008 = oliva2008()
    print("-" * 20, "Oliva 2008", f'({len(o2008)})', "-" * 20)
    print(o2008)
    print("-" * 50)
