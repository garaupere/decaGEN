"""Possibles decasíl·labs segons Oliva"""
from core.pattern_parser import read_pattern as gen, parse_patter_hierarchy as gen2, parse_patter_hierarchyB as gen3


def oliva1980():
    __name__ = 'Oliva 1980'
    gen_patterns = ['WXWXWXWXWS', 'SWWXWXWXWS', 'WYSWWXWYWS', 'WYWXSWWYWS', 'WXWXWXSWWS']
    patterns = []
    for pattern in gen_patterns:
        result = gen(pattern)
        patterns.extend(result)
    return list(set(patterns))


def oliva1988():
    __name__ = 'Oliva 1988'
    gen_patterns = ['WXWXWXWXWS', 'SWWYWYWXWS', 'WXSWWSWXWS', 'WXWYSWWYWS', 'WXWSWXSWWS']
    patterns = []
    for pattern in gen_patterns:
        result = gen(pattern)
        patterns.extend(result)
    return list(set(patterns))


def oliva1992():
    __name__ = 'Oliva 1992'
    gen_patterns = ['WXWSWXWXWS', 'SWWSWXWXWS', 'WXWSWWSWWS', 'WXWXWSWXWS', 'SWWXWSWXWS', 'SWSWWSWXWS', 'WWXWWSWXWS']
    patterns = []
    for pattern in gen_patterns:
        result = gen2(pattern)
        patterns.extend(result)
    return list(set(patterns))


def oliva1992b():
    __name__ = 'Oliva 1992b'
    gen_patterns = ['WXWSWXWXWS', 'SWWSWXWXWS', 'WXWSWWSWWS', 'WXWXWSWXWS', 'SWWXWSWXWS', 'SWSWWSWXWS', 'WWXWWSWXWS']
    patterns = []
    for pattern in gen_patterns:
        result = gen3(pattern)
        patterns.extend(result)
    return list(set(patterns))

def oliva2008():
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


def dols2016():
    __name__ = 'Dols 2016'
    gen_patterns = ['WWWWWSWWWS', 'WSWSWSWSWS', 'SWWSWSWSWS', 'WWSWWSWSWS', 'WSWXWSWSWS', 'WSWSWXWSWS', 'WSWSWSWXWS', 'WSWXWSWXWS', 'SWWSWXWSWS', 'SWWSWSWXWS', 'WWSWWSWXWS', 'SWSWWSWSWS', 'SWSWWSWXWS']
    patterns = []
    for pattern in gen_patterns:
        result = gen(pattern)
        patterns.extend(result)
    return list(set(patterns))


def garau2025():
    __name__ = 'Garau 2025'
    gen_patterns = ['XXXXXXXXXS']
    patterns = []
    for pattern in gen_patterns:
        result = gen(pattern)
        patterns.extend(result)
    return list(set(patterns))

def jimenez2019():
    """"""
    __name__ = 'Jiménez 2019'
    gen_patterns = ['XXXXXXXXXS']
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
    d2006 = dols2016()
    print("-" * 20, "Dols 2016", f'({len(d2006)})', "-" * 20)
    print(d2006)
    print("-" * 50)
    o2008 = oliva2008()
    print("-" * 20, "Oliva 2008", f'({len(o2008)})', "-" * 20)
    print(o2008)
    print("-" * 50)
    j2019 = jimenez2019()
    print("-" * 20, "Jiménez 2019", f'({len(j2019)})', "-" * 20)
    print(j2019)
    print("-" * 50)


