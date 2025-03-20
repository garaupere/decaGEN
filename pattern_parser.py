def read_pattern(pattern):
    """
    A partir d'un patró de 10 posicions torna totes les combinacions possibles.
    - X = qualsevol síl·laba (A/T)
    - S = síl·laba forta (T)
    - W = síl·laba feble (A)
    - Y = síl·laba condicionada (si un patró conté més d'una Y, almenys una ha de ser T. No admet que totes les Y siguin A)
    """

    def generate_combinations(pat, index, current, results, y_count, t_count):
        if index == len(pat):
            if t_count > 0 or y_count == 0:
                results.append(current)
            return
        if pat[index] == 'X':
            generate_combinations(pat, index + 1, current + 'A', results, y_count, t_count)
            generate_combinations(pat, index + 1, current + 'T', results, y_count, t_count)
        elif pat[index] == 'S':
            generate_combinations(pat, index + 1, current + 'T', results, y_count, t_count)
        elif pat[index] == 'W':
            generate_combinations(pat, index + 1, current + 'A', results, y_count, t_count)
        elif pat[index] == 'Y':
            if y_count > 1:
                generate_combinations(pat, index + 1, current + 'A', results, y_count - 1, t_count)
            generate_combinations(pat, index + 1, current + 'T', results, y_count - 1, t_count + 1)

    results = []
    y_count = pattern.count('Y')
    generate_combinations(pattern, 0, '', results, y_count, 0)
    return list(set(results))


def parse_patter_hierarchy(pattern):
    """Usat per a Oliva (1992), en què el valor de T pot ser entre 1-4 (depèn del nivell del batec en la jerarquia prosòdica)."""
    def generate_combinations_2(pat, index, current, results, y_count, t_count):
        if index == len(pat):
            if t_count > 0 or y_count == 0:
                results.append(current)
            return
        if pat[index] == 'X':
            generate_combinations_2(pat, index + 1, current + '0', results, y_count, t_count)
            generate_combinations_2(pat, index + 1, current + '1', results, y_count, t_count)
            generate_combinations_2(pat, index + 1, current + '2', results, y_count, t_count)
            generate_combinations_2(pat, index + 1, current + '3', results, y_count, t_count)
        elif pat[index] == 'S':
            generate_combinations_2(pat, index + 1, current + '3', results, y_count, t_count)
        elif pat[index] == 'W':
            generate_combinations_2(pat, index + 1, current + '0', results, y_count, t_count)
        elif pat[index] == 'Y':
            if y_count > 1:
                generate_combinations_2(pat, index + 1, current + '0', results, y_count - 1, t_count)
            generate_combinations_2(pat, index + 1, current + '3', results, y_count, t_count)

    results = []
    y_count = pattern.count('Y')
    generate_combinations_2(pattern, 0, '', results, y_count, 0)
    return list(set(results))


def parse_patter_hierarchyB(pattern):
    """Usat per a Oliva (1992), en què el valor de T pot ser entre 1-4 (depèn del nivell del batec en la jerarquia prosòdica).
    Versió modificada per a treballar amb 2 nivells de T: 'T' i 't'."""
    def generate_combinations_2(pat, index, current, results, y_count, t_count):
        if index == len(pat):
            if t_count > 0 or y_count == 0:
                results.append(current)
            return
        if pat[index] == 'X':
            generate_combinations_2(pat, index + 1, current + 'A', results, y_count, t_count)
            generate_combinations_2(pat, index + 1, current + 't', results, y_count, t_count)
            generate_combinations_2(pat, index + 1, current + 'T', results, y_count, t_count)
        elif pat[index] == 'S':
            generate_combinations_2(pat, index + 1, current + 'T', results, y_count, t_count)
        elif pat[index] == 'W':
            generate_combinations_2(pat, index + 1, current + 'A', results, y_count, t_count)
        elif pat[index] == 'Y':
            if y_count > 1:
                generate_combinations_2(pat, index + 1, current + 'A', results, y_count - 1, t_count)
            generate_combinations_2(pat, index + 1, current + 'T', results, y_count, t_count)

    results = []
    y_count = pattern.count('Y')
    generate_combinations_2(pattern, 0, '', results, y_count, 0)
    return list(set(results))


if __name__ == '__main__':
    patterns = ['WXWXWXWXWS', 'SWWYWYWXWS', 'WXSWWSWXWS', 'WXWYSWWYWS', 'WXWSWXSWWS']
    for pattern in patterns:
        rp = parse_patter_hierarchy(pattern)
        print('-' * 20, pattern, f'{len(rp)}', '-' * 20)
        print(rp)  # Ha de mostrar totes les combinacions possibles
        print('-' * 50)

