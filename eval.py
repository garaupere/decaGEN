"""Un mòdul per a avaluar la complexitat dels patrons rítmics dels versos."""

import pandas as pd


def oliva1980(model, exemple):
    """
        RC1: S→T; W→A
        RC2: S→A/_WS
        RC3: WS→TA/_WS
            (Oliva 1980:76)
    """

    def aplicar_rc3(model, exemple):
        posicions = []
        for i in range(len(model) - 1):
            if model[i:i + 2] == "WS" and exemple[i:i + 2] == "TA":
                posicions.append(i+1)

        return posicions

    def aplicar_rc2(model, exemple):
        posicions = []
        for i in range(len(model) - 1):
            if model[i:i + 2] == "WS" and exemple[i:i + 2] == "AA":
                posicions.append(i+1)
        return posicions

    rc3_positions = aplicar_rc3(model, exemple)
    rc2_positions = aplicar_rc2(model, exemple)

    total = len(rc3_positions) + len(rc2_positions)

    def print_rc(total):
        rules_p = [''] * 10
        rules_p.append(total)
        for position in rc2_positions:
            rules_p[position] = 'RC2'
        for position in rc3_positions:
            rules_p[position] = 'RC3'
        table = [list(model) + [''], list(exemple.replace('W', 'A').replace('S', 'T')) + [''], rules_p]
        taula = pd.DataFrame(table)
        taula.index = ['Model', 'Exemple', 'Regles']
        taula.rename(columns={10: 'Complexitat'}, inplace=True)
        taula.rename(columns={i: i + 1 for i in range(10)}, inplace=True)
        print(taula)
    #print_rc(total)

    regles = {'RC2': len(rc2_positions), 'RC3': len(rc3_positions), 'Complexitat': total, 'RC2 posicions': ', '.join([str(i + 1) for i in rc2_positions]), 'RC3 posicions': ', '.join([str(i + 1) for i in rc3_positions])}
    df = pd.DataFrame(regles, index=[0])
    # Afegeix l'exemple i el model al principi del DataFrame
    df.insert(0, 'Exemple', exemple)
    df.insert(1, 'Model', model)
    return df



def oliva1992(model, exemple):
    """"""
    def rm1(exemple):
        posicions = []
        for i in range(len(exemple) - 1):
            if exemple[i:i + 2] == "TA":
                posicions.append(i+1)
        return posicions

    def rm2(exemple):
        posicions = []
        for i in range(len(exemple) - 1):
            if exemple[i:i + 3] == "AAT":
                posicions.append(i+1)
        return posicions

    aplicar_rm1 = rm1(exemple)
    aplicar_rm2 = rm2(exemple)
    total = len(aplicar_rm1) + len(aplicar_rm2)
    regles = {
        "RM1": len(aplicar_rm1),
        "RM2": len(aplicar_rm2),
        "Complexitat": total,
        "RC2 posicions": ', '.join([str(i + 1) for i in aplicar_rm1]),
        "RC3 posicions": ', '.join([str(i + 1) for i in aplicar_rm2])
        }
    df = pd.DataFrame(regles, index=[0])
    # Afegeix l'exemple i el model al principi del DataFrame
    df.insert(0, 'Exemple', exemple)
    df.insert(1, 'Model', model)
    return df


if __name__ == '__main__':
    model = "WSWSWSWSWS"
    exemple = "TAATAATAAT"
    print(oliva1980(model, exemple))

    print(oliva1992(model, exemple))
