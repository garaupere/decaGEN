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
                posicions.append(i)

        return posicions

    def aplicar_rc2(model, exemple):
        posicions = []
        for i in range(len(model) - 1):
            if model[i:i + 2] == "WS" and exemple[i:i + 2] == "AA":
                posicions.append(i)
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
        "Inversió iàmbica."
        posicions = []
        for i in range(0, len(exemple), 2):
            if exemple[i:i + 2] in ['30', '20', '10']:
                posicions.append(i)
        return posicions

    def rm2(exemple):
        "Binarització del còlon."
        posicions = []
        for i in range(0, len(exemple), 3):
            if exemple[i:i + 3] in ['003', '002', '001']:
                posicions.append(i)
        return posicions

    def rab(exemple):
        "Traslladeu l’accent cap a la dreta, des del segon batec per amunt, fins que trobeu un altre accent que pugui absorbir tots els batecs que són traslladats."
        posicions = []
        for i in range(len(exemple)-1):
            val = int(exemple[i])
            next_val = int(exemple[i + 1])
            if next_val:
                if 0 < val < next_val < 3:
                    posicions.append(i)
        return posicions

    def radb(exemple):
        "En una configuració definida com a vall, s’afegeix un batec de manera que no es creï cap xoc accentual."
        posicions = []
        for i in range(len(exemple)-2):
            val = list(str(exemple[i:i + 3]))
            val1 = int(val[0])
            val2 = int(val[1])
            val3 = int(val[2])
            if 3 > val2 > 0 and val3 < 1 and val1 < 1:
                posicions.append(i)
        return posicions

    def rc1(exemple):
        "Cada posició terminal W requereix una síl·laba d’un batec, com a mínim."
        W_positions = [0, 2, 4, 6, 8]
        positions = []
        exemple = list(exemple)
        for i in W_positions:
            if int(exemple[i]) > 0:
                positions.append(i)
        return positions

    def rc2(exemple):
        "Cada posició terminal S dominada per W requereix una síl·laba de dos batecs, com a mínim."
        exemple = list(exemple)
        S_domWpositions = [1, 3, 5, 7]
        positions = []
        for i in S_domWpositions:
            if int(exemple[i]) < 2:
                positions.append(i)
        return positions

    def rc3(exemple):
        "Cada posició terminal S dominada per S requereix una síl·laba de quatre batecs, com a mínim. (Si en té tres, el vers adquireix tensió; però la majoria dels poetes no eliminen del tot aquesta possibilitat. La posició S dominada per S i ocupada per una síl·laba de menys de quatre batecs és típica en els versos discontinus.)"
        exemple = list(exemple)
        S_domSpositions = [3, 5, 9]
        positions = []
        for i in S_domSpositions:
            if int(exemple[i]) < 3:
                positions.append(i)
        return positions

    aplicar_rm1 = rm1(exemple)
    aplicar_rm2 = rm2(exemple)
    aplicar_rab = rab(exemple)
    aplicar_radb = radb(exemple)
    aplicar_rc1 = rc1(exemple)
    aplicar_rc2 = rc2(exemple)
    aplicar_rc3 = rc3(exemple)
    total = len(aplicar_rm1) + len(aplicar_rm2) + len(aplicar_rab) + len(aplicar_radb) + len(aplicar_rc1) + len(aplicar_rc2) + len(aplicar_rc3)
    regles = {
        "RM1": len(aplicar_rm1),
        "RM2": len(aplicar_rm2),
        "RAB": len(aplicar_rab),
        "RADB": len(aplicar_radb),
        "RC1": len(aplicar_rc1),
        "RC2": len(aplicar_rc2),
        "RC3": len(aplicar_rc3),
        "Complexitat": total,
        "RM1 posicions": ', '.join([str(i + 1) for i in aplicar_rm1]),
        "RM2 posicions": ', '.join([str(i + 1) for i in aplicar_rm2]),
        "RAB posicions": ', '.join([str(i + 1) for i in aplicar_rab]),
        "RADB posicions": ', '.join([str(i + 1) for i in aplicar_radb]),
        "RC1 posicions": ', '.join([str(i + 1) for i in aplicar_rc1]),
        "RC2 posicions": ', '.join([str(i + 1) for i in aplicar_rc2]),
        "RC3 posicions": ', '.join([str(i + 1) for i in aplicar_rc3])
        }
    df = pd.DataFrame(regles, index=[0])
    # Afegeix l'exemple i el model al principi del DataFrame
    df.insert(0, 'Exemple', exemple)
    df.insert(1, 'Model', model)
    return df


def oliva1992b(model, exemple):
    def rm1(exemple):
        "Inversió iàmbica."
        posicions = []
        for i in range(0, len(exemple), 2):
            if exemple[i:i + 2] in ['TA', 'tA']:
                posicions.append(i)
        return posicions

    def rm2(exemple):
        "Binarització del còlon."
        posicions = []
        for i in range(0, len(exemple), 3):
            if exemple[i:i + 3] in ['AAT', 'AAt']:
                posicions.append(i)
        return posicions

    def rab(exemple):
        "Traslladeu l’accent cap a la dreta, des del segon batec per amunt, fins que trobeu un altre accent que pugui absorbir tots els batecs que són traslladats."
        posicions = []
        for i in range(len(exemple) - 1):
            val = exemple[i]
            next_val = exemple[i + 1]
            if next_val:
                if val == 't' and next_val == 'T':
                    posicions.append(i)
        return posicions

    def radb(exemple):
        "En una configuració definida com a vall, s’afegeix un batec de manera que no es creï cap xoc accentual."
        posicions = []
        for i in range(len(exemple) - 2):
            val = list(str(exemple[i:i + 3]))
            val1 = val[0]
            val2 = val[1]
            val3 = val[2]
            if val1 == 'A' and val2 == 't' and val3 == 'A':
                posicions.append(i)
        return posicions

    def rc1(exemple):
        "Cada posició terminal W requereix una síl·laba d’un batec, com a mínim."
        W_positions = [0, 2, 4, 6, 8]
        positions = []
        exemple = list(exemple)
        for i in W_positions:
            if exemple[i] in ['T', 't']:
                positions.append(i)
        return positions

    def rc2(exemple):
        "Cada posició terminal S dominada per W requereix una síl·laba de dos batecs, com a mínim."
        exemple = list(exemple)
        S_domWpositions = [1, 3, 5, 7]
        positions = []
        for i in S_domWpositions:
            if exemple[i] == 'A':
                positions.append(i)
        return positions

    def rc3(exemple):
        "Cada posició terminal S dominada per S requereix una síl·laba de quatre batecs, com a mínim. (Si en té tres, el vers adquireix tensió; però la majoria dels poetes no eliminen del tot aquesta possibilitat. La posició S dominada per S i ocupada per una síl·laba de menys de quatre batecs és típica en els versos discontinus.)"
        exemple = list(exemple)
        S_domSpositions = [3, 5, 9]
        positions = []
        for i in S_domSpositions:
            if exemple[i] in ['A', 't']:
                positions.append(i)
        return positions

    aplicar_rm1 = rm1(exemple)
    aplicar_rm2 = rm2(exemple)
    aplicar_rab = rab(exemple)
    aplicar_radb = radb(exemple)
    aplicar_rc1 = rc1(exemple)
    aplicar_rc2 = rc2(exemple)
    aplicar_rc3 = rc3(exemple)
    total = len(aplicar_rm1) + len(aplicar_rm2) + len(aplicar_rab) + len(aplicar_radb) + len(aplicar_rc1) + len(
        aplicar_rc2) + len(aplicar_rc3)
    regles = {
        "RM1": len(aplicar_rm1),
        "RM2": len(aplicar_rm2),
        "RAB": len(aplicar_rab),
        "RADB": len(aplicar_radb),
        "RC1": len(aplicar_rc1),
        "RC2": len(aplicar_rc2),
        "RC3": len(aplicar_rc3),
        "Complexitat": total,
        "RM1 posicions": ', '.join([str(i + 1) for i in aplicar_rm1]),
        "RM2 posicions": ', '.join([str(i + 1) for i in aplicar_rm2]),
        "RAB posicions": ', '.join([str(i + 1) for i in aplicar_rab]),
        "RADB posicions": ', '.join([str(i + 1) for i in aplicar_radb]),
        "RC1 posicions": ', '.join([str(i + 1) for i in aplicar_rc1]),
        "RC2 posicions": ', '.join([str(i + 1) for i in aplicar_rc2]),
        "RC3 posicions": ', '.join([str(i + 1) for i in aplicar_rc3])
    }
    df = pd.DataFrame(regles, index=[0])
    # Afegeix l'exemple i el model al principi del DataFrame
    df.insert(0, 'Exemple', exemple)
    df.insert(1, 'Model', model)
    return df



if __name__ == '__main__':
    model = "WSWSWSWSWS"
    #exemple = "TAATAATAAT"
    #print(oliva1980(model, exemple))
    exemple = "0040040404"
    print(oliva1992(model, exemple).to_string())
