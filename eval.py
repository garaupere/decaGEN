"""Un mòdul per a avaluar la complexitat dels patrons rítmics dels versos."""
import re

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

    regles = {'RC2': len(rc2_positions), 'RC3': len(rc3_positions), 'Complexitat': total,
              'RC2 posicions': ', '.join([str(i + 1) for i in rc2_positions]),
              'RC3 posicions': ', '.join([str(i + 1) for i in rc3_positions])}
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
        for i in range(len(exemple) - 1):
            val = int(exemple[i])
            next_val = int(exemple[i + 1])
            if next_val:
                if 0 < val < next_val < 3:
                    posicions.append(i)
        return posicions

    def radb(exemple):
        "En una configuració definida com a vall, s’afegeix un batec de manera que no es creï cap xoc accentual."
        posicions = []
        for i in range(len(exemple) - 2):
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


def oliva1992b(model, exemple):
    """"""

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


def jimenez2019(model, exemple):
    def group_feet(pattern: str) -> list:
        """
            Separa el string en peus mètrics donant prioritat a:
              - peus de 2 síl·labes,
              - després peus de 3 síl·labes,
              - després peus d'1 síl·laba,
              - després peus més llargs (>3 síl·labes) *si només contenen un '1'*.
            Cada peu ha de contenir exactament una '1' (síl·laba forta).
            """
        out = []
        i = 0
        N = len(pattern)
        priorities = [2, 3, 1]
        while i < N:
            found = False
            for l in priorities:
                if i + l <= N:
                    piece = pattern[i:i + l]
                    if piece.count('T') == 1:
                        out.append(piece)
                        i += l
                        found = True
                        break
            if not found:
                # Busca el peu més llarg possible (mínim 4) amb exactament un '1'
                for l in range(4, N - i + 1):
                    piece = pattern[i:i + l]
                    if piece.count('T') == 1:
                        out.append(piece)
                        i += l
                        found = True
                        break
            if not found:
                raise ValueError(
                    f"Impossible dividir: no es pot crear peu mètric amb '1' únic a partir de la posició {i}")
        return out

    def group_colons(patrons):
        """
        Organitza un patró mètric dividit en peus en còlons.
        Prioritza còlons de 3+2 si és possible. Els còlons han de tenir
        entre 2 i 3 peus cadascun.

        :param patrons: Llista dels peus mètrics ['AAT', 'AT', ...]
        :return: Una llista de llistes amb els còlons organitzats.
        """
        n = len(patrons)  # Número total de peus mètrics
        colons = []  # Llista per emmagatzemar els còlons
        i = 0  # Índex per recórrer els peus

        while i < n:
            restants = n - i  # Peus que queden per distribuir

            # Si queden exactament 5 peus, forcem 3+2
            if restants == 5:
                colons.append(patrons[i:i + 3])
                colons.append(patrons[i + 3:i + 5])
                break

            # Si queden 4 peus, forcem 2+2
            elif restants == 4:
                colons.append(patrons[i:i + 2])
                colons.append(patrons[i + 2:i + 4])
                break

            # Si queden 3 peus, fem un còlon de 3
            elif restants == 3:
                colons.append(patrons[i:i + 3])
                break

            # Si queden 2 peus, fem un còlon de 2
            elif restants == 2:
                colons.append(patrons[i:i + 2])
                break

            # En qualsevol altre cas, prioritzem formar un còlon de 3 peus
            else:
                colons.append(patrons[i:i + 3])
                i += 3

        return colons


    def clash(pattern):
        """Compta les violacions de *CLASH, Jiménez (2019)"""
        if pattern is not None:
            return len(re.findall('TT', pattern))
        else:
            return 0

    def extra_clash(pattern):
        """Compta les violacions de *EXTRA-CLASH, Jiménez (2019)"""
        if pattern is not None:
            return len((re.findall('TTT', pattern)))
        else:
            return 0

    def lapse(pattern):
        """Compta les violacions de *LAPSE, Jiménez (2019)"""
        if pattern is not None:
            return len(re.findall('AAA', pattern))
        else:
            return 0

    def extra_lapse(pattern):
        """Compta les violacions de *EXTRA-LAPSE, Jiménez (2019)"""
        if pattern is not None:
            return len(re.findall('AAAA', pattern))
        else:
            return 0

    def global_ws_hl(pattern):
        """Compta les violacions de *WS-HL, Jiménez (2019)"""
        if pattern is not None:
            return len(re.findall('TT$', pattern))
        else:
            return 0

    def strict_ws_hl(pattern):
        """Compta les violacions de *WS-HL, Jiménez (2019)"""
        if pattern is not None:
            return len(re.findall(r'TT\|', pattern))
        else:
            return 0

    def non_initial(pattern):
        """Compta les violacions de *NON-INITIAL, Jiménez (2019)"""
        if pattern is not None:
            return len(re.findall('^T', pattern))
        else:
            return 0

    def symetry(pattern):
        """Compta les violacions de *SYMETRY, Jiménez (2019)"""
        if pattern is not None:
            part_1 = pattern[:len(pattern) // 2]
            part_2 = pattern[len(pattern) // 2:]
            if part_1 == part_2:
                return 0
            else:
                return 1
        else:
            return 0

    def foot_bin(feet):
        """Els peus han de contenir dues síl·labes. (Violat pels peus ternaris i els peus degenerats.)"""
        cons = 0
        for foot in feet:
            if len(foot) > 2 or len(foot) < 2:
                cons += 1
        return cons

    def colon_bin(colons):
        """Els còlons han de contenir dos peus mètrics. (Violat pels còlons ternaris o pels hemistiquis amb un únic peu, i. e., amb una sola posició tònica.)"""
        penalties = 0
        for colon in colons:
            if len(colon) != 2:
                penalties += 1
        return penalties
    def verse_bin(colons):
        """Els versos han de contenir dos còlons. (Violat pels dodecasíl·labs continus, sense particions, i pels alexandrins trimembres amb tres còlons.)"""
        if len(colons) != 2:
            return 1
        else:
            return 0

    def fit(feet):
        """Les llengües seleccionen metres en què el vocabulari complet pot ser usat en una varietat més gran de formes.
        Assumim que el peu mètric prototipic del català és el troqueu (Lloret). Comptam l'ocurrència de troqueus en el patró i restam l'ocurrència de peus."""
        cons = 0
        for foot in feet:
            if foot != 'TA':
                cons += 1
        return cons

    def interest(pattern):
        """"Els paràmetres del vers es configuren de manera que es maximitze l’interès estètic del vers. Plantejat com: el vers no ha de presentar l'estructura tipificada del decasíl·lab"""
        tipic = ['ATATATATAT', 'ATATAATAAT', 'AATAATATAT']  # Segons Oliva (2008)
        if pattern in tipic:
            return 1
        else:
            return 0

    def long_last(feet):
        """"En una seqüència de grups de llargària desigual, els membres més llargs han d’anar els últims."""
        penalties = 0
        lengths = [len(foot) for foot in feet]
        for i in range(len(lengths) - 1):
            if lengths[i] > lengths[i + 1]:
                penalties += 1
        return penalties

    feet = group_feet(exemple)
    colons = group_colons(feet)

    regles = {
        "CLASH": clash(exemple),
        "EXTRA-CLASH": extra_clash(exemple),
        "LAPSE": lapse(exemple),
        "EXTRA-LAPSE": extra_lapse(exemple)#,
        #"GLOBAL-WS-HL": global_ws_hl(exemple),
        #"STRICT-WS-HL": strict_ws_hl(exemple),
        #"NON-INITIAL": non_initial(exemple),
        #"SYMETRY": symetry(exemple),
        #"FOOT-BIN": foot_bin(feet),
        #"COLON-BIN": colon_bin(colons),
        #"VERSE-BIN": verse_bin(colons),
        #"FIT": fit(feet),
        #"INTEREST": interest(exemple),
        #"LONG-LAST": long_last(feet),
    }
    total = sum(regles.values())
    regles["Complexitat"] = total
    df = pd.DataFrame(regles, index=[0])
    # Afegeix l'exemple i el model al principi del DataFrame
    df.insert(0, 'Exemple', exemple)
    df.insert(1, 'Model', model)
    return df


if __name__ == '__main__':
    model = "WSWSWSWSWS"
    #exemple = "TAATAATAAT"
    #print(oliva1980(model, exemple))
    #exemple = "0040040404"
    #print(oliva1992(model, exemple).to_string())

    print(jimenez2019('', 'ATATATATAT').to_string())
