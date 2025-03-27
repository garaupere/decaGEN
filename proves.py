import numpy as np
import matplotlib.pyplot as plt

# Diccionari complet de patrons, generadors i complexitat segons la taula proporcionada
patrons_complexitat_generadors = [
    {"patro": "ATATATATAT", "complexitat": 0.0, "generadors": ["o1980", "o1988", "o1992b", "d2006", "o2008"]},
    {"patro": "TAATATATAT", "complexitat": 0.2, "generadors": ["o1980", "o1988", "o1992b", "d2006", "o2008"]},
    {"patro": "ATAAATATAT", "complexitat": 0.25, "generadors": ["o1980", "o1988", "o1992b", "d2006"]},
    {"patro": "AAATATATAT", "complexitat": 0.25, "generadors": ["o1980", "o1988", "o1992b", "d2006"]},
    {"patro": "ATATATAAAT", "complexitat": 0.25, "generadors": ["o1980", "o1988", "o1992b", "d2006"]},
    {"patro": "ATATATTAAT", "complexitat": 0.33, "generadors": ["o1980", "o1988", "o2008"]},
    {"patro": "ATATTAATAT", "complexitat": 0.33, "generadors": ["o1980", "o1988", "o2008"]},
    {"patro": "ATATAAATAT", "complexitat": 0.33, "generadors": ["o1980", "o1988", "o1992b"]},
    {"patro": "ATTAATATAT", "complexitat": 0.33, "generadors": ["o1980", "o1988", "o2008"]},
    {"patro": "AAAAATATAT", "complexitat": 0.5, "generadors": ["o1980", "o1988", "o1992b", "d2006"]},
    {"patro": "AATAATATAT", "complexitat": 0.5, "generadors": ["o1980", "o1988", "o1992b", "o2008"]},
    {"patro": "ATATAATAAT", "complexitat": 0.5, "generadors": ["o1980", "o1988", "o1992b", "o2008"]},
    {"patro": "TAATATAAAT", "complexitat": 0.5, "generadors": ["o1980", "o1988", "o1992b", "d2006"]},
    {"patro": "AAATATAAAT", "complexitat": 0.5, "generadors": ["o1980", "o1988", "o1992b", "d2006"]},
    {"patro": "ATAAATAAAT", "complexitat": 0.5, "generadors": ["o1980", "o1988", "o1992b", "d2006"]},
    {"patro": "TAAAATATAT", "complexitat": 0.5, "generadors": ["o1980", "o1988", "o1992b", "d2006"]},
    {"patro": "AAATAAATAT", "complexitat": 0.67, "generadors": ["o1980", "o1988", "o1992b"]},
    {"patro": "ATATAAAAAT", "complexitat": 0.67, "generadors": ["o1980", "o1988", "o1992b"]},
    {"patro": "AAAAATAAAT", "complexitat": 0.75, "generadors": ["o1980", "o1988", "o1992b", "d2006"]},
    {"patro": "TAAAATAAAT", "complexitat": 0.75, "generadors": ["o1980", "o1988", "o1992b", "d2006"]},
    {"patro": "TAATAAATAT", "complexitat": 1.0, "generadors": ["o1980", "o1992b"]},
    {"patro": "AAATTAATAT", "complexitat": 1.0, "generadors": ["o1980", "o1988"]},
    {"patro": "AAATATTAAT", "complexitat": 1.0, "generadors": ["o1980", "o1988"]},
    {"patro": "AAATAAAAAT", "complexitat": 1.0, "generadors": ["o1980", "o1988", "o1992b"]},
    {"patro": "AAATAATAAT", "complexitat": 1.0, "generadors": ["o1980", "o1988", "o1992b"]},
    {"patro": "ATAATAATAT", "complexitat": 1.0, "generadors": ["o1980", "o1988"]},
    {"patro": "ATAAAAATAT", "complexitat": 1.0, "generadors": ["o1980", "o1988"]},
    {"patro": "ATAAAAAAAT", "complexitat": 1.5, "generadors": ["o1980", "o1988"]},
    {"patro": "TAATAAAAAT", "complexitat": 1.5, "generadors": ["o1980", "o1992b"]},
    {"patro": "AAAAAAATAT", "complexitat": 1.5, "generadors": ["o1980", "o1988"]},
    {"patro": "AAAATAATAT", "complexitat": 1.5, "generadors": ["o1980", "o1988"]},
    {"patro": "AATAATAAAT", "complexitat": 1.5, "generadors": ["o1988", "o1992b"]},
    {"patro": "AAAAAAAAAT", "complexitat": 2.0, "generadors": ["o1980", "o1988"]},
    {"patro": "ATAAATTAAT", "complexitat": 2.0, "generadors": ["o1980"]},
    {"patro": "ATTAAAATAT", "complexitat": 2.0, "generadors": ["o1980"]},
    {"patro": "TAATTAATAT", "complexitat": 2.0, "generadors": ["o2008"]},
    {"patro": "ATTAATTAAT", "complexitat": 2.0, "generadors": ["o2008"]},
    {"patro": "TAATATTAAT", "complexitat": 2.0, "generadors": ["o2008"]},
    {"patro": "ATATTATAAT", "complexitat": 2.0, "generadors": ["o2008"]},
    {"patro": "ATTAATAAAT", "complexitat": 2.0, "generadors": ["o1988"]},
    {"patro": "TAATTATAAT", "complexitat": 3.0, "generadors": ["o2008"]},
    {"patro": "TATAATTAAT", "complexitat": 3.0, "generadors": ["o2008"]},
    {"patro": "TAATAATAAT", "complexitat": 3.0, "generadors": ["o2008"]},
    {"patro": "TAAAAAATAT", "complexitat": 3.0, "generadors": ["o1980"]},
    {"patro": "AATAATTAAT", "complexitat": 3.0, "generadors": ["o2008"]},
    {"patro": "AATAAAATAT", "complexitat": 3.0, "generadors": ["o1980"]},
    {"patro": "ATAAAATAAT", "complexitat": 3.0, "generadors": ["o1980"]},
    {"patro": "AAAAATTAAT", "complexitat": 3.0, "generadors": ["o1980"]},
    {"patro": "AAAAAATAAT", "complexitat": 4.0, "generadors": ["o1980"]},
    {"patro": "TAAAAAAAAT", "complexitat": 4.0, "generadors": ["o1980"]},
    {"patro": "TATAATATAT", "complexitat": 4.25, "generadors": ["o1992b", "o2008"]},
    {"patro": "TATAATAAAT", "complexitat": 9.0, "generadors": ["o1992b"]}
]

# Calcular puntuacions ajustades amb la transformació logarítmica
for p in patrons_complexitat_generadors:
    C = len(p["generadors"])
    complexitat = p["complexitat"]
    # Aplicar la transformació logarítmica a les coincidències i complexitat
    C_transformed = np.log(C + 1)  # +1 per evitar log(0)
    complexitat_transformed = np.log(complexitat + 1)  # +1 per evitar log(0)
    p["puntuacio_ajustada"] = C_transformed / (1 + complexitat_transformed)

# Calcular la suma de totes les puntuacions ajustades
suma_puntuacions_ajustades = sum(p["puntuacio_ajustada"] for p in patrons_complexitat_generadors)

# Calcular les probabilitats ajustades
for p in patrons_complexitat_generadors:
    p["probabilitat_ajustada"] = p["puntuacio_ajustada"] / suma_puntuacions_ajustades

# Nombre de simulacions
num_simulacions = 10000

# Simulació de la probabilitat d'aparició de cada patró
resultats_simulacio = np.random.choice(
    [p["patro"] for p in patrons_complexitat_generadors],
    size=num_simulacions,
    p=[p["probabilitat_ajustada"] for p in patrons_complexitat_generadors]
)

# Comptar les aparicions de cada patró en la simulació
aparicions = {p["patro"]: 0 for p in patrons_complexitat_generadors}
for patro in resultats_simulacio:
    aparicions[patro] += 1

# Calcular la probabilitat simulada per a cada patró
probabilitats_simulades = {patro: aparicions[patro] / num_simulacions for patro in aparicions}

# Mostrar els resultats
print("Probabilitats ajustades:")
for p in patrons_complexitat_generadors:
    print(f"Patró: {p['patro']}, Probabilitat ajustada: {p['probabilitat_ajustada']:.4f}")

print("\nProbabilitats simulades:")
for patro, probabilitat in probabilitats_simulades.items():
    print(f"Patró: {patro}, Probabilitat simulada: {probabilitat:.4f}")


# Visualització dels resultats
patrons = [p["patro"] for p in patrons_complexitat_generadors]
probabilitats_simulades = [probabilitats_simulades[patro] for patro in patrons]

plt.figure(figsize=(10, 6))
# Etiquetes de X són el text de patrons
plt.bar(patrons, probabilitats_simulades)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
plt.close()

# Freqüències reals dels patrons a partir de dades empíriques
freq_real = {
    "ATATATATAT": 35.88083416, "AATAATATAT": 14.20456802, "TAATATATAT": 10.39126117, "TATAATATAT": 6.716981132,
    "ATATAATAAT": 5.167825223, "TAATAATAAT": 2.387288977, "AATATATAAT": 1.982125124, "ATAATAATAT": 1.882820258,
    "ATTAATATAT": 1.620655412, "ATAATATAAT": 1.557100298, "AATTATATAT": 1.441906653, "ATATAATTAT": 1.410129096,
    "TATATATAAT": 1.374379345, "ATATATAATT": 1.279046673, "AATATAATAT": 1.203574975, "TATATAATAT": 1.024826216,
    "ATAAATATAT": 0.691161867, "AATAATAATT": 0.683217478, "TATTATATAT": 0.623634558, "TAATAATTAT": 0.512413108,
    "ATATAATATT": 0.476663357, "TATAATAATT": 0.413108242, "ATAATTATAT": 0.381330685, "ATATATAAAT": 0.309831182,
    "ATAATATATT": 0.297914598, "TAATATAATT": 0.293942403, "AATAATTAAT": 0.285998014, "AATTAATAAT": 0.278053625,
    "ATATATTAAT": 0.262164846, "ATATAAATAT": 0.250248262, "TATATATATT": 0.238331678, "ATTATATAAT": 0.2224429,
    "AATAATAAAT": 0.206554121, "AATATATATT": 0.186693148, "TAATAATATT": 0.178748759, "AAATATATAT": 0.174776564,
    "TATTAATAAT": 0.158887786, "ATATATATTT": 0.158887786, "AATATTATAT": 0.142999007, "ATATTAATAT": 0.139026812,
    "TATATTATAT": 0.135054618, "ATTATAATAT": 0.119165839, "AATATATTAT": 0.107249255, "AATTAATTAT": 0.103277061,
    "ATTTATATAT": 0.099304866, "TATAATTAAT": 0.095332671, "ATATTATAAT": 0.095332671, "ATTAATAATT": 0.091360477,
    "AATAAATAAT": 0.087388282, "TATAATAAAT": 0.079443893, "ATAATATTAT": 0.079443893, "TATATATTAT": 0.075471698,
    "TAATAAATAT": 0.063555114, "TAATATATTT": 0.05958292, "TATAAATAAT": 0.05958292, "TAATTAATAT": 0.055610725,
    "TAAAATATAT": 0.047666336, "TAATATAAAT": 0.047666336, "AATTATAATT": 0.047666336, "AAATAATAAT": 0.043694141,
    "TATTAATTAT": 0.043694141, "TAATATTAAT": 0.043694141, "ATTTAATAAT": 0.043694141, "ATTAATAAAT": 0.043694141,
    "TAAATATAAT": 0.039721946, "AATTAATATT": 0.039721946, "ATTATATATT": 0.039721946, "AATAATTATT": 0.039721946,
    "ATAAATAATT": 0.039721946, "ATATATTTAT": 0.035749752, "ATATTATATT": 0.031777557, "TATAATATTT": 0.031777557,
    "TATTATAATT": 0.027805362, "ATAATAATTT": 0.027805362, "TAATTATAAT": 0.027805362, "TAAATAATAT": 0.027805362,
    "ATTAAATAAT": 0.023833168, "AATAATATTT": 0.023833168, "ATTTAATTAT": 0.023833168, "ATTATTATAT": 0.023833168,
    "ATTATATTAT": 0.023833168, "AATATTAATT": 0.019860973, "TATATAATTT": 0.019860973, "ATATATTATT": 0.019860973,
    "TATAATTATT": 0.019860973, "ATAATTAATT": 0.019860973, "AATTAAATAT": 0.015888779, "ATAAATTAAT": 0.015888779,
    "TAAATATATT": 0.011916584, "AATTATATTT": 0.011916584, "TAATTTATAT": 0.011916584, "ATAATTAAAT": 0.011916584,
    "AAATAATATT": 0.011916584, "ATAATAAATT": 0.011916584, "ATTAATTAAT": 0.011916584, "TATTAATATT": 0.011916584,
    "AATTTAATAT": 0.011916584, "TATAATTTAT": 0.011916584, "ATTAATATTT": 0.011916584, "AATATAATTT": 0.011916584,
    "ATTTATAATT": 0.011916584, "ATATTTATAT": 0.011916584, "TAAATTATAT": 0.007944389, "AAATATTAAT": 0.007944389,
    "ATTAATTATT": 0.007944389, "AATATAAATT": 0.007944389, "AATATATTTT": 0.007944389, "AATTAATTTT": 0.007944389,
    "AATAAATTAT": 0.007944389, "ATATAAAATT": 0.007944389, "TATAAATATT": 0.007944389, "ATATTATTAT": 0.007944389,
    "ATAAAATAAT": 0.007944389, "TATATAAAAT": 0.007944389, "ATATTAATTT": 0.007944389, "AAATAATTAT": 0.007944389,
    "ATATTTAAAT": 0.003972195, "ATAATAAAAT": 0.003972195, "AATATTAAAT": 0.003972195, "AATATTTATT": 0.003972195,
    "ATTTAATATT": 0.003972195, "ATTTTATAAT": 0.003972195, "TATATATTTT": 0.003972195, "ATATTTTAAT": 0.003972195,
    "ATAAATAAAT": 0.003972195, "TAATTATTAT": 0.003972195, "TAATAAATTT": 0.003972195, "ATTTTTATAT": 0.003972195,
    "TAATTAAATT": 0.003972195, "ATATTTATTT": 0.003972195, "AATATAAAAT": 0.003972195, "ATTATTTAAT": 0.003972195,
    "AATTATAAAT": 0.003972195, "ATAAATATTT": 0.003972195, "AAATTATAAT": 0.003972195, "AATAATTTAT": 0.003972195,
    "AAATATTATT": 0.003972195, "AAATATAATT": 0.003972195, "ATTTTAATTT": 0.003972195, "ATTTATTATT": 0.003972195,
    "TATATAAATT": 0.003972195, "AATTTATAAT": 0.003972195, "TAATAATTTT": 0.003972195, "TATTTATAAT": 0.003972195,
    "AATTATTATT": 0.003972195, "ATATAATTTT": 0.003972195, "AAATAAATAT": 0.003972195, "ATTAATTTTT": 0.003972195,
    "AATTTTATAT": 0.003972195, "TAATTAATTT": 0.003972195, "TATATTTAAT": 0.003972195, "TAAATTATTT": 0.003972195,
    "ATATTTAATT": 0.003972195, "TATTATTATT": 0.003972195, "ATTAATTTAT": 0.003972195, "TAATTATATT": 0.003972195,
    "TAATATTATT": 0.003972195, "TATTATTAAT": 0.003972195, "AAATATAAAT": 0.003972195, "TATTATATTT": 0.003972195,
    "AAATTAATAT": 0.003972195, "ATAATTTTAT": 0.003972195, "TATTTAATTT": 0.003972195, "AATTTATATT": 0.003972195,
    "ATTTATTAAT": 0.003972195, "ATAATTTAAT": 0.003972195, "TATATTAATT": 0.003972195
}

freq_reals = [freq_real[patro] for patro in patrons if patro in freq_real]

#Compara la simulació amb les freqüències reals
plt.figure(figsize=(10, 6))
plt.bar(patrons, freq_reals, alpha=0.5, label='Freqüències Reals')
plt.bar(patrons, probabilitats_simulades, alpha=0.5, label='Probabilitats Simulades')
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
plt.close()

