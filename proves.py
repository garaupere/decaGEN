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
    "ATATATATAT": 37.91, "TAATATATAT": 11.14, "ATAAATATAT": 0.14, "AAATATATAT": 0.79,
    "ATATATAAAT": 0.34, "ATATATTAAT": 0.00, "ATATTAATAT": 0.001, "ATATAAATAT": 0.001,
    "ATTAATATAT": 8.74, "AAAAATATAT": 5.51, "AATAATATAT": 0.16, "ATATAATAAT": 0.02,
    "TAATATAAAT": 0.00, "AAATATAAAT": 0.001, "ATAAATAAAT": 1.78, "TAAAATATAT": 1.03,
    "AAATAAATAT": 0.86, "ATATAAAAAT": 0.45, "AAAAATAAAT": 0.001, "TAAAATAAAT": 0.001,
    "TAATAAATAT": 0.00, "AAATTAATAT": 0.04, "AAATATTAAT": 0.00, "AAATAAAAAT": 0.17,
    "AAATAATAAT": 0.01, "ATAATAATAT": 0.01, "ATAAAAATAT": 1.14, "ATAAAAAAAT": 0.00,
    "TAATAAAAAT": 0.00, "AAAAAAATAT": 0.00, "AAAATAATAT": 0.00, "AATAATAAAT": 0.27,
    "AAAAAAAAAT": 0.00, "ATAAATTAAT": 4.32, "ATTAAAATAT": 0.07, "TAATTAATAT": 0.32,
    "ATTAATTAAT": 0.00, "TAATATTAAT": 0.44, "ATATTATAAT": 0.48, "ATTAATAAAT": 0.03,
    "TAATTATAAT": 0.02, "TATAATTAAT": 0.59, "TAATAATAAT": 0.17, "TAAAAAATAT": 0.001,
    "AATAATTAAT": 0.00, "AATAAAATAT": 0.00, "ATAAAATAAT": 2.43, "AAAAATTAAT": 0.29,
    "AAAAAATAAT": 0.00, "TAAAAAAAAT": 0.00, "TATAATATAT": 0.00, "TATAATAAAT": 0.15
}



