# Copyright © 2025 Pere Garau Borràs

import pandas as pd

import gen
import eval


def grammar(model, generator, evaluator):
    print("-" * 20, generator.__name__, f'({len(generator())})', "-" * 20)
    print(generator())
    print("-" * 50)
    d = pd.DataFrame()
    for example in generator():
        d = pd.concat([d, evaluator(model, example)], axis=0)
    print(d)
    # Desa el DataFrame a un fitxer Excel
    d.to_excel(f"generated/{generator.__name__}.xlsx", index=False)
    print("-" * 50)


if __name__ == "__main__":
    model = 'WSWSWSWSWS'
    grammar(model, gen.oliva1980, eval.oliva1980)
    grammar(model, gen.oliva1988, eval.oliva1980)
    grammar(model, gen.oliva1992, eval.oliva1992)
    grammar(model, gen.dols2006, eval.oliva1992)
