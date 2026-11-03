"""Paso 1e: comprobar en el PC que el motor de enteros (motor/npc_ia.c) piensa igual que el modelo de Python.

Compila prueba_pc.c, le pasa textos que el modelo no ha visto al aprender y compara, token a token, lo que predicen
los dos: cuántas veces coinciden en el token más probable y el error medio de cada uno (cuanto más bajo, mejor). Luego
saca unas frases con el motor de enteros, como las diría la GBA.

Uso: python3 ia/comprobar.py NOMBRE [N_TEXTOS]
"""
import os, subprocess, sys
import numpy as np
import torch
import torch.nn.functional as F
from entrenar import CARPETA, Modelo, legible
from tokenizador import Tokenizador

AQUI = os.path.dirname(os.path.abspath(__file__))


def main():
    nombre = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    exe = f'{CARPETA}/prueba_pc'
    subprocess.run(['gcc', '-O2', '-iquote', f'{AQUI}/../motor', '-o', exe, f'{AQUI}/prueba_pc.c'], check=True)
    binario = f'{CARPETA}/modelos/{nombre}/modelo.bin'
    tok = Tokenizador.cargar(f'{CARPETA}/tokenizador.json')
    g = torch.load(f'{CARPETA}/modelos/{nombre}/modelo.pt', map_location='cpu')
    m = Modelo(**g['conf']); m.load_state_dict(g['pesos']); m.eval()
    datos = np.fromfile(f'{CARPETA}/tokens.bin', dtype=np.uint16).astype(np.int64)
    textos = [t for t in np.split(datos, np.flatnonzero(datos == 0) + 1) if len(t)]
    prueba = [np.concatenate([[0], t])[:128] for i, t in enumerate(textos) if i % 20 == 0]
    # la mitad de Pokémon y la otra mitad de Backrooms
    pk = [t for t in prueba if t[1] == tok.id['\x01']][: n // 2]
    br = [t for t in prueba if t[1] == tok.id['\x02']][: n - n // 2]
    prueba = pk + br
    entrada = ''.join(' '.join(map(str, t)) + '\n' for t in prueba)
    salida = subprocess.run([exe, binario, 'evaluar'], input=entrada, capture_output=True, text=True, check=True).stdout
    bloques = salida.split('---\n')
    coinciden = total = 0
    err_f = err_e = 0.0
    for t, b in zip(prueba, bloques):
        enteros = torch.tensor([[int(v) for v in l.split()] for l in b.strip().splitlines()], dtype=torch.float32) / 4096
        with torch.no_grad():
            flot = m(torch.tensor(t)[None])[0]
        y = torch.tensor(t[1:])
        k = len(y)
        coinciden += (enteros[:k].argmax(-1) == flot[:k].argmax(-1)).sum().item()
        total += k
        err_f += F.cross_entropy(flot[:k], y, reduction='sum').item()
        err_e += F.cross_entropy(enteros[:k], y, reduction='sum').item()
    print(f'{len(prueba)} textos, {total} tokens:')
    print(f'  el token más probable coincide en el {100 * coinciden / total:.1f} %')
    print(f'  error medio: decimales (Python) {err_f / total:.3f}; enteros (GBA) {err_e / total:.3f}')
    for tema in ('pokemon', 'backrooms'):
        r = subprocess.run([exe, binario, 'generar', tema, '5', '7'], capture_output=True, text=True, check=True)
        for l in r.stdout.splitlines():
            print(f'  [{tema}] ' + legible(tok.decodificar([int(v) for v in l.split()])))


if __name__ == '__main__':
    main()
