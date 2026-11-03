"""Paso 1d: pasar el modelo entrenado al formato de la GBA (solo números enteros).

Cada matriz se guarda en int8 con una escala por fila (real = peso * escala / 2^24), con los pesos de cada 4 filas
juntos en cada palabra (así motor/npc_ia.c lee 4 pesos de una vez). Las normas, en Q12 (real * 4096). Los tokens van
con las letras de la GBA (charmap.txt); «@» es el nombre del jugador ({PLAYER}) y «¶», cambio de página.

Uso: python3 ia/exportar.py NOMBRE
Escribe datos/modelos/NOMBRE/modelo.bin (el que va en el cartucho) y la tabla de e^-x en motor/exp.h. Necesita el
charmap.txt de pokeemerald (las letras de la GBA), que no va aquí: en la raíz o en la variable CHARMAP.
"""
import json, math, os, re, shutil, struct, sys
import numpy as np
import torch

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # la carpeta del repositorio
CARPETA = os.path.join(RAIZ, 'datos')
CHARMAP = os.environ.get('CHARMAP', os.path.join(RAIZ, 'charmap.txt'))
ESPECIALES = {'@': [0xFD, 0x01], '¶': [0xFB], '\x01': [], '\x02': [], '<fin>': []}


def charmap():
    t = {}
    for l in open(CHARMAP, encoding='utf-8'):
        m = re.match(r"^'\\?(.)'\s*=\s*([0-9A-F]{2})\s*$", l)   # el apóstrofo va escapado: '\''
        if m and m.group(1) not in t:
            t[m.group(1)] = int(m.group(2), 16)
    return t


def matriz(w):
    w = w.detach().float().cpu().numpy()
    R, C = w.shape
    assert R % 4 == 0, (R, C)
    m = np.abs(w).max(1)
    m[m == 0] = 1e-8
    esc = np.maximum(np.round(m / 127 * 2 ** 24), 1).astype(np.int32)
    q = np.clip(np.round(w / (esc[:, None] / 2 ** 24)), -127, 127).astype(np.int8)
    return esc.tobytes() + q.reshape(R // 4, 4, C).transpose(0, 2, 1).tobytes()


def q12(v):
    return np.round(v.detach().float().cpu().numpy() * 4096).astype(np.int32).tobytes()


def main():
    nombre = sys.argv[1]
    m = torch.load(f'{CARPETA}/modelos/{nombre}/modelo.pt', map_location='cpu')
    c, w = m['conf'], m['pesos']
    tokens = json.load(open(f'{CARPETA}/tokenizador.json'))['tokens']
    tabla = charmap()
    cuerpo = matriz(w['tokens.weight']) + matriz(w['pos.weight'])
    for l in range(c['capas']):
        b = f'bloques.{l}.'
        cuerpo += q12(w[b + 'n1.peso']) + matriz(w[b + 'qkv.weight']) + matriz(w[b + 'o.weight'])
        cuerpo += q12(w[b + 'n2.peso']) + matriz(w[b + 'sube.weight']) + matriz(w[b + 'baja.weight'])
    cuerpo += q12(w['nf.peso'])
    texto, inicios = bytearray(), [0]
    for t in tokens:
        for ch in ([t] if t in ESPECIALES else t):
            if ch in ESPECIALES:
                texto += bytes(ESPECIALES[ch])
            elif ch in tabla:
                texto.append(tabla[ch])
            else:
                sys.exit(f'La letra {ch!r} (del token {t!r}) no está en charmap.txt')
        inicios.append(len(texto))
    tokens_bin = struct.pack(f'<{len(inicios)}H', *inicios) + bytes(texto)
    off = 40 + len(cuerpo)
    cab = struct.pack('<10I', 0x3143504E, c['vocab'], c['d'], c['capas'], c['cabezas'], c['oculta'], c['contexto'],
                      tokens.index('\x01'), tokens.index('\x02'), off)
    blob = cab + cuerpo + tokens_bin
    blob += bytes(-len(blob) % 4)
    salida = f'{CARPETA}/modelos/{nombre}/modelo.bin'
    open(salida, 'wb').write(blob)
    exp = [round(32768 * math.exp(-i / 16)) for i in range(256)] + [0]
    open(f'{RAIZ}/motor/exp.h', 'w').write(
        '// e^(-i/16) en Q15 (lo escribe ia/exportar.py)\n' +
        ''.join(f'{v},' + ('\n' if i % 16 == 15 else ' ') for i, v in enumerate(exp)) + '\n')
    n = sum(v.numel() for v in w.values())
    print(f'{nombre}: {n:,} parámetros -> {len(blob):,} bytes en el cartucho ({salida})'.replace(',', '.'))


if __name__ == '__main__':
    main()
