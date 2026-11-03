"""Paso 1a: los textos con los que aprende la IA de los NPC (corpus).

Junta los diálogos en español de Esmeralda y Rojo Fuego y las entradas de la Pokédex (no van aquí: ver el README; sin
ellos, solo con los propios) y los deja limpios, uno por línea:
- sin códigos de control ({PAUSE 15}, {FONT 0}…); los textos con huecos que rellena el juego ({STR_VAR_1}) se quitan;
- {PLAYER} pasa a ser «@», para que el NPC pueda decir el nombre del jugador;
- el cambio de página (\\p) pasa a ser «¶»; los saltos de línea (\\n, \\l), un espacio: en la GBA se vuelve a partir
  el texto según quepa en el cuadro.

Cada texto empieza con su tema: «\\x01» Pokémon (los del juego) o «\\x02» Backrooms (los propios, escritos para el
canal, en textos/*.txt, uno por línea, con «¶» y «@»). En la GBA, el tema con el que se
empieza decide de qué habla el NPC, y se puede cambiar a mitad de frase.

Uso: python3 ia/corpus.py [CARPETA_SALIDA]   (por defecto datos/). Escribe corpus.txt y un resumen.
"""
import glob, json, os, re, sys

POKEMON, BACKROOMS = '\x01', '\x02'
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # la carpeta del repositorio
PROPIOS = os.path.join(RAIZ, 'textos')
TRAD = os.environ.get('TEXTOS_POKEMON', os.path.join(RAIZ, 'pokemon'))   # los de los juegos: no van aquí
SALIDA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RAIZ, 'datos')

FUERA = re.compile(r'\{(PAUSE[^}]*|KUN|CLEAR[^}]*|FONT[^}]*|COLOR[^}]*|SHADOW[^}]*|PLAY_BGM[^}]*|PAUSE_MUSIC|'
                   r'RESUME_MUSIC|DYNAMIC)\}')
CAMBIOS = {'{PLAYER}': '@', '{RIVAL}': 'AURA', '{POKEBLOCK}': 'POKéCUBO'}


def limpiar(t):
    t = FUERA.sub('', t)
    for a, b in CAMBIOS.items():
        t = t.replace(a, b)
    if '{' in t or '}' in t:          # huecos del juego (STR_VAR…) o símbolos raros: fuera
        return None
    t = t.replace('\\p', '¶').replace('\\n', ' ').replace('\\l', ' ')
    t = t.replace('’', "'")
    t = re.sub(r' *¶ *', '¶', t)
    t = re.sub(r' {2,}', ' ', t).strip(' ¶')
    if len(t) < 12 or ' ' not in t:   # nombres sueltos («CUEVA», «AZURILL»): no son frases
        return None
    return t


def main():
    os.makedirs(SALIDA, exist_ok=True)
    fuentes = {
        'esmeralda': json.load(open(f'{TRAD}/esmeralda_es.json')).values(),
        'rojofuego': json.load(open(f'{TRAD}/rojofuego_es.json')).values(),
        'pokedex': [e['descripcion'].replace('Pokémon', 'POKéMON')
                    for e in json.load(open(f'{TRAD}/pokeapi_es.json'))['especies'].values() if e.get('descripcion')],
    } if os.path.isdir(TRAD) else {}                  # sin los de los juegos, solo con los propios
    propios = [l.strip() for f in sorted(glob.glob(f'{PROPIOS}/*.txt')) for l in open(f, encoding='utf-8')]
    vistos, lineas, resumen = set(), [], [] if fuentes else [f'sin los textos de los juegos ({TRAD})']
    for nombre, textos in fuentes.items():
        n = 0
        for t in textos:
            t = limpiar(t)
            if t and t not in vistos:
                vistos.add(t)
                lineas.append(POKEMON + t)
                n += 1
        resumen.append(f'{nombre}: {n} textos')
    permitidas = set(''.join(lineas))
    n = 0
    for t in propios:
        if t and t not in vistos:
            raras = set(t) - permitidas
            if raras and permitidas:
                resumen.append(f'AVISO: letras que no están en los textos del juego {sorted(raras)}: {t[:60]}')
            vistos.add(t)
            lineas.append(BACKROOMS + t)
            n += 1
    resumen.append(f'backrooms (propios): {n} textos')
    with open(f'{SALIDA}/corpus.txt', 'w') as f:
        f.write('\n'.join(lineas) + '\n')
    letras = sorted(set(''.join(lineas)) - {POKEMON, BACKROOMS})
    resumen += [f'total: {len(lineas)} textos, {sum(map(len, lineas))} caracteres',
                f'{len(letras)} caracteres distintos: ' + ''.join(letras)]
    open(f'{SALIDA}/corpus_resumen.txt', 'w').write('\n'.join(resumen) + '\n')
    print('\n'.join(resumen))


if __name__ == '__main__':
    main()
