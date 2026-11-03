"""Paso 1b: el diccionario de tokens (tokenizador BPE).

La IA no lee letra a letra: lee tokens («Pok», «émon», « CENTRO»). Con tokens, cada paso de la IA escribe unas 3
letras en vez de una, y en la GBA eso es el triple de rápido. Los tokens se sacan de los propios textos: se empieza
con las letras sueltas y se va juntando la pareja que más se repite, hasta tener VOCAB tokens (BPE, «byte pair
encoding»). El espacio va pegado delante de la palabra: « hola» es un token distinto de «hola».

Tokens especiales: 0 = fin de texto (también separa un texto del siguiente).

Uso: python3 ia/tokenizador.py [CARPETA] [VOCAB]   (por defecto datos/ y 512)
Lee corpus.txt y escribe tokenizador.json (los tokens, en orden) y tokens.bin (el corpus en tokens, uint16).
"""
import collections, json, os, re, sys
from array import array

PALABRA = re.compile(r' ?[A-Za-zÀ-ÿ]+| ?[0-9]+| ?[^A-Za-zÀ-ÿ0-9 ]| ')


def palabras(texto):
    return PALABRA.findall(texto)


def entrenar(lineas, VOCAB):
    frec = collections.Counter(p for l in lineas for p in palabras(l))
    letras = sorted({c for p in frec for c in p})
    tokens = ['<fin>'] + letras
    # cada palabra, como tupla de tokens; se juntan parejas hasta llegar a VOCAB
    partes = {p: tuple(p) for p in frec}
    while len(tokens) < VOCAB:
        parejas = collections.Counter()
        for p, n in frec.items():
            s = partes[p]
            for a, b in zip(s, s[1:]):
                parejas[a, b] += n
        if not parejas:
            break
        (a, b), _ = parejas.most_common(1)[0]
        nuevo = a + b
        tokens.append(nuevo)
        for p, s in partes.items():
            if len(s) < 2:
                continue
            r, i = [], 0
            while i < len(s):
                if i + 1 < len(s) and s[i] == a and s[i + 1] == b:
                    r.append(nuevo); i += 2
                else:
                    r.append(s[i]); i += 1
            partes[p] = tuple(r)
    return tokens


class Tokenizador:
    def __init__(self, tokens):
        self.tokens = tokens
        self.id = {t: i for i, t in enumerate(tokens)}
        self.rango = {t: i for i, t in enumerate(tokens)}   # los tokens más antiguos se juntan antes
        self.cache = {}

    @classmethod
    def cargar(cls, ruta):
        return cls(json.load(open(ruta))['tokens'])

    def _palabra(self, p):
        if p in self.cache:
            return self.cache[p]
        s = list(p)
        while len(s) > 1:
            mejor = min(((self.rango.get(s[i] + s[i + 1], 1 << 30), i) for i in range(len(s) - 1)))
            if mejor[0] == 1 << 30:
                break
            i = mejor[1]
            s[i:i + 2] = [s[i] + s[i + 1]]
        r = [self.id[t] for t in s]
        self.cache[p] = r
        return r

    def codificar(self, texto):
        return [i for p in palabras(texto) for i in self._palabra(p)]

    def decodificar(self, ids):
        return ''.join(self.tokens[i] for i in ids if i != 0)


def main():
    CARPETA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datos')
    VOCAB = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    lineas = open(f'{CARPETA}/corpus.txt').read().splitlines()
    tokens = entrenar(lineas, VOCAB)
    json.dump({'tokens': tokens}, open(f'{CARPETA}/tokenizador.json', 'w'), ensure_ascii=False, indent=0)
    tok = Tokenizador(tokens)
    ids = array('H')
    for l in lineas:
        ids.extend(tok.codificar(l))
        ids.append(0)
    ids.tofile(open(f'{CARPETA}/tokens.bin', 'wb'))
    letras = sum(map(len, lineas))
    print(f'{len(tokens)} tokens; {len(ids)} tokens en el corpus ({letras / len(ids):.2f} letras por token)')
    print('Algunos tokens largos:', sorted(tokens, key=len)[-25:])
    prueba = '¡Hola, @! ¿Vas al CENTRO POKéMON?'
    print(prueba, '->', [tokens[i] for i in tok.codificar(prueba)])


if __name__ == '__main__':
    main()
