"""Paso 1c: entrenar la IA de los NPC en la gráfica.

Es un transformador diminuto (la misma familia que ChatGPT), pensado para que luego se pueda hacer en la GBA con
números enteros:
- posiciones aprendidas (una tabla que se suma), en vez de senos y cosenos;
- capa de «pensar» con ReLU (deja los positivos y pone a 0 los negativos), que en la GBA es una comparación;
- normalización RMS (dividir por el tamaño medio), una raíz cuadrada por capa;
- la tabla de entrada de los tokens se reutiliza a la salida (ahorra cartucho).

Durante el entrenamiento escribe cada cierto tiempo frases de ejemplo en datos/entrenamiento_NOMBRE.txt:
así se ve cómo la IA pasa del galimatías a hablar. Al acabar guarda el mejor modelo (el de menos error en textos que no ha visto).

Uso: python3 ia/entrenar.py NOMBRE [--dim 128] [--capas 4] [--cabezas 4] [--pasos 6000] ...
"""
import argparse, json, math, os, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizador import Tokenizador

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # la carpeta del repositorio
CARPETA = os.path.join(RAIZ, 'datos')
BITACORA = CARPETA                 # el registro de cada entrenamiento, junto a sus datos


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.peso = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * self.peso


class Bloque(nn.Module):
    def __init__(self, d, cabezas, oculta, drop):
        super().__init__()
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.sube = nn.Linear(d, oculta, bias=False)
        self.baja = nn.Linear(oculta, d, bias=False)
        self.cabezas, self.drop = cabezas, drop

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).split(D, dim=-1)
        forma = lambda t: t.view(B, T, self.cabezas, D // self.cabezas).transpose(1, 2)
        a = F.scaled_dot_product_attention(forma(q), forma(k), forma(v), is_causal=True,
                                           dropout_p=self.drop if self.training else 0.0)
        x = x + F.dropout(self.o(a.transpose(1, 2).reshape(B, T, D)), self.drop, self.training)
        h = F.relu(self.sube(self.n2(x)))
        return x + F.dropout(self.baja(h), self.drop, self.training)


class Modelo(nn.Module):
    def __init__(self, vocab, d, capas, cabezas, oculta, contexto, drop=0.0):
        super().__init__()
        self.conf = dict(vocab=vocab, d=d, capas=capas, cabezas=cabezas, oculta=oculta, contexto=contexto)
        self.tokens = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(contexto, d)
        self.bloques = nn.ModuleList(Bloque(d, cabezas, oculta, drop) for _ in range(capas))
        self.nf = RMSNorm(d)
        self.drop = drop
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
        for b in self.bloques:   # las salidas de cada bloque, más pequeñas al principio (como GPT-2)
            nn.init.normal_(b.o.weight, std=0.02 / math.sqrt(2 * capas))
            nn.init.normal_(b.baja.weight, std=0.02 / math.sqrt(2 * capas))

    def forward(self, ids):
        T = ids.shape[1]
        x = self.tokens(ids) + self.pos(torch.arange(T, device=ids.device))
        x = F.dropout(x, self.drop, self.training)
        for b in self.bloques:
            x = b(x)
        return self.nf(x) @ self.tokens.weight.T

    @torch.no_grad()
    def generar(self, inicio, n, temperatura=0.8, top_k=40):
        ids = torch.tensor([inicio], device=self.tokens.weight.device)
        for _ in range(n):
            logits = self(ids[:, -self.conf['contexto']:])[0, -1] / temperatura
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[-1]] = -float('inf')
            sig = torch.multinomial(F.softmax(logits, -1), 1)
            if sig.item() == 0:
                break
            ids = torch.cat([ids, sig[None]], 1)
        return ids[0, len(inicio):].tolist()


def legible(texto):
    return texto.replace('@', 'KAYBURN').replace('¶', ' / ').replace('\x01', '[POKéMON]').replace('\x02', '[BACKROOMS]')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('nombre')
    p.add_argument('--dim', type=int, default=128)
    p.add_argument('--capas', type=int, default=4)
    p.add_argument('--cabezas', type=int, default=4)
    p.add_argument('--oculta', type=int, default=0, help='por defecto, 3 veces dim')
    p.add_argument('--contexto', type=int, default=128)
    p.add_argument('--pasos', type=int, default=6000)
    p.add_argument('--lote', type=int, default=64)
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--drop', type=float, default=0.1)
    p.add_argument('--semilla', type=int, default=1)
    p.add_argument('--repetir', type=int, default=8, help='veces que se repiten los textos de los Backrooms')
    a = p.parse_args()
    torch.manual_seed(a.semilla); random.seed(a.semilla)
    dev = 'cuda'
    tok = Tokenizador.cargar(f'{CARPETA}/tokenizador.json')
    TEMAS = {'POKéMON': tok.id['\x01'], 'BACKROOMS': tok.id['\x02']}
    datos = np.fromfile(f'{CARPETA}/tokens.bin', dtype=np.uint16).astype(np.int64)
    # textos sueltos (cada uno acaba en 0); uno de cada 20, para medir con lo que no ha visto. Los de los Backrooms son
    # muchos menos, así que se repiten al aprender
    textos = np.split(datos, np.flatnonzero(datos == 0) + 1)
    textos = [t for t in textos if len(t)]
    prueba = [t for i, t in enumerate(textos) if i % 20 == 0]
    entreno = [t for i, t in enumerate(textos) if i % 20 for _ in range(a.repetir if t[0] == TEMAS['BACKROOMS'] else 1)]
    random.shuffle(entreno)
    entreno, prueba = torch.tensor(np.concatenate(entreno)), torch.tensor(np.concatenate(prueba))
    C = a.contexto

    def lote(t):
        i = torch.randint(0, len(t) - C - 1, (a.lote,))
        x = torch.stack([t[j:j + C] for j in i.tolist()])
        y = torch.stack([t[j + 1:j + C + 1] for j in i.tolist()])
        return x.to(dev), y.to(dev)

    m = Modelo(len(tok.tokens), a.dim, a.capas, a.cabezas, a.oculta or 3 * a.dim, C, a.drop).to(dev)
    n_param = sum(x.numel() for x in m.parameters())
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.1, betas=(0.9, 0.95))
    salida = f'{CARPETA}/modelos/{a.nombre}'
    os.makedirs(salida, exist_ok=True); os.makedirs(BITACORA, exist_ok=True)
    diario = open(f'{BITACORA}/entrenamiento_{a.nombre}.txt', 'w')

    def escribir(s):
        print(s); diario.write(s + '\n'); diario.flush()

    escribir(f'Modelo {a.nombre}: {n_param:,} parámetros ({json.dumps(m.conf)}), {len(entreno):,} tokens para '
             f'aprender. {time.strftime("%d/%m/%Y %H:%M")}'.replace(',', '.'))

    @torch.no_grad()
    def error_prueba():
        m.eval()
        e = sum(F.cross_entropy(m(x).flatten(0, 1), y.flatten()).item() for x, y in (lote(prueba) for _ in range(20)))
        m.train()
        return e / 20

    def ejemplos(paso, n=3):
        m.eval()
        escribir(f'\n--- Paso {paso} ---')
        for tema, i in TEMAS.items():
            for _ in range(n):
                escribir(f'  [{tema}] ' + legible(tok.decodificar(m.generar([0, i], 60))))
        m.train()

    muestras = {0, 25, 50, 100, 200, 400, 800, 1500, 3000}
    mejor, t0 = 1e9, time.time()
    for paso in range(a.pasos + 1):
        lr = a.lr * min(1, paso / 200) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * paso / a.pasos)))
        for g in opt.param_groups:
            g['lr'] = lr
        if paso in muestras or paso == a.pasos:
            ejemplos(paso)
        if paso % 250 == 0:
            e = error_prueba()
            escribir(f'paso {paso}: error con textos no vistos {e:.3f} ({time.time() - t0:.0f} s)')
            if e < mejor:
                mejor = e
                torch.save({'conf': m.conf, 'pesos': m.state_dict(), 'paso': paso, 'error': e}, f'{salida}/modelo.pt')
        if paso == a.pasos:
            break
        x, y = lote(entreno)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            perdida = F.cross_entropy(m(x).flatten(0, 1), y.flatten())
        opt.zero_grad(set_to_none=True)
        perdida.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    escribir(f'\nMejor error con textos no vistos: {mejor:.3f}. Modelo en {salida}/modelo.pt')


if __name__ == '__main__':
    main()
