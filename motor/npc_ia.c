// IA de los NPC: el transformador de ia/entrenar.py hecho solo con números enteros (la GBA no tiene
// decimales). Convenios:
// - Q12: un número real guardado multiplicado por 4096 (int32). Así van las activaciones que se suman (x, logits…).
// - Para multiplicar por una matriz, el vector se pasa a int8 con una escala: real = x8 * escala / 2^20. Los pesos son
//   int8 con una escala por fila: real = peso * escala / 2^24. El producto se acumula en int32 y se pasa a Q12 con una
//   sola multiplicación de 64 bits por fila.
// - Las normas RMS no dividen cada elemento: el 1/rms se mete en la escala del vector cuantizado.
// Lo único que va en la IWRAM (la RAM rápida) es el bucle de multiplicar, que es donde se va casi todo el tiempo; el
// resto, no, porque en la IWRAM vive la pila del juego.
#ifdef NPCIA_PC
#include <stdlib.h>
#define NPCIA_RAPIDA
#define Reloj() 0
#define NPCIA_ARM
#else
#include "global.h"
#define NPCIA_RAPIDA __attribute__((section(".iwram.code"), target("arm"), noinline))
// en modo ARM (en el cartucho, sin gastar IWRAM): las multiplicaciones de 64 bits son una instrucción (SMULL); en
// Thumb, que es como se compila el juego, son una llamada a __aeabi_lmul de unos 100 ciclos
#define NPCIA_ARM __attribute__((target("arm")))
// ciclos de la CPU, con los temporizadores 2 y 3 encadenados (solo los pone en marcha la medida de velocidad del juego)
static inline uint32_t Reloj(void)
{
    uint32_t alto, bajo;
    do
    {
        alto = REG_TM3CNT_L;
        bajo = REG_TM2CNT_L;
    } while (alto != REG_TM3CNT_L);
    return (alto << 16) | bajo;
}
#endif
#include "npc_ia.h"

#define MEDIR(ia, parte, codigo) do { uint32_t t0_ = Reloj(); codigo; (ia)->ciclos[parte] += Reloj() - t0_; } while (0)

#define MAGIA 0x3143504E   // «NPC1»

// e^(-i/16) en Q15, para el softmax (diferencias de 0 a 16; más allá, 0)
static const uint16_t sExp[257] = {
#include "exp.h"
};

static int32_t Exp(int32_t difQ12)   // e^(-dif), en Q15
{
    uint32_t i = (uint32_t)difQ12 >> 8, f = difQ12 & 255;
    if (i >= 256)
        return 0;
    return sExp[i] - (((sExp[i] - sExp[i + 1]) * f) >> 8);
}

static uint32_t RaizEntera(uint64_t n)
{
    uint64_t r = 0, bit = (uint64_t)1 << 62;
    while (bit > n)
        bit >>= 2;
    while (bit)
    {
        if (n >= r + bit)
        {
            n -= r + bit;
            r = (r >> 1) + bit;
        }
        else
        {
            r >>= 1;
        }
        bit >>= 2;
    }
    return (uint32_t)r;
}

// acc[r] = suma de peso[r][j] * x[j], de 4 en 4 filas (los pesos de las 4 filas vienen juntos en cada palabra)
NPCIA_RAPIDA static void Producto(const uint32_t *w, const int8_t *x, int cols, int filas, int32_t *acc)
{
    int b, j;
    for (b = 0; b < filas; b += 4)
    {
        int32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        const int8_t *xp = x;
        for (j = 0; j < cols; j++)
        {
            uint32_t p = *w++;
            int32_t xj = *xp++;
            a0 += (int32_t)(int8_t)p * xj;
            a1 += (int32_t)(int8_t)(p >> 8) * xj;
            a2 += (int32_t)(int8_t)(p >> 16) * xj;
            a3 += ((int32_t)p >> 24) * xj;
        }
        acc[b] = a0;
        acc[b + 1] = a1;
        acc[b + 2] = a2;
        acc[b + 3] = a3;
    }
}

// salida (Q12) = matriz * x8, con x8 de escala xs
NPCIA_ARM static void Multiplicar(struct NpcIa *ia, const struct NpcIaMatriz *m, const int8_t *x8, int32_t xs, int32_t *salida)
{
    int r;
    MEDIR(ia, 0, Producto(m->datos, x8, m->cols, m->filas, salida));
    MEDIR(ia, 1, {
        for (r = 0; r < m->filas; r++)
            salida[r] = (int32_t)(((int64_t)salida[r] * ((int64_t)xs * m->escala[r])) >> 32);
    });
}

NPCIA_ARM static int32_t MaximoAbs(const int32_t *v, int n)
{
    int32_t m = 1;
    int i;
    for (i = 0; i < n; i++)
    {
        int32_t a = v[i] < 0 ? -v[i] : v[i];
        if (a > m)
            m = a;
    }
    return m;
}

// v (Q12) -> q (int8); devuelve la escala (real = q * escala / 2^20)
NPCIA_ARM static int32_t Cuantizar(const int32_t *v, int n, int8_t *q)
{
    int32_t m = MaximoAbs(v, n);
    int32_t r = (127 << 24) / m;
    int i;
    for (i = 0; i < n; i++)
        q[i] = (int8_t)(((int64_t)v[i] * r + (1 << 23)) >> 24);
    return (int32_t)(((int64_t)m << 8) / 127);
}

// norma RMS de x con los pesos g, ya cuantizada en q; devuelve la escala
NPCIA_ARM static int32_t Normalizar(struct NpcIa *ia, const int32_t *x, const int32_t *g, int8_t *q)
{
    int n = ia->d, i;
    int64_t s = 0;
    int32_t rms, m;
    for (i = 0; i < n; i++)
        s += (int64_t)x[i] * x[i];
    rms = (int32_t)RaizEntera((uint64_t)s / n);   // Q12
    if (rms < 1)
        rms = 1;
    for (i = 0; i < n; i++)
        ia->tmp[i] = (int32_t)(((int64_t)x[i] * g[i]) >> 12);
    m = MaximoAbs(ia->tmp, n);
    Cuantizar(ia->tmp, n, q);
    return (int32_t)(((int64_t)m << 20) / ((int64_t)127 * rms));
}

static const uint8_t *LeerMatriz(const uint8_t *p, struct NpcIaMatriz *m, int filas, int cols)
{
    m->filas = filas;
    m->cols = cols;
    m->escala = (const int32_t *)p;
    p += 4 * filas;
    m->datos = (const uint32_t *)p;
    return p + filas * cols;
}

// fila r de una matriz (para las tablas de tokens y posiciones), sumada a x en Q12
NPCIA_ARM static void SumarFila(const struct NpcIaMatriz *m, int r, int32_t *x)
{
    const int8_t *base = (const int8_t *)(m->datos + (r >> 2) * m->cols) + (r & 3);
    int32_t e = m->escala[r];
    int j;
    for (j = 0; j < m->cols; j++)
        x[j] += (base[4 * j] * e) >> 12;
}

static const uint32_t *Cabecera(const uint8_t *modelo)
{
    const uint32_t *c = (const uint32_t *)modelo;
    return c[0] == MAGIA ? c : 0;
}

uint32_t NpcIa_Memoria(const uint8_t *modelo, int contexto)
{
    const uint32_t *c = Cabecera(modelo);
    uint32_t vocab = c[1], d = c[2], capas = c[3], cabezas = c[4], oculta = c[5];
    uint32_t mayor = oculta > 3 * d ? oculta : 3 * d;
    return 4 * (d + 3 * d + d + oculta + d + vocab + 2 * capas * contexto * cabezas + contexto)
         + (mayor + 2 * capas * contexto * d) + 16;
}

void NpcIa_Iniciar(struct NpcIa *ia, void *memoria, const uint8_t *modelo, int contexto)
{
    const uint32_t *c = Cabecera(modelo);
    const uint8_t *p;
    int8_t *b;
    int l, mayor;
    ia->vocab = c[1];
    ia->d = c[2];
    ia->capas = c[3];
    ia->cabezas = c[4];
    ia->oculta = c[5];
    ia->contexto = c[6];
    ia->temaPokemon = c[7];
    ia->temaBackrooms = c[8];
    ia->contextoMax = contexto < ia->contexto ? contexto : ia->contexto;
    for (l = 0; l < 6; l++)
        ia->ciclos[l] = 0;
    p = modelo + 4 * 10;
    p = LeerMatriz(p, &ia->tokens, ia->vocab, ia->d);
    p = LeerMatriz(p, &ia->pos, ia->contexto, ia->d);
    for (l = 0; l < ia->capas; l++)
    {
        struct NpcIaCapa *k = &ia->capa[l];
        k->norma1 = (const int32_t *)p;
        p += 4 * ia->d;
        p = LeerMatriz(p, &k->qkv, 3 * ia->d, ia->d);
        p = LeerMatriz(p, &k->o, ia->d, ia->d);
        k->norma2 = (const int32_t *)p;
        p += 4 * ia->d;
        p = LeerMatriz(p, &k->sube, ia->oculta, ia->d);
        p = LeerMatriz(p, &k->baja, ia->d, ia->oculta);
    }
    ia->normaFinal = (const int32_t *)p;
    p += 4 * ia->d;
    ia->tokenInicio = (const uint16_t *)(modelo + c[9]);
    ia->tokenTexto = (const uint8_t *)(ia->tokenInicio + ia->vocab + 1);

    ia->x = (int32_t *)memoria;
    ia->qkv = ia->x + ia->d;
    ia->att = ia->qkv + 3 * ia->d;
    ia->h = ia->att + ia->d;
    ia->tmp = ia->h + ia->oculta;
    ia->logits = ia->tmp + ia->d;
    ia->escalaK = ia->logits + ia->vocab;
    ia->escalaV = ia->escalaK + ia->capas * ia->contextoMax * ia->cabezas;
    ia->puntos = ia->escalaV + ia->capas * ia->contextoMax * ia->cabezas;
    b = (int8_t *)(ia->puntos + ia->contextoMax);
    mayor = ia->oculta > 3 * ia->d ? ia->oculta : 3 * ia->d;
    ia->x8 = b;
    ia->cacheK = b + mayor;
    ia->cacheV = ia->cacheK + ia->capas * ia->contextoMax * ia->d;
}

NPCIA_ARM static void Atencion(struct NpcIa *ia, int l, int pos)
{
    int d = ia->d, H = ia->cabezas, hd = d / H, h, t, i;
    int8_t *k8 = ia->cacheK + (l * ia->contextoMax) * d, *v8 = ia->cacheV + (l * ia->contextoMax) * d;
    int32_t *ks = ia->escalaK + l * ia->contextoMax * H, *vs = ia->escalaV + l * ia->contextoMax * H;
    // 1/raíz(hd) en Q16
    int32_t invRaiz = (int32_t)(((int64_t)65536 << 12) / RaizEntera((uint64_t)hd << 24));
    for (h = 0; h < H; h++)
    {
        int32_t qs, c, mx = -0x7FFFFFFF, total = 0, inv;
        int8_t *q8 = ia->x8;
        ks[pos * H + h] = Cuantizar(ia->qkv + d + h * hd, hd, k8 + pos * d + h * hd);
        vs[pos * H + h] = Cuantizar(ia->qkv + 2 * d + h * hd, hd, v8 + pos * d + h * hd);
        qs = Cuantizar(ia->qkv + h * hd, hd, q8);
        c = (int32_t)(((int64_t)qs * invRaiz) >> 16);
        for (t = 0; t <= pos; t++)
        {
            const int8_t *k = k8 + t * d + h * hd;
            int32_t acc = 0, s;
            for (i = 0; i < hd; i++)
                acc += q8[i] * k[i];
            s = (int32_t)(((int64_t)acc * c * ks[t * H + h]) >> 28);   // Q12
            ia->puntos[t] = s;
            if (s > mx)
                mx = s;
        }
        for (t = 0; t <= pos; t++)
        {
            ia->puntos[t] = Exp(mx - ia->puntos[t]);   // Q15
            total += ia->puntos[t];
        }
        for (i = 0; i < hd; i++)
            ia->att[h * hd + i] = 0;
        inv = (1 << 30) / total;   // una división por cabeza, no una por posición
        for (t = 0; t <= pos; t++)
        {
            int32_t pn = (ia->puntos[t] * inv) >> 15;                   // probabilidad, Q15
            int32_t w = (int32_t)(((int64_t)pn * vs[t * H + h]) >> 15);
            const int8_t *v = v8 + t * d + h * hd;
            for (i = 0; i < hd; i++)
                ia->att[h * hd + i] += w * v[i];
        }
        for (i = 0; i < hd; i++)
            ia->att[h * hd + i] >>= 8;   // a Q12
    }
}

void NpcIa_Paso(struct NpcIa *ia, int token, int pos)
{
    int d = ia->d, l, i;
    int32_t xs;
    uint32_t t0 = Reloj();
    for (i = 0; i < d; i++)
        ia->x[i] = 0;
    SumarFila(&ia->tokens, token, ia->x);
    SumarFila(&ia->pos, pos, ia->x);
    for (l = 0; l < ia->capas; l++)
    {
        struct NpcIaCapa *k = &ia->capa[l];
        MEDIR(ia, 3, xs = Normalizar(ia, ia->x, k->norma1, ia->x8));
        Multiplicar(ia, &k->qkv, ia->x8, xs, ia->qkv);
        MEDIR(ia, 2, Atencion(ia, l, pos));
        MEDIR(ia, 3, xs = Cuantizar(ia->att, d, ia->x8));
        Multiplicar(ia, &k->o, ia->x8, xs, ia->tmp);
        for (i = 0; i < d; i++)
            ia->x[i] += ia->tmp[i];
        MEDIR(ia, 3, xs = Normalizar(ia, ia->x, k->norma2, ia->x8));
        Multiplicar(ia, &k->sube, ia->x8, xs, ia->h);
        for (i = 0; i < ia->oculta; i++)
            if (ia->h[i] < 0)
                ia->h[i] = 0;
        MEDIR(ia, 3, xs = Cuantizar(ia->h, ia->oculta, ia->x8));
        Multiplicar(ia, &k->baja, ia->x8, xs, ia->tmp);
        for (i = 0; i < d; i++)
            ia->x[i] += ia->tmp[i];
    }
    MEDIR(ia, 3, xs = Normalizar(ia, ia->x, ia->normaFinal, ia->x8));
    Multiplicar(ia, &ia->tokens, ia->x8, xs, ia->logits);
    ia->ciclos[5] += Reloj() - t0;
}

static uint32_t Azar(uint32_t *s)   // xorshift32
{
    uint32_t x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return *s = x;
}

#define NPCIA_TOPK_MAX 64

NPCIA_ARM int NpcIa_Elegir(struct NpcIa *ia, int temperaturaQ8, int topK, int permitirFin, uint32_t *azar)
{
    int32_t mejor[NPCIA_TOPK_MAX], id[NPCIA_TOPK_MAX], n = 0, r, i, j;
    int32_t inv = (1 << 24) / temperaturaQ8;
    uint32_t total = 0, x, t0 = Reloj();
    if (topK > NPCIA_TOPK_MAX)
        topK = NPCIA_TOPK_MAX;
    for (r = permitirFin ? 0 : 1; r < ia->vocab; r++)
    {
        int32_t z = (int32_t)(((int64_t)ia->logits[r] * inv) >> 16);
        if (n == topK && z <= mejor[n - 1])
            continue;
        j = n < topK ? n++ : n - 1;
        while (j > 0 && mejor[j - 1] < z)
        {
            mejor[j] = mejor[j - 1];
            id[j] = id[j - 1];
            j--;
        }
        mejor[j] = z;
        id[j] = r;
    }
    for (i = n - 1; i >= 0; i--)   // de atrás adelante: mejor[0] (el máximo) se cambia el último
    {
        mejor[i] = Exp(mejor[0] - mejor[i]);
        total += mejor[i];
    }
    x = Azar(azar) % total;
    for (i = 0; i < n - 1; i++)
    {
        if (x < (uint32_t)mejor[i])
            break;
        x -= mejor[i];
    }
    ia->ciclos[4] += Reloj() - t0;
    return id[i];
}

const uint8_t *NpcIa_Token(const struct NpcIa *ia, int token, int *largo)
{
    *largo = ia->tokenInicio[token + 1] - ia->tokenInicio[token];
    return ia->tokenTexto + ia->tokenInicio[token];
}
