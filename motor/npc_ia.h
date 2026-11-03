#ifndef GUARD_NPC_IA_H
#define GUARD_NPC_IA_H

// IA de los NPC: un transformador diminuto (ia/entrenar.py) que se ejecuta solo con números enteros.
// El modelo va en el cartucho (modelo.bin, lo escribe ia/exportar.py); para pensar usa un
// bloque de memoria que se le da al empezar (en la GBA, del montón: Alloc) y se devuelve al acabar.
// El mismo código se compila en el PC para comprobarlo (ia/prueba_pc.c, con NPCIA_PC).

#include <stdint.h>

#define NPCIA_MAX_CAPAS 8

struct NpcIaMatriz
{
    const int32_t *escala;   // una por fila: valor real = peso * escala / 2^24
    const uint32_t *datos;   // int8, de 4 en 4 filas: para cada columna, los pesos de las 4 filas en una palabra
    int filas, cols;
};

struct NpcIaCapa
{
    const int32_t *norma1, *norma2;   // Q12
    struct NpcIaMatriz qkv, o, sube, baja;
};

struct NpcIa
{
    int vocab, d, capas, cabezas, oculta, contexto;
    int temaPokemon, temaBackrooms;
    struct NpcIaMatriz tokens, pos;
    struct NpcIaCapa capa[NPCIA_MAX_CAPAS];
    const int32_t *normaFinal;
    const uint16_t *tokenInicio;   // vocab + 1 posiciones dentro de tokenTexto
    const uint8_t *tokenTexto;     // los tokens en la codificación de letras de la GBA
    // memoria de trabajo
    int32_t *x, *qkv, *att, *h, *tmp, *logits, *escalaK, *escalaV, *puntos;
    int8_t *x8, *cacheK, *cacheV;
    int contextoMax;
    uint32_t ciclos[6];   // medida de velocidad (en la GBA): multiplicar, escalas, atención, normas, elegir, total
};

uint32_t NpcIa_Memoria(const uint8_t *modelo, int contexto);
void NpcIa_Iniciar(struct NpcIa *ia, void *memoria, const uint8_t *modelo, int contexto);
// Mete el token «token» en la posición «pos» y deja en ia->logits lo probable que es cada token siguiente (Q12).
void NpcIa_Paso(struct NpcIa *ia, int token, int pos);
// Elige el token siguiente: temperatura en Q8 (205 = 0,8), entre los «topK» más probables. Sin fin si !permitirFin.
int NpcIa_Elegir(struct NpcIa *ia, int temperaturaQ8, int topK, int permitirFin, uint32_t *azar);
const uint8_t *NpcIa_Token(const struct NpcIa *ia, int token, int *largo);

#endif // GUARD_NPC_IA_H
