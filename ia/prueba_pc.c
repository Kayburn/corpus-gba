// Prueba en el PC del motor de la GBA (motor/npc_ia.c, el mismo código). La compila y la usa comprobar.py.
//   prueba_pc MODELO.bin generar TEMA N SEMILLA   -> N textos (tokens separados por espacios, uno por línea)
//   prueba_pc MODELO.bin evaluar < textos         -> para cada token de cada texto, los logits del siguiente (Q12)
#define NPCIA_PC
#include <stdio.h>
#include <string.h>
#include "../motor/npc_ia.c"

static uint8_t *Cargar(const char *ruta)
{
    FILE *f = fopen(ruta, "rb");
    long n;
    uint8_t *b;
    if (!f)
        return 0;
    fseek(f, 0, SEEK_END);
    n = ftell(f);
    fseek(f, 0, SEEK_SET);
    b = aligned_alloc(4, (n + 3) & ~3);
    if (fread(b, 1, n, f) != (size_t)n)
        return 0;
    fclose(f);
    return b;
}

int main(int argc, char **argv)
{
    uint8_t *modelo = Cargar(argv[1]);
    struct NpcIa ia;
    void *mem;
    if (!modelo)
        return 1;
    mem = aligned_alloc(4, (NpcIa_Memoria(modelo, 128) + 3) & ~3);
    NpcIa_Iniciar(&ia, mem, modelo, 128);
    if (!strcmp(argv[2], "generar"))
    {
        int tema = !strcmp(argv[3], "backrooms") ? ia.temaBackrooms : ia.temaPokemon, n = atoi(argv[4]), i;
        uint32_t azar = (uint32_t)atoi(argv[5]) | 1;
        for (i = 0; i < n; i++)
        {
            int pos = 0, t = 0;
            NpcIa_Paso(&ia, 0, pos++);
            NpcIa_Paso(&ia, tema, pos++);
            while (pos < ia.contextoMax)
            {
                t = NpcIa_Elegir(&ia, 205, 40, pos > 4, &azar);
                if (t == 0)
                    break;
                printf("%d ", t);
                NpcIa_Paso(&ia, t, pos++);
            }
            printf("\n");
        }
    }
    else
    {
        char linea[1 << 16];
        while (fgets(linea, sizeof(linea), stdin))
        {
            char *p = linea, *fin;
            int pos = 0, r;
            for (;;)
            {
                long t = strtol(p, &fin, 10);
                if (fin == p || pos >= ia.contextoMax)
                    break;
                p = fin;
                NpcIa_Paso(&ia, (int)t, pos++);
                for (r = 0; r < ia.vocab; r++)
                    printf("%d%c", ia.logits[r], r + 1 < ia.vocab ? ' ' : '\n');
            }
            printf("---\n");
        }
    }
    return 0;
}
