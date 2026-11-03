# Corpus

Corpus es una IA diminuta que habla dentro de una Game Boy Advance. Es un transformador (la misma familia que ChatGPT,
en pequeñito) que aprende en el PC y luego se ejecuta en la GBA solo con números enteros, porque la GBA no tiene
decimales. Es el NPC del vídeo de Kayburn sobre meter una IA en los NPC de Pokémon Esmeralda.

- 430.944 parámetros: unos 450 KB en el cartucho
- 512 tokens, 4 capas de 96, 4 cabezas de atención y hasta 128 tokens de contexto
- en la GBA tarda de 5 a 20 segundos por frase
- el motor de enteros coincide con el modelo original en el token más probable el 97,7 % de las veces

## Qué hay aquí

| | |
|---|---|
| `ia/corpus.py` | 1. Junta los textos del entrenamiento, limpios y uno por línea |
| `ia/tokenizador.py` | 2. Saca los 512 tokens (BPE) y pasa los textos a tokens |
| `ia/entrenar.py` | 3. Entrena el transformador en la gráfica (PyTorch) |
| `ia/exportar.py` | 4. Lo pasa a números enteros para la GBA (`modelo.bin`) |
| `ia/comprobar.py` | 5. Comprueba en el PC que el motor de la GBA piensa igual que el modelo de Python |
| `ia/prueba_pc.c` | El motor de la GBA compilado para el PC (lo usa `comprobar.py`) |
| `motor/` | El motor de la GBA en C: el transformador hecho solo con enteros |
| `textos/` | Los textos propios del entrenamiento (Backrooms) |

## Cómo se hace Corpus

Hace falta Python 3 con PyTorch y NumPy, una gráfica NVIDIA (CUDA) y gcc. Desde la carpeta del repositorio:

```
python3 ia/corpus.py
python3 ia/tokenizador.py
python3 ia/entrenar.py d96 --dim 96
CHARMAP=ruta/de/pokeemerald/charmap.txt python3 ia/exportar.py d96
python3 ia/comprobar.py d96
```

Todo lo que sale va a `datos/`: el modelo para la GBA queda en `datos/modelos/d96/modelo.bin`. Corpus se entrenó
así, con 6.000 pasos (menos de un minuto en la gráfica). `exportar.py` necesita el `charmap.txt` de
[pokeemerald](https://github.com/pret/pokeemerald), que dice qué número es cada letra en la GBA.

## Por dentro

- En el PC, el modelo aprende con decimales. Al exportarlo, cada peso pasa a un número de 8 bits con una escala por
  fila, y el resto de números van multiplicados por 4096 (Q12).
- En la GBA, casi todo el tiempo se va en multiplicar matrices: ese bucle va en la RAM rápida (IWRAM), en modo ARM.
- `motor/npc_ia.c` se compila dentro de pokeemerald o en el PC con `NPCIA_PC` (como hace `ia/prueba_pc.c`). La parte
  que lo une a los NPC del juego (el script y el cuadro de texto) no va aquí.

## Los textos

Corpus aprendió con dos tipos de textos:

- **Backrooms**: los de `textos/`, escritos para el canal (uno por línea; `¶` es cambio de página y `@`, el nombre del
  jugador).
- **Pokémon**: los diálogos en español de Pokémon Esmeralda y Rojo Fuego y las descripciones de la Pokédex (de
  PokéAPI). No están aquí, porque son de Nintendo, Creatures y Game Freak. `corpus.py` los busca en `pokemon/` (o en
  la carpeta de la variable `TEXTOS_POKEMON`): `esmeralda_es.json` y `rojofuego_es.json`, con los textos del juego
  como valores y con los códigos de pokeemerald (`{PLAYER}`, `\p`, `\n`…), y `pokeapi_es.json`, con
  `{"especies": {…: {"descripcion": …}}}`. Sin ellos, Corpus aprende solo con los textos de `textos/`.

## Licencia

© 2026 Kayburn. GPL-3.0 (ver `LICENSE`): se puede usar, cambiar y compartir, pero quien reparta algo hecho a partir de
él tiene que dar también su código, con esta misma licencia.

Pokémon es una marca de Nintendo, Creatures y Game Freak; este proyecto no tiene ninguna relación con ellos.
