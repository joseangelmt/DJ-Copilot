"""Puntúa con CLAP cada canción de la caché local de audio con descriptores de texto y genera una
tabla HTML.

La similitud de una canción con un descriptor es el coseno entre el embedding de audio de la canción
y el embedding de texto del descriptor. La tabla muestra, para cada descriptor, el percentil de cada
canción dentro de la colección analizada, calculado sobre la diferencia entre ese coseno y la media
de los cosenos de la canción con todos los descriptores.
"""
import argparse
import datetime
import html
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile
import torch
import torchaudio.functional
from transformers import ClapModel, ClapProcessor

EXTENSIONES = {".aiff", ".aif", ".wav", ".flac", ".mp3", ".ogg", ".m4a"}

IDENTIFICADOR_MODELO = "laion/clap-htsat-unfused"
REVISION_MODELO = "8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a"

CARPETA_DEL_SCRIPT = Path(__file__).resolve().parent
SEGUNDOS_MINIMOS_VENTANA_FINAL = 1.0

# Categoría -> lista de (etiqueta en español, adjetivo en inglés que se pasa a CLAP).
DESCRIPTORES = {
    "Ambiente": [
        ("bailable", "danceable"),
        ("hipnótico", "hypnotic"),
        ("festivo", "festive"),
        ("místico", "mystical"),
        ("orgánico", "organic"),
    ],
    "Emociones": [
        ("euforia", "euphoric"),
        ("confianza", "confident"),
        ("sensualidad", "sensual"),
        ("esperanza", "hopeful"),
        ("alegría", "joyful"),
    ],
}

PLANTILLAS_TEXTO = ("{} music", "a {} song", "this track sounds {}")


def genero_de_ruta(ruta_relativa):
    """Primera carpeta de la ruta relativa (con separador '/'), o '(raíz)' si el fichero no está en una carpeta."""
    partes = ruta_relativa.split("/")
    return partes[0] if len(partes) > 1 else "(raíz)"


def titulo_de_ruta(ruta_relativa):
    return os.path.splitext(ruta_relativa.rpartition("/")[2])[0]


def listar_canciones(directorio_audio, limite):
    """Devuelve las rutas relativas (con '/') de los ficheros de audio, ordenadas; con limite, solo las primeras."""
    rutas = sorted(ruta.relative_to(directorio_audio).as_posix()
                   for ruta in directorio_audio.rglob("*")
                   if ruta.is_file() and ruta.suffix.lower() in EXTENSIONES)
    return rutas[:limite] if limite else rutas


def leer_audio_mono(ruta, frecuencia_muestreo_hz, dispositivo):
    """Lee el fichero, promedia los canales y re-muestrea a frecuencia_muestreo_hz. Devuelve float32 (n,)."""
    datos, frecuencia_original_hz = soundfile.read(str(ruta), dtype="float32", always_2d=True)
    mono = torch.from_numpy(datos.mean(axis=1)).to(dispositivo)
    if frecuencia_original_hz != frecuencia_muestreo_hz:
        mono = torchaudio.functional.resample(mono, frecuencia_original_hz, frecuencia_muestreo_hz)
    return mono.cpu().numpy()


def cortar_ventanas(audio, muestras_ventana, muestras_minimas_ultima):
    """Ventanas consecutivas sin solape; la última, incompleta, solo se conserva si tiene al menos
    muestras_minimas_ultima muestras."""
    ventanas = []
    for inicio in range(0, len(audio), muestras_ventana):
        ventana = audio[inicio:inicio + muestras_ventana]
        if len(ventana) == muestras_ventana or len(ventana) >= muestras_minimas_ultima:
            ventanas.append(ventana)
    return ventanas


def normalizar(vectores):
    """Normaliza en L2 cada fila de un array [n, d]."""
    return vectores / np.linalg.norm(vectores, axis=1, keepdims=True)


def calcular_embedding_audio(modelo, procesador, audio, segundos_ventana, lote, dispositivo):
    """Devuelve el embedding de la canción, float32 [512], normalizado: media de los embeddings
    normalizados de las ventanas, normalizada de nuevo."""
    frecuencia_muestreo_hz = procesador.feature_extractor.sampling_rate
    ventanas = cortar_ventanas(audio,
                               round(segundos_ventana * frecuencia_muestreo_hz),
                               round(SEGUNDOS_MINIMOS_VENTANA_FINAL * frecuencia_muestreo_hz))
    if not ventanas:
        raise ValueError(f"el audio dura menos de {SEGUNDOS_MINIMOS_VENTANA_FINAL:g} s")

    vectores = []
    for inicio in range(0, len(ventanas), lote):
        # Con truncation="rand_trunc" solo se recorta al azar un audio más largo que max_length_s;
        # segundos_ventana no supera max_length_s, y las ventanas más cortas se rellenan con
        # repetición y ceros (padding="repeatpad"), sin azar.
        entradas = procesador(audio=ventanas[inicio:inicio + lote],
                              sampling_rate=frecuencia_muestreo_hz, return_tensors="pt")
        with torch.inference_mode():
            vectores.append(modelo.get_audio_features(
                input_features=entradas["input_features"].to(dispositivo)).float().cpu().numpy())
    por_ventana = normalizar(np.concatenate(vectores))
    return normalizar(por_ventana.mean(axis=0, keepdims=True))[0].astype(np.float32)


def calcular_embeddings_de_texto(modelo, procesador, dispositivo):
    """Devuelve float32 [D, 512], uno por descriptor en el orden de DESCRIPTORES: media de los
    embeddings normalizados de las plantillas, normalizada de nuevo."""
    adjetivos = [adjetivo for lista in DESCRIPTORES.values() for _, adjetivo in lista]
    resultado = []
    for adjetivo in adjetivos:
        textos = [plantilla.format(adjetivo) for plantilla in PLANTILLAS_TEXTO]
        entradas = procesador(text=textos, return_tensors="pt", padding=True).to(dispositivo)
        with torch.inference_mode():
            vectores = modelo.get_text_features(**entradas).float().cpu().numpy()
        resultado.append(normalizar(normalizar(vectores).mean(axis=0, keepdims=True))[0])
    return np.stack(resultado).astype(np.float32)


def similitudes_centradas(similitudes):
    """Resta a cada coseno la media de los cosenos de su canción con todos los descriptores."""
    return similitudes - similitudes.mean(axis=1, keepdims=True)


def percentiles(similitudes):
    """Percentil 0-100 de cada canción para cada descriptor, calculado sobre las similitudes centradas:
    rango medio entre las N canciones, con la de menor diferencia en 0 y la de mayor en 100.
    Con una sola canción vale 50."""
    similitudes = similitudes_centradas(similitudes)
    canciones_count, descriptores_count = similitudes.shape
    resultado = np.full(similitudes.shape, 50.0)
    if canciones_count < 2:
        return resultado
    for columna in range(descriptores_count):
        valores = similitudes[:, columna]
        orden = np.argsort(valores, kind="stable")
        ordenados = valores[orden]
        rangos = np.empty(canciones_count)
        inicio = 0
        while inicio < canciones_count:
            fin = inicio
            while fin + 1 < canciones_count and ordenados[fin + 1] == ordenados[inicio]:
                fin += 1
            rangos[orden[inicio:fin + 1]] = (inicio + fin) / 2  # rango medio, empezando en 0
            inicio = fin + 1
        resultado[:, columna] = rangos / (canciones_count - 1) * 100
    return resultado


def guardar_npz(ruta_npz, **arrays):
    temporal = ruta_npz.with_name(ruta_npz.name + ".parte")
    with open(temporal, "wb") as fichero:
        np.savez(fichero, **arrays)
    os.replace(temporal, ruta_npz)


def escribir_texto(ruta, texto):
    temporal = ruta.with_name(ruta.name + ".parte")
    temporal.write_text(texto, encoding="utf-8")
    os.replace(temporal, ruta)


CSS = """
:root {
  --fondo: #f4f5f2;
  --superficie: #ffffff;
  --texto: #1d2321;
  --texto-suave: #5d6763;
  --borde: #d9ddd8;
  --acento: #0f6e66;
  --fuente-titulo: "Archivo", "Helvetica Neue", Arial, sans-serif;
  --fuente-texto: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif;
  --fuente-mono: "IBM Plex Mono", Consolas, monospace;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) {
  :root {
    --fondo: #141917;
    --superficie: #1c2220;
    --texto: #e6ebe8;
    --texto-suave: #9aa59f;
    --borde: #2f3835;
    --acento: #4fc2b6;
    color-scheme: dark;
  }
}
* { box-sizing: border-box; }
html, body { margin: 0; }
body {
  background: var(--fondo);
  color: var(--texto);
  font-family: var(--fuente-texto);
  font-size: 15px;
  line-height: 1.45;
}
header { padding: 20px 24px 16px; border-bottom: 1px solid var(--borde); display: grid; gap: 12px; }
h1 { font-family: var(--fuente-titulo); font-weight: 700; font-size: 24px; margin: 0; text-wrap: balance; }
h2 { font-family: var(--fuente-titulo); font-size: 18px; margin: 0 0 8px; }
.explicacion { margin: 0; color: var(--texto-suave); max-width: 90ch; }
main { padding: 20px 24px; display: grid; gap: 28px; }
.controles { display: flex; align-items: center; gap: 8px; margin-bottom: 10px; }
.controles label { font-size: 13px; color: var(--texto-suave); }
select {
  padding: 6px 10px;
  border: 1px solid var(--borde);
  border-radius: 4px;
  background: var(--superficie);
  color: var(--texto);
  font: inherit;
}
select:focus-visible, th.ordenable:focus-visible { outline: 2px solid var(--acento); outline-offset: 2px; }
.contenedor { overflow-x: auto; border: 1px solid var(--borde); border-radius: 6px; background: var(--superficie); }
table { border-collapse: collapse; width: 100%; font-size: 14px; }
th, td { padding: 6px 10px; border-bottom: 1px solid var(--borde); text-align: left; }
thead th { background: var(--superficie); font-weight: 600; white-space: nowrap; }
thead tr.categorias th { text-align: center; font-family: var(--fuente-titulo); font-size: 12px;
  text-transform: uppercase; letter-spacing: .05em; color: var(--texto-suave); border-bottom: 1px solid var(--borde); }
thead tr.categorias th + th { border-left: 1px solid var(--borde); }
th.ordenable { cursor: pointer; user-select: none; }
th.ordenable:hover { color: var(--acento); }
th.ordenable[aria-sort="ascending"]::after { content: " \\25B2"; font-size: 10px; }
th.ordenable[aria-sort="descending"]::after { content: " \\25BC"; font-size: 10px; }
td.num, th.num { text-align: right; font-family: var(--fuente-mono); font-variant-numeric: tabular-nums; }
td.valor { background: color-mix(in srgb, var(--acento) calc(var(--v) * 0.55%), var(--superficie)); }
tbody tr:last-child td { border-bottom: 0; }
td.titulo { min-width: 320px; }
.nota { font-size: 13px; color: var(--texto-suave); margin: 8px 0 0; max-width: 90ch; }
"""

JAVASCRIPT = """
const tabla = document.getElementById("tabla-principal");
const cuerpo = tabla.tBodies[0];
const cabeceras = tabla.querySelectorAll("th.ordenable");
function valorDeCelda(fila, columna, numerica) {
  const celda = fila.cells[columna];
  return numerica ? Number(celda.dataset.orden) : celda.textContent.trim().toLocaleLowerCase("es");
}
cabeceras.forEach(cabecera => {
  const ordenar = () => {
    const columna = cabecera.cellIndex;
    const numerica = cabecera.classList.contains("num");
    const ascendente = cabecera.getAttribute("aria-sort") !== "ascending";
    cabeceras.forEach(otra => otra.removeAttribute("aria-sort"));
    cabecera.setAttribute("aria-sort", ascendente ? "ascending" : "descending");
    const filas = Array.from(cuerpo.rows).sort((a, b) => {
      const va = valorDeCelda(a, columna, numerica), vb = valorDeCelda(b, columna, numerica);
      const orden = numerica ? va - vb : va.localeCompare(vb, "es");
      return ascendente ? orden : -orden;
    });
    filas.forEach(fila => cuerpo.appendChild(fila));
  };
  cabecera.addEventListener("click", ordenar);
  cabecera.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); ordenar(); } });
});
document.getElementById("filtro-genero").addEventListener("change", e => {
  Array.from(cuerpo.rows).forEach(fila => {
    fila.hidden = e.target.value !== "" && fila.cells[0].textContent !== e.target.value;
  });
});
"""


def cabecera_agrupada(categorias, etiquetas, columnas_previas_html, ordenables):
    """Dos filas de cabecera: la primera agrupa los descriptores por categoría; la segunda contiene las
    columnas previas y una columna por descriptor."""
    grupos = []
    for categoria in categorias:
        if grupos and grupos[-1][0] == categoria:
            grupos[-1][1] += 1
        else:
            grupos.append([categoria, 1])
    fila_categorias = '<tr class="categorias"><th colspan="{}"></th>{}</tr>'.format(
        columnas_previas_html[0],
        "".join(f'<th colspan="{cantidad}">{html.escape(categoria)}</th>' for categoria, cantidad in grupos))
    clase = ' class="ordenable num" tabindex="0"' if ordenables else ' class="num"'
    fila_columnas = "<tr>{}{}</tr>".format(
        columnas_previas_html[1],
        "".join(f"<th{clase}>{html.escape(etiqueta)}</th>" for etiqueta in etiquetas))
    return fila_categorias + fila_columnas


def celda_valor(percentil, similitud=None, diferencia=None):
    titulo = (f' title="coseno {similitud:.4f} · diferencia con la media de la canción {diferencia:+.4f}"'
              if similitud is not None else "")
    return (f'<td class="num valor" style="--v:{percentil:.1f}" data-orden="{percentil:.4f}"{titulo}>'
            f"{round(percentil)}</td>")


def medias_por_genero(generos, percentiles_canciones):
    """Devuelve ({género: media de percentiles [D]}, géneros ordenados)."""
    ordenados = sorted(set(generos))
    generos_array = np.array(generos)
    return {genero: percentiles_canciones[generos_array == genero].mean(axis=0) for genero in ordenados}, ordenados


def generar_html(generos, titulos, categorias, etiquetas, similitudes, modelo, revision, segundos_ventana):
    canciones_count = len(titulos)
    percentiles_canciones = percentiles(similitudes)
    diferencias = similitudes_centradas(similitudes)
    medias, generos_ordenados = medias_por_genero(generos, percentiles_canciones)
    fecha = datetime.date.today().isoformat()

    filas = []
    for i in range(canciones_count):
        celdas = "".join(celda_valor(percentiles_canciones[i, d], similitudes[i, d], diferencias[i, d])
                         for d in range(len(etiquetas)))
        filas.append(f"<tr><td>{html.escape(generos[i])}</td>"
                     f'<td class="titulo">{html.escape(titulos[i])}</td>{celdas}</tr>')
    cabecera = cabecera_agrupada(
        categorias, etiquetas,
        (2, '<th class="ordenable" tabindex="0">Género</th><th class="ordenable" tabindex="0">Título</th>'),
        ordenables=True)

    filas_medias = []
    for genero in generos_ordenados:
        celdas = "".join(celda_valor(medias[genero][d]) for d in range(len(etiquetas)))
        filas_medias.append(f"<tr><td>{html.escape(genero)}</td>{celdas}</tr>")
    cabecera_medias = cabecera_agrupada(categorias, etiquetas, (1, "<th>Género</th>"), ordenables=False)

    opciones = '<option value="">Todos</option>' + "".join(
        f'<option value="{html.escape(genero, quote=True)}">{html.escape(genero)}</option>'
        for genero in generos_ordenados)

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Descriptores de canciones según CLAP</title>
<style>{CSS}</style>
</head>
<body>
<header>
  <h1>Descriptores de canciones según CLAP</h1>
  <p class="explicacion">Cada valor es el percentil, dentro de esta colección, de la diferencia entre el coseno de la
  canción con ese descriptor y la media de sus cosenos con los {len(etiquetas)} descriptores (0 es la canción con menor
  diferencia y 100 la de mayor). Un valor alto indica que ese descriptor destaca en esa canción más que en las demás;
  no es una probabilidad. El coseno y la diferencia se ven al pasar el ratón por la celda. Modelo {html.escape(modelo)} (revisión {html.escape(revision[:8])}),
  {canciones_count} canciones, ventanas de audio de {segundos_ventana:g} s, generado el {fecha}.</p>
</header>
<main>
  <section>
    <h2>Percentil de cada canción</h2>
    <div class="controles">
      <label for="filtro-genero">Género</label>
      <select id="filtro-genero">{opciones}</select>
    </div>
    <div class="contenedor">
      <table id="tabla-principal">
        <thead>{cabecera}</thead>
        <tbody>
{chr(10).join(filas)}
        </tbody>
      </table>
    </div>
    <p class="nota">Pulsa la cabecera de una columna para ordenar por ella; al pulsarla otra vez se invierte el orden.</p>
  </section>
  <section>
    <h2>Media del percentil por género</h2>
    <div class="contenedor">
      <table id="tabla-medias">
        <thead>{cabecera_medias}</thead>
        <tbody>
{chr(10).join(filas_medias)}
        </tbody>
      </table>
    </div>
  </section>
</main>
<script>{JAVASCRIPT}</script>
</body>
</html>
"""


def mostrar_medias_por_genero(generos, categorias, etiquetas, similitudes):
    medias, generos_ordenados = medias_por_genero(generos, percentiles(similitudes))
    print("\nMedia del percentil por género")
    print(f"{'género':<20}" + "".join(f"{etiqueta[:11]:>12}" for etiqueta in etiquetas))
    for genero in generos_ordenados:
        print(f"{genero:<20}" + "".join(f"{valor:>12.1f}" for valor in medias[genero]))


def leer_argumentos():
    parser = argparse.ArgumentParser(
        description="Puntúa con CLAP cada canción de la caché local con descriptores de texto y genera "
                    "una tabla HTML con el percentil de cada canción para cada descriptor.")
    parser.add_argument("--dir-audio", type=Path, default=CARPETA_DEL_SCRIPT / ".cache" / "audio",
                        help="Caché local de los audios (por defecto tools\\.cache\\audio).")
    parser.add_argument("--dir-salida", type=Path, default=CARPETA_DEL_SCRIPT / "salida",
                        help="Directorio de salida (por defecto tools\\salida); los resultados van a su subcarpeta clap.")
    parser.add_argument("--segundos-ventana", type=float, default=10.0,
                        help="Duración en segundos de cada ventana de audio que entra en CLAP: de 1 a 10 "
                             "(por defecto 10). CLAP recorta al azar los audios de más de 10 s.")
    parser.add_argument("--lote", type=int, default=8, help="Ventanas por llamada al modelo (por defecto 8).")
    parser.add_argument("--limite", type=int,
                        help="Procesa solo las N primeras canciones en orden alfabético de ruta relativa.")
    parser.add_argument("--solo-html", action="store_true",
                        help="No calcula: regenera el HTML desde el descriptores.npz existente.")
    argumentos = parser.parse_args()
    if not SEGUNDOS_MINIMOS_VENTANA_FINAL <= argumentos.segundos_ventana <= 10:
        parser.error(f"--segundos-ventana debe estar entre {SEGUNDOS_MINIMOS_VENTANA_FINAL:g} y 10.")
    if argumentos.lote < 1 or (argumentos.limite is not None and argumentos.limite < 1):
        parser.error("--lote y --limite deben ser mayores que 0.")
    return argumentos


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    argumentos = leer_argumentos()
    directorio_clap = argumentos.dir_salida / "clap"
    ruta_npz = directorio_clap / "descriptores.npz"
    ruta_html = directorio_clap / "descriptores.html"

    if argumentos.solo_html:
        if not ruta_npz.is_file():
            sys.exit(f"No existe {ruta_npz}; ejecuta antes el cálculo sin --solo-html.")
        with np.load(ruta_npz) as datos:
            generos = [str(valor) for valor in datos["generos"]]
            titulos = [str(valor) for valor in datos["titulos"]]
            categorias = [str(valor) for valor in datos["categorias"]]
            etiquetas = [str(valor) for valor in datos["etiquetas"]]
            similitudes = datos["similitudes"]
            modelo_guardado = str(datos["modelo"])
            revision_guardada = str(datos["revision"])
            segundos_ventana = float(datos["segundos_ventana"])
        escribir_texto(ruta_html, generar_html(generos, titulos, categorias, etiquetas, similitudes,
                                               modelo_guardado, revision_guardada, segundos_ventana))
        print(f"Guardado {ruta_html}: {len(titulos)} canciones")
        mostrar_medias_por_genero(generos, categorias, etiquetas, similitudes)
        return

    rutas = listar_canciones(argumentos.dir_audio, argumentos.limite)
    if not rutas:
        sys.exit(f"No hay ficheros de audio en {argumentos.dir_audio}.")
    directorio_clap.mkdir(parents=True, exist_ok=True)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {dispositivo}, modelo {IDENTIFICADOR_MODELO} (revisión {REVISION_MODELO[:8]}), "
          f"{len(rutas)} canciones")

    inicio_total = time.perf_counter()
    procesador = ClapProcessor.from_pretrained(IDENTIFICADOR_MODELO, revision=REVISION_MODELO)
    modelo = ClapModel.from_pretrained(IDENTIFICADOR_MODELO, revision=REVISION_MODELO).to(dispositivo).eval()
    frecuencia_muestreo_hz = procesador.feature_extractor.sampling_rate

    categorias = [categoria for categoria, lista in DESCRIPTORES.items() for _ in lista]
    etiquetas = [etiqueta for lista in DESCRIPTORES.values() for etiqueta, _ in lista]
    adjetivos = [adjetivo for lista in DESCRIPTORES.values() for _, adjetivo in lista]
    embeddings_texto = calcular_embeddings_de_texto(modelo, procesador, dispositivo)

    embeddings_audio = []
    rutas_validas = []
    for indice, ruta_relativa in enumerate(rutas, 1):
        try:
            inicio = time.perf_counter()
            audio = leer_audio_mono(argumentos.dir_audio / ruta_relativa, frecuencia_muestreo_hz, dispositivo)
            embedding = calcular_embedding_audio(modelo, procesador, audio, argumentos.segundos_ventana,
                                                 argumentos.lote, dispositivo)
            segundos_calculo = time.perf_counter() - inicio
        except Exception as error:
            print(f"[{indice}/{len(rutas)}] ERROR en {ruta_relativa}: {type(error).__name__}: {error}")
            continue
        embeddings_audio.append(embedding)
        rutas_validas.append(ruta_relativa)
        print(f"[{indice}/{len(rutas)}] {ruta_relativa}: audio {len(audio) / frecuencia_muestreo_hz:.1f} s, "
              f"cálculo {segundos_calculo:.1f} s")
    if not rutas_validas:
        sys.exit("No hay ninguna canción con embedding.")

    embeddings_audio = np.stack(embeddings_audio).astype(np.float32)
    similitudes = (embeddings_audio @ embeddings_texto.T).astype(np.float32)
    generos = [genero_de_ruta(ruta) for ruta in rutas_validas]
    titulos = [titulo_de_ruta(ruta) for ruta in rutas_validas]

    guardar_npz(ruta_npz,
                embeddings_audio=embeddings_audio,
                similitudes=similitudes,
                rutas=np.array(rutas_validas),
                generos=np.array(generos),
                titulos=np.array(titulos),
                categorias=np.array(categorias),
                etiquetas=np.array(etiquetas),
                adjetivos=np.array(adjetivos),
                modelo=np.array(IDENTIFICADOR_MODELO),
                revision=np.array(REVISION_MODELO),
                segundos_ventana=np.float64(argumentos.segundos_ventana))
    escribir_texto(ruta_html, generar_html(generos, titulos, categorias, etiquetas, similitudes,
                                           IDENTIFICADOR_MODELO, REVISION_MODELO, argumentos.segundos_ventana))
    if dispositivo.type == "cuda":
        torch.cuda.synchronize()
    segundos_total = time.perf_counter() - inicio_total

    print(f"\nGuardado {ruta_npz} y {ruta_html}")
    print(f"Canciones: {len(rutas_validas)}, tiempo total {segundos_total:.1f} s")
    if dispositivo.type == "cuda":
        print(f"Memoria máxima de GPU asignada: {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
    print("\nCoseno mínimo y máximo por descriptor")
    for d, etiqueta in enumerate(etiquetas):
        print(f"  {etiqueta:<12}{similitudes[:, d].min():.4f}  {similitudes[:, d].max():.4f}")
    mostrar_medias_por_genero(generos, categorias, etiquetas, similitudes)


if __name__ == "__main__":
    main()
