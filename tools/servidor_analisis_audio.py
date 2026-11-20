"""Arranque: python servidor_analisis_audio.py (escucha en 127.0.0.1:8000).
Al arrancar carga MERT-v1-330M y CLAP (laion/clap-htsat-unfused), y calcula una vez los embeddings de
texto de los descriptores de DESCRIPTORES (analizar_canciones_clap.py).

GET /embedding?url=URL_DEL_AUDIO devuelve en JSON el embedding MERT de la canción, de forma [25, 1024].
GET /descriptores?url=URL_DEL_AUDIO devuelve un diccionario categoría -> etiqueta -> número, en el orden de
DESCRIPTORES. El número es el coseno entre el embedding CLAP de la canción y el del descriptor, menos la media
de los cosenos de la canción con todos los descriptores. Positivo: el descriptor encaja con la canción mejor
que la media de sus descriptores; negativo: peor."""
import io

import requests
import soundfile
import torch
import torchaudio.functional
import uvicorn
from fastapi import FastAPI
from transformers import AutoModel, ClapModel, ClapProcessor, Wav2Vec2FeatureExtractor

from analizar_canciones_clap import (DESCRIPTORES, IDENTIFICADOR_MODELO, REVISION_MODELO,
                                     calcular_embedding_audio, calcular_embeddings_de_texto,
                                     similitudes_centradas)

# trust_remote_code ejecuta el código del repositorio de Hugging Face: se fija el commit.
IDENTIFICADOR = "m-a-p/MERT-v1-330M"
REVISION = "5240c2708a5acaee1007f43fb9735c7dcd0b78c9"
SEGUNDOS_VENTANA_CLAP = 10.0
LOTE_CLAP = 8
dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
extractor = Wav2Vec2FeatureExtractor.from_pretrained(IDENTIFICADOR, revision=REVISION, trust_remote_code=True)
modelo = AutoModel.from_pretrained(IDENTIFICADOR, revision=REVISION, trust_remote_code=True).to(dispositivo).eval()
hz = extractor.sampling_rate
procesador_clap = ClapProcessor.from_pretrained(IDENTIFICADOR_MODELO, revision=REVISION_MODELO)
modelo_clap = ClapModel.from_pretrained(IDENTIFICADOR_MODELO, revision=REVISION_MODELO).to(dispositivo).eval()
hz_clap = procesador_clap.feature_extractor.sampling_rate
embeddings_texto = calcular_embeddings_de_texto(modelo_clap, procesador_clap, dispositivo)  # [D, 512]
app = FastAPI()


def descargar_audio(url):
    """Descarga la URL y la decodifica sin remuestrear. Devuelve (float32 [muestras, canales], hz del fichero)."""
    respuesta = requests.get(url)
    respuesta.raise_for_status()
    return soundfile.read(io.BytesIO(respuesta.content), dtype="float32", always_2d=True)


def a_mono_con_frecuencia(datos, hz_original, hz_destino):
    """Promedia los canales y remuestrea a hz_destino. Devuelve float32 [muestras]."""
    audio = torch.from_numpy(datos.mean(axis=1)).to(dispositivo)
    return torchaudio.functional.resample(audio, hz_original, hz_destino).cpu().numpy()


@app.get("/embedding")
def embedding(url: str):
    datos, hz_original = descargar_audio(url)
    audio = a_mono_con_frecuencia(datos, hz_original, hz)
    # Ventanas de 10 s; la última se conserva si dura al menos 1 s.
    ventanas = [audio[inicio:inicio + 10 * hz] for inicio in range(0, len(audio) - hz + 1, 10 * hz)]
    medias = []
    with torch.inference_mode():
        for ventana in ventanas:
            entrada = extractor(ventana, sampling_rate=hz, return_tensors="pt").input_values.to(dispositivo)
            capas = modelo(input_values=entrada, output_hidden_states=True).hidden_states
            medias.append(torch.stack([capa.mean(dim=1)[0] for capa in capas]))
    return torch.stack(medias).mean(dim=0).cpu().tolist()


@app.get("/descriptores")
def descriptores(url: str):
    datos, hz_original = descargar_audio(url)
    audio = a_mono_con_frecuencia(datos, hz_original, hz_clap)
    embedding_audio = calcular_embedding_audio(modelo_clap, procesador_clap, audio,
                                               SEGUNDOS_VENTANA_CLAP, LOTE_CLAP, dispositivo)
    cosenos = (embeddings_texto @ embedding_audio)[None, :]  # [1, D]
    diferencias = similitudes_centradas(cosenos)[0]
    resultado = {}
    indice = 0
    for categoria, lista in DESCRIPTORES.items():
        resultado[categoria] = {}
        for etiqueta, _ in lista:
            resultado[categoria][etiqueta] = float(diferencias[indice])
            indice += 1
    return resultado


if __name__ == "__main__":
    uvicorn.run(app)
