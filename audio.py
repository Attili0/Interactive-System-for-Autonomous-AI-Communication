"""
Módulo de Áudio do Projeto Isaac
Responsável por captura de microfone em tempo real, detecção de atividade de voz (VAD)
com Silero VAD e transcrição de fala para texto (STT) via Faster-Whisper.
"""

import sys
import time
import queue
import threading
from typing import Optional

import numpy as np
import torch
import sounddevice as sd
from faster_whisper import WhisperModel

import config

class AudioPipeline:
    """Gerencia a captura contínua do microfone, segmentação por fala e transcrição."""
    def __init__(
        self,
        whisper_model: str = config.MODELO_WHISPER,
        language: str = config.IDIOMA_AUDIO,
        sample_rate: int = config.TAXA_AMOSTRAGEM,
        channels: int = config.CANAIS_AUDIO,
        block_size: int = config.TAMANHO_BLOCO_SD,
        vad_threshold: float = config.VAD_THRESHOLD,
        min_speech_s: float = config.MIN_SPEECH_SEGUNDOS,
        speech_pause_ms: int = config.SPEECH_PAUSE,
        speech_pad_ms: int = config.SPEECH_PAD
    ):
        self.whisper_model_name = whisper_model
        self.language = language
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_size = block_size
        self.vad_threshold = vad_threshold
        self.min_speech_s = min_speech_s
        self.speech_pause_ms = speech_pause_ms
        self.speech_pad_ms = speech_pad_ms

        self.vad_iterator = None
        self.modelo_whisper = None

    def load_models(self):
        """Carrega os modelos neurais de VAD (Silero) e ASR (Faster-Whisper)."""
        vad_device = getattr(config, "VAD_DEVICE", "cpu").lower()
        print(f"[Áudio] Carregando modelo Silero VAD no dispositivo '{vad_device.upper()}'...")
        try:
            model_vad, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False
            )
            if vad_device == "cuda" and torch.cuda.is_available():
                model_vad = model_vad.to("cuda")
            (_, _, _, VADIterator, _) = utils
            self.vad_iterator = VADIterator(
                model_vad,
                min_silence_duration_ms=self.speech_pause_ms,
                speech_pad_ms=self.speech_pad_ms,
                threshold=self.vad_threshold,
                sampling_rate=self.sample_rate
            )
            print("[Áudio] Silero VAD inicializado.")
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar Silero VAD: {e}")

        # Configura dispositivo para o Faster-Whisper conforme o config.py
        target_device = getattr(config, "WHISPER_DEVICE", "cuda").lower()
        if target_device == "cuda":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif target_device == "cpu":
            device = "cpu"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        target_compute = getattr(config, "WHISPER_COMPUTE_TYPE", "float16").lower()
        if target_compute == "auto":
            compute_type = "float16" if device == "cuda" else "int8"
        else:
            compute_type = target_compute

        print(f"[Áudio] Carregando Faster-Whisper '{self.whisper_model_name}' em {device.upper()} ({compute_type})...")
        try:
            self.modelo_whisper = WhisperModel(
                self.whisper_model_name,
                device=device,
                compute_type=compute_type
            )
            print(f"[Áudio] Faster-Whisper carregado.")
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar Faster-Whisper: {e}")

    def run(self, stop_event: threading.Event, sensor_data_queue: queue.Queue):
        """Inicia a captura de microfone e processamento assíncrono."""
        print("[Áudio] Pipeline de áudio iniciando...")
        try:
            self.load_models()
        except Exception as e:
            print(f"[Áudio] ERRO CRÍTICO ao carregar modelos de áudio: {e}")
            stop_event.set()
            return

        raw_audio_queue = queue.Queue()
        speech_segments_queue = queue.Queue()

        def audio_callback(indata, frames, time_info, status):
            if status:
                print(f"[Áudio] Status do stream: {status}", file=sys.stderr)
            raw_audio_queue.put(indata.copy())

        def processador_vad():
            """Worker 1: Consome blocos de áudio e detecta início/fim de sentenças faladas."""
            speech_chunks_buffer = []
            while not stop_event.is_set():
                try:
                    while not stop_event.is_set():
                        try:
                            raw_audio_chunk = raw_audio_queue.get(timeout=0.5)
                            speech_dict = self.vad_iterator(
                                torch.from_numpy(raw_audio_chunk.flatten()),
                                return_seconds=False
                            )

                            if speech_dict and 'start' in speech_dict:
                                speech_chunks_buffer.append(raw_audio_chunk)
                                print("[Áudio] 🎙️ Início de fala detectado...")
                            elif speech_dict and 'end' in speech_dict:
                                if speech_chunks_buffer:
                                    full_segment = np.concatenate(speech_chunks_buffer)
                                    speech_chunks_buffer.clear()
                                    duracao = len(full_segment) / self.sample_rate
                                    print(f"[Áudio] Fim de fala detectado ({duracao:.2f}s).")
                                    if duracao >= self.min_speech_s:
                                        speech_segments_queue.put(full_segment)
                            elif speech_chunks_buffer:
                                speech_chunks_buffer.append(raw_audio_chunk)

                        except queue.Empty:
                            if speech_chunks_buffer:
                                full_segment = np.concatenate(speech_chunks_buffer)
                                speech_chunks_buffer.clear()
                                duracao = len(full_segment) / self.sample_rate
                                if duracao >= self.min_speech_s:
                                    speech_segments_queue.put(full_segment)
                            continue
                except Exception as e:
                    print(f"[Áudio] Erro fatal na thread VAD: {e}. Revivendo em 1s...")
                    time.sleep(1)

        def processador_transcricao():
            """Worker 2: Transcreve os segmentos de fala para texto."""
            while not stop_event.is_set():
                try:
                    while not stop_event.is_set():
                        try:
                            segmento_fala = speech_segments_queue.get(timeout=0.5)
                            segmentos, _ = self.modelo_whisper.transcribe(
                                segmento_fala.flatten(),
                                beam_size=5,
                                language=self.language
                            )
                            texto_transcrito = "".join(seg.text for seg in segmentos).strip()

                            if texto_transcrito:
                                print(f"[Áudio] 💬 Transcrição: \"{texto_transcrito}\"")
                                sensor_data = {
                                    "source": "audio",
                                    "data": texto_transcrito,
                                    "timestamp": time.time()
                                }
                                sensor_data_queue.put(sensor_data)

                        except queue.Empty:
                            continue
                except Exception as e:
                    print(f"[Áudio] Erro fatal na transcrição: {e}. Revivendo thread STT em 1s...")
                    time.sleep(1)

        thread_vad = threading.Thread(target=processador_vad, name="AudioVAD")
        thread_transcricao = threading.Thread(target=processador_transcricao, name="AudioSTT")
        thread_vad.start()
        thread_transcricao.start()

        stream = None
        try:
            stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                channels=self.channels,
                dtype='float32',
                callback=audio_callback
            )
            stream.start()
            print("[Áudio] Microfone ativo e aguardando fala.")
            stop_event.wait()
        except Exception as e:
            print(f"[Áudio] Erro na captura de áudio: {e}")
            stop_event.set()
        finally:
            if stream is not None:
                stream.stop()
                stream.close()
            thread_vad.join(timeout=1.0)
            thread_transcricao.join(timeout=1.0)
            print("[Áudio] Pipeline de áudio finalizada.")
