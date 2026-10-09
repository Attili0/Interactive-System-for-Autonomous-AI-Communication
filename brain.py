"""
Módulo de Inteligência (Cognição e Síntese) do Projeto Isaac
Compreende:
1. LLMPipeline: Consulta ao modelo de linguagem (Ollama / Phi-3) com contexto multimodal.
2. CerebroPipeline: Síntese de voz neural (Piper TTS) e cálculo de sincronização labial (Rhubarb).
"""

import os
import re
import json
import time
import queue
import tempfile
import threading
import subprocess
import requests
import traceback
from typing import Dict, Any, List

import ollama
import numpy as np
from scipy.io.wavfile import write as write_wav

import config

class LLMPipeline:
    """Consome dados dos sensores (visão e áudio), monta o contexto e consulta o LLM."""
    def __init__(self, model_name: str = config.LLM_MODEL_NAME):
        self.model_name = model_name
        self.conversation_history: List[Dict[str, str]] = [
            {"role": "system", "content": config.SYSTEM_PROMPT}
        ]
        self.estado_da_cena_atual: List[Dict[str, Any]] = []

    def _extrair_json(self, texto: str) -> Dict[str, Any]:
        """Extrai um objeto JSON válido da resposta do modelo, tratando blocos markdown."""
        texto = texto.strip()
        # Remove blocos ```json ... ``` se existirem
        if texto.startswith("```"):
            texto = re.sub(r"^```(?:json)?\s*", "", texto)
            texto = re.sub(r"\s*```$", "", texto)

        # Procura pelo primeiro objeto JSON {...}
        match = re.search(r"\{.*\}", texto, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(texto)

    def run(self, stop_event: threading.Event, sensor_data_queue: queue.Queue, llm_output_queue: queue.Queue):
        """Loop de monitoramento de eventos de sensores e acionamento do LLM."""
        print(f"[LLM] Inicializando pipeline do modelo '{self.model_name}'...")

        # Verifica/Baixa o modelo Ollama se não existir
        try:
            ollama.show(self.model_name)
        except ollama.ResponseError as e:
            if e.status_code == 404:
                print(f"[LLM] Modelo '{self.model_name}' não encontrado. Iniciando download (isso pode demorar)...")
                ollama.pull(self.model_name)
                print(f"[LLM] Modelo '{self.model_name}' baixado com sucesso.")
            else:
                print(f"[LLM] Aviso ao verificar modelo: {e}")

        while not stop_event.is_set():
            try:
                while not stop_event.is_set():
                    try:
                        sensor_data = sensor_data_queue.get(timeout=0.5)

                        # 1. Atualização do contexto visual
                        if sensor_data.get("source") == "vision":
                            self.estado_da_cena_atual = sensor_data.get("raw_detections", [])
                            print(f"[LLM] 👁️ Contexto visual atualizado ({len(self.estado_da_cena_atual)} pessoas).")
                            continue

                        # 2. Reação ao áudio falado
                        if sensor_data.get("source") == "audio":
                            texto_transcrito = sensor_data.get("data")
                            if not texto_transcrito:
                                continue

                            # Constrói o prompt enriquecido com contexto visual
                            prompt_usuario = (
                                "Contexto da Cena Atual (Pessoas na frente do robô):\n"
                                f"{json.dumps(self.estado_da_cena_atual, indent=2, ensure_ascii=False)}\n\n"
                                "Fala do Usuário:\n"
                                f'"{texto_transcrito}"'
                            )

                            print(f"\n[LLM] Enviando prompt ao modelo:\n---\n{prompt_usuario}\n---")
                            self.conversation_history.append({"role": "user", "content": prompt_usuario})

                            # Mantém o histórico enxuto (sistema + últimas 8 mensagens)
                            if len(self.conversation_history) > 10:
                                self.conversation_history = [self.conversation_history[0]] + self.conversation_history[-8:]

                            print("[LLM] 🧠 Gerando resposta...")
                            try:
                                response = ollama.chat(
                                    model=self.model_name,
                                    messages=self.conversation_history,
                                    format='json'
                                )
                                conteudo_resposta = response['message']['content']
                                print(f"[LLM] Resposta gerada: {conteudo_resposta}")

                                # Valida JSON antes de enfileirar
                                comando_dict = self._extrair_json(conteudo_resposta)
                                llm_output_json = json.dumps(comando_dict, ensure_ascii=False)

                                self.conversation_history.append({"role": "assistant", "content": llm_output_json})
                                llm_output_queue.put(llm_output_json)

                            except Exception as e_ollama:
                                print(f"[LLM] Erro ao consultar Ollama ({e_ollama}). Usando resposta de fallback.")
                                fallback_cmd = {
                                    "texto_resposta": "Desculpe, tive um pequeno lapso de conexão mental, mas estou te ouvindo!",
                                    "emocao": "pensativo",
                                    "target_coord": [config.SCREEN_WIDTH // 2, config.SCREEN_HEIGHT // 2]
                                }
                                llm_output_queue.put(json.dumps(fallback_cmd, ensure_ascii=False))

                    except queue.Empty:
                        continue

            except Exception as e:
                print(f"[LLM] Erro inesperado na pipeline LLM: {e}. Revivendo thread em 2s com memória de contexto intacta...")
                traceback.print_exc()
                time.sleep(2)

        print("[LLM] Pipeline finalizada.")

class CerebroPipeline:
    """Consome respostas do LLM, sintetiza áudio com Piper TTS e gera visemas labiais com Rhubarb."""
    def __init__(
        self,
        voice_path: str = str(config.VOICE_MODEL_PATH),
        rhubarb_path: str = str(config.RHUBARB_EXE)
    ):
        self.voice_path = voice_path
        self.rhubarb_path = rhubarb_path
        self.voice = None

    def load_models(self):
        """Carrega a voz neural do Piper (baixa se necessário) e valida o binário do Rhubarb."""
        print(f"[Cérebro] Carregando modelo de voz Piper ({self.voice_path})...")
        
        # Download automático da voz caso não exista
        if not os.path.exists(self.voice_path):
            base_name = os.path.basename(self.voice_path).replace(".onnx", "")
            if base_name in getattr(config, "PIPER_MODELS_URLS", {}):
                print(f"[Cérebro] Voz {base_name} não encontrada. Iniciando download...")
                os.makedirs(os.path.dirname(self.voice_path), exist_ok=True)
                url_base = config.PIPER_MODELS_URLS[base_name]
                
                headers = {}
                if getattr(config, "HF_TOKEN", None):
                    headers["Authorization"] = f"Bearer {config.HF_TOKEN}"

                # Download do arquivo .onnx
                with requests.get(f"{url_base}.onnx", stream=True, headers=headers) as r:
                    r.raise_for_status()
                    with open(self.voice_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                
                # Download do arquivo .onnx.json
                with requests.get(f"{url_base}.onnx.json", stream=True, headers=headers) as r:
                    r.raise_for_status()
                    with open(f"{self.voice_path}.json", 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                print(f"[Cérebro] Voz {base_name} baixada com sucesso!")
            else:
                raise FileNotFoundError(f"Modelo TTS não encontrado em: {self.voice_path} e nenhuma URL configurada para auto-download.")

        try:
            from piper.voice import PiperVoice
            self.voice = PiperVoice.load(self.voice_path)
            print("[Cérebro] Piper TTS carregado com sucesso.")
        except ImportError:
            raise ImportError("Biblioteca 'piper-tts' (ou 'piper') não encontrada. Instale com: pip install piper-tts")

        if getattr(config, "ANIMATION_MODE", "sprites") == "sprites":
            if not os.path.exists(self.rhubarb_path):
                raise FileNotFoundError(f"Executável do Rhubarb não encontrado em: {self.rhubarb_path}")
            print("[Cérebro] Executável do Rhubarb verificado.")
        else:
            print("[Cérebro] Modo de animação orgânica detectado. Rhubarb não será utilizado.")

    def run(self, stop_event: threading.Event, llm_output_queue: queue.Queue, actuator_command_queue: queue.Queue):
        """Loop de síntese e alinhamento fonético."""
        print("[Cérebro] Inicializando pipeline de síntese e sincronização...")

        # Utilizar um diretório de cache persistente em vez do tempfile autodeletável para gerenciar a rotação
        cache_dir = os.path.join(tempfile.gettempdir(), "isaac_tts_cache")
        os.makedirs(cache_dir, exist_ok=True)
        print(f"[Cérebro] Diretório de cache de áudios: {cache_dir}")

        while not stop_event.is_set():
            try:
                self.load_models()
            except Exception as e:
                print(f"[Cérebro] ERRO ao iniciar modelos: {e}. Tentando novamente em 5s...")
                time.sleep(5)
                continue

            counter = 0
            while not stop_event.is_set():
                try:
                    raw_command = llm_output_queue.get(timeout=0.5)
                    counter += 1

                    comando_info = json.loads(raw_command)
                    texto_para_falar = comando_info.get("texto_resposta") or comando_info.get("texto")
                    emocao = comando_info.get("emocao", "neutro")
                    target_coord = comando_info.get("target_coord")

                    comando_final = {
                        "emocao": emocao,
                        "target_coord": target_coord
                    }

                    if texto_para_falar and texto_para_falar.strip():
                        print(f"[Cérebro] 🗣️ Sintetizando áudio para: \"{texto_para_falar}\"")
                        
                        # SOLUÇÃO DISK LEAK: Rotação de arquivos com buffer de apenas 10 posições
                        base_filename = f"audio_buffer_{counter % 10}"
                        caminho_audio = os.path.join(cache_dir, f"{base_filename}.wav")
                        caminho_dialogo = os.path.join(cache_dir, f"{base_filename}.txt")
                        caminho_json_lipsync = os.path.join(cache_dir, f"{base_filename}.json")

                        # 1. Síntese de áudio com Piper
                        chunks = list(self.voice.synthesize(texto_para_falar))
                        audio_arrays = [chunk.audio_int16_array for chunk in chunks]
                        audio_array_final = np.concatenate(audio_arrays).astype(np.int16)
                        write_wav(caminho_audio, self.voice.config.sample_rate, audio_array_final)

                        # 2. Geração do arquivo de diálogo para apoiar o Rhubarb
                        with open(caminho_dialogo, 'w', encoding='utf-8') as f:
                            f.write(texto_para_falar)

                        # 3. Execução do Rhubarb Lip Sync (Apenas se for sprites)
                        if getattr(config, "ANIMATION_MODE", "sprites") == "sprites":
                            comando_rhubarb = [
                                self.rhubarb_path,
                                "-f", "json",
                                "-o", caminho_json_lipsync,
                                caminho_audio,
                                "-d", caminho_dialogo
                            ]
                            subprocess.run(comando_rhubarb, check=True, capture_output=True, text=True)

                            # 4. Leitura dos visemas sincronizados
                            if os.path.exists(caminho_json_lipsync):
                                with open(caminho_json_lipsync, 'r', encoding='utf-8') as f:
                                    lip_sync_data = json.load(f).get("mouthCues", [])
                                comando_final["lip_sync_data"] = lip_sync_data

                        comando_final["audio_path"] = caminho_audio
                    else:
                        print("[Cérebro] Comando não-verbal recebido.")

                    actuator_command_queue.put(comando_final)

                except queue.Empty:
                    continue
                except subprocess.CalledProcessError as e:
                    print(f"[Cérebro] Erro ao executar Rhubarb: {e.stderr}")
                except Exception as e:
                    print(f"[Cérebro] Erro na síntese/lip-sync: {e}")
                    break # Sai do loop interno e entra na tentativa de reviver

            if not stop_event.is_set():
                print(f"[Cérebro] Revivendo thread de síntese em 2s...")
                time.sleep(2)

        print("[Cérebro] Pipeline finalizada.")
