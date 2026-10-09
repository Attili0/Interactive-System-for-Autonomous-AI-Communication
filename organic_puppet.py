import os
import sys
import time
import queue
import subprocess
import threading
import json
import numpy as np
import cv2
import sounddevice as sd
import soundfile as sf
import requests

import config

class OrganicRenderer:
    def __init__(self, width: int, height: int, fps: int):
        """
        Renderizador baseado em Deep Learning para animações orgânicas.
        Gerencia o repositório Wav2Lip, seus pesos, e a reprodução sincronizada
        do vídeo gerado com o áudio recebido do 'Cérebro'.
        """
        self.width = width
        self.height = height
        self.fps = fps
        
        self.is_generating = False
        self.is_playing = False
        self.current_video_cap = None
        self.audio_start_time = 0
        
        print("[OrganicRenderer] Instanciado. Preparando ecossistema Wav2Lip...")
        self.setup_wav2lip()
        
        # Carrega a imagem base do avatar
        avatar_path = str(getattr(config, "ORGANIC_AVATAR_IMAGE", "assets_fantoche/avatar_base.jpg"))
        if not os.path.exists(avatar_path):
            print("[OrganicRenderer] Imagem base do avatar não encontrada. Criando uma padrão de fallback...")
            os.makedirs(os.path.dirname(avatar_path), exist_ok=True)
            blank = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            cv2.putText(blank, "Avatar Organico", (50, self.height // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
            cv2.imwrite(avatar_path, blank)
        self.avatar_base = cv2.imread(avatar_path)

    def setup_wav2lip(self):
        """Verifica se o Wav2Lip e os pesos (S3FD e GAN) estão baixados."""
        wav2lip_dir = str(getattr(config, "WAV2LIP_DIR", "Wav2Lip"))
        checkpoint_path = str(getattr(config, "WAV2LIP_CHECKPOINT", os.path.join(wav2lip_dir, "checkpoints", "wav2lip_gan.pth")))
        s3fd_path = str(getattr(config, "WAV2LIP_S3FD_PATH", os.path.join(wav2lip_dir, "face_detection", "detection", "sfd", "s3fd.pth")))
        
        if not os.path.exists(wav2lip_dir):
            print("[OrganicRenderer] Clonando repositório Wav2Lip. Isso é feito apenas uma vez...")
            repo_url = getattr(config, "WAV2LIP_REPO_URL", "https://github.com/Rudrabha/Wav2Lip.git")
            subprocess.run(["git", "clone", repo_url, wav2lip_dir], check=True)
            
        def download_file(url, path, desc):
            if not os.path.exists(path):
                print(f"[OrganicRenderer] Baixando pesos {desc} para {path}...")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                
                headers = {}
                if getattr(config, "HF_TOKEN", None):
                    headers["Authorization"] = f"Bearer {config.HF_TOKEN}"

                response = requests.get(url, stream=True, headers=headers)
                response.raise_for_status()
                with open(path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"[OrganicRenderer] Download de {desc} concluído.")

        model_name = getattr(config, "ORGANIC_MODEL_NAME", "wav2lip_gan")
        weights_urls = getattr(config, "WAV2LIP_WEIGHTS_URLS", {})
        url_wav2lip = weights_urls.get(model_name, "https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/wav2lip_gan.pth")
        
        download_file(
            url_wav2lip, 
            checkpoint_path, 
            f"Wav2Lip ({model_name})"
        )
        download_file(
            getattr(config, "WAV2LIP_S3FD_URL", "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth"), 
            s3fd_path, 
            "S3FD Face Detection"
        )

    def generate_video(self, audio_path: str) -> str:
        """Invoca o script de inferência do Wav2Lip. Processa o arquivo .wav com a imagem avatar_base.jpg"""
        self.is_generating = True
        wav2lip_dir = str(getattr(config, "WAV2LIP_DIR", "Wav2Lip"))
        checkpoint_path = str(getattr(config, "WAV2LIP_CHECKPOINT", os.path.join(wav2lip_dir, "checkpoints", "wav2lip_gan.pth")))
        avatar_path = str(getattr(config, "ORGANIC_AVATAR_IMAGE", "assets_fantoche/avatar_base.jpg"))
        
        out_video = audio_path.replace(".wav", "_lipsync.mp4")
        
        cmd = [
            sys.executable, os.path.join(wav2lip_dir, "inference.py"),
            "--checkpoint_path", checkpoint_path,
            "--face", avatar_path,
            "--audio", audio_path,
            "--outfile", out_video,
            "--pads", "0", "10", "0", "0"
        ]
        
        try:
            print(f"[OrganicRenderer] Gerando vídeo usando Wav2Lip... Isso requer poder de GPU/CPU intenso.")
            # Wav2Lip pode quebrar se faltarem pacotes como librosa, torchvision no venv local.
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return out_video
        except subprocess.CalledProcessError as e:
            print(f"[OrganicRenderer] ERRO na geração de vídeo Wav2Lip: {e}")
            return ""
        finally:
            self.is_generating = False

    def play_audio_and_video(self, audio_path: str, video_path: str):
        """
        Inicia a reprodução do áudio em uma thread separada e seta o VideoCapture para a main thread exibir.
        """
        try:
            self.current_video_cap = cv2.VideoCapture(video_path)
            self.video_fps = self.current_video_cap.get(cv2.CAP_PROP_FPS)
            if self.video_fps <= 0 or np.isnan(self.video_fps):
                self.video_fps = 25.0
            
            self.audio_data, self.audio_sr = sf.read(audio_path)
            
            def audio_thread():
                self.is_playing = True
                self.audio_start_time = time.time()
                
                sd.play(self.audio_data, self.audio_sr)
                sd.wait() # Aguarda a fala terminar
                
                self.is_playing = False
                if self.current_video_cap:
                    self.current_video_cap.release()
                    self.current_video_cap = None
                
                # Cleanup (Deleta áudio e mp4 gerado)
                try:
                    if os.path.exists(audio_path): os.remove(audio_path)
                    if os.path.exists(video_path): os.remove(video_path)
                except Exception as e:
                    print(f"[OrganicRenderer] Erro ao limpar arquivos temporários: {e}")

            threading.Thread(target=audio_thread, daemon=True).start()
            
        except Exception as e:
            print(f"[OrganicRenderer] Erro ao carregar as mídias geradas: {e}")
            self.is_playing = False

    def run(self, stop_event, command_queue):
        """
        Loop principal de renderização orgânica via OpenCV. Ocupa a thread atual.
        - Exibe o frame da imagem base (ocioso)
        - Exibe um aviso quando o modelo está calculando
        - Exibe os frames do vídeo gerado em sincronia temporal perfeita com o áudio sendo reproduzido
        """
        print("[OrganicRenderer] Iniciando loop visual (OpenCV)...")
        window_name = "Isaac - Avatar Organico"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.width, self.height)

        generation_thread = None

        while not stop_event.is_set():
            try:
                comando = command_queue.get(block=False)
                if "audio_path" in comando and os.path.exists(comando["audio_path"]):
                    audio_path = comando["audio_path"]
                    
                    if not self.is_generating and not self.is_playing:
                        def generate_and_play():
                            video_path = self.generate_video(audio_path)
                            if video_path and os.path.exists(video_path):
                                self.play_audio_and_video(audio_path, video_path)
                            else:
                                if os.path.exists(audio_path): os.remove(audio_path)
                        
                        generation_thread = threading.Thread(target=generate_and_play, daemon=True)
                        generation_thread.start()
            except queue.Empty:
                pass

            # ======================== LÓGICA DE EXIBIÇÃO DA TELA ========================
            if self.is_playing and self.current_video_cap and self.current_video_cap.isOpened():
                # A tela de vídeo tenta parear com o timestamp real do áudio sendo tocado
                elapsed_time = time.time() - self.audio_start_time
                target_frame = int(elapsed_time * self.video_fps)
                
                current_frame_pos = int(self.current_video_cap.get(cv2.CAP_PROP_POS_FRAMES))
                
                if target_frame > current_frame_pos:
                    # Pula frames do opencv se a renderização estiver mais devagar que o tempo
                    while current_frame_pos < target_frame:
                        ret = self.current_video_cap.grab()
                        if not ret: break
                        current_frame_pos += 1
                    ret, frame = self.current_video_cap.retrieve()
                elif target_frame < current_frame_pos:
                    # Caso atípico: aguarda o tempo passar segurando o último frame lido
                    ret, frame = self.current_video_cap.read()
                else:
                    ret, frame = self.current_video_cap.read()

                if ret and frame is not None:
                    frame = cv2.resize(frame, (self.width, self.height))
                    cv2.imshow(window_name, frame)
                else:
                    cv2.imshow(window_name, cv2.resize(self.avatar_base, (self.width, self.height)))
            else:
                # Caso esteja gerando o vídeo, ou apenas aguardando falas
                display_frame = self.avatar_base.copy()
                display_frame = cv2.resize(display_frame, (self.width, self.height))
                
                if self.is_generating:
                    cv2.putText(
                        display_frame, 
                        "Gerando frames da boca (Wav2Lip Inference)...", 
                        (40, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (0, 255, 255), 
                        2
                    )
                                
                cv2.imshow(window_name, display_frame)

            # Limitador de FPS local
            if cv2.waitKey(int(1000 / self.fps)) & 0xFF == ord('q'):
                stop_event.set()
                break

        # Limpeza
        if self.current_video_cap:
            self.current_video_cap.release()
        sd.stop()
        cv2.destroyAllWindows()
        print("[OrganicRenderer] Finalizado. Fechando janelas.")
