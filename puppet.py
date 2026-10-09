"""
Módulo de Renderização do Fantoche 2D do Projeto Isaac
Utiliza Pygame para compor o avatar 2D, sincronizar os visemas da boca com o áudio (lip-sync),
mover as sobrancelhas conforme a emoção e orientar as pupilas na direção do usuário.
"""

import os
import time
import queue
import threading
from typing import Dict, Any, Tuple

import pygame
import sounddevice as sd
from scipy.io.wavfile import read as read_wav

import config

def carregar_assets(assets_path: str, target_height: int) -> Dict[str, Any]:
    """Carrega todos os sprites visuais e redimensiona proporcionalmente à altura alvo."""
    print(f"[Fantoche] Carregando assets visuais de: {assets_path}")
    assets = {"bocas": {}, "sobrancelhas": {}, "pupilas": {}}

    caminho_cabeca = os.path.join(assets_path, "cabeça.png")
    if not os.path.exists(caminho_cabeca):
        raise FileNotFoundError(f"Asset da cabeça não encontrado em: {caminho_cabeca}")

    cabeca_original = pygame.image.load(caminho_cabeca).convert_alpha()
    original_w, original_h = cabeca_original.get_size()
    scale_factor = target_height / original_h
    target_width = int(original_w * scale_factor)

    assets['cabeça'] = pygame.transform.smoothscale(cabeca_original, (target_width, target_height))

    def scale_asset(path: str) -> pygame.Surface:
        img = pygame.image.load(path).convert_alpha()
        w, h = img.get_size()
        return pygame.transform.smoothscale(img, (int(w * scale_factor), int(h * scale_factor)))

    # Olhos / pupilas
    assets['pupilas']['esquerda'] = scale_asset(os.path.join(assets_path, "olho_esquerdo.png"))
    assets['pupilas']['direita'] = scale_asset(os.path.join(assets_path, "olho_direito.png"))

    # Bocas fonéticas e emocionais
    pasta_bocas = os.path.join(assets_path, "bocas")
    for file in os.listdir(pasta_bocas):
        if file.lower().endswith(".png"):
            nome = os.path.splitext(file)[0]
            chave = nome.replace("boca_", "") if nome.startswith("boca_") else nome
            assets["bocas"][chave] = scale_asset(os.path.join(pasta_bocas, file))

    # Sobrancelhas emocionais
    pasta_sobrancelhas = os.path.join(assets_path, "sobrancelhas")
    for file in os.listdir(pasta_sobrancelhas):
        if file.lower().endswith(".png"):
            nome = os.path.splitext(file)[0]
            chave = nome.replace("sobrancelha_", "") if nome.startswith("sobrancelha_") else nome
            assets["sobrancelhas"][chave] = scale_asset(os.path.join(pasta_sobrancelhas, file))

    print(f"[Fantoche] {len(assets['bocas'])} bocas e {len(assets['sobrancelhas'])} sobrancelhas carregadas.")
    return assets

class PuppetRenderer:
    """Renderizador gráfico 2D em Pygame com suporte a lip-sync e animação facial."""
    def __init__(
        self,
        assets_path: str = str(config.ASSETS_DIR),
        target_height: int = config.PUPPET_TARGET_HEIGHT,
        width: int = config.SCREEN_WIDTH,
        height: int = config.SCREEN_HEIGHT,
        fps: int = config.FPS
    ):
        self.assets_path = assets_path
        self.target_height = target_height
        self.width = width
        self.height = height
        self.fps = fps

    def run(self, stop_event: threading.Event, command_queue: queue.Queue):
        """Loop principal do Pygame."""
        print("[Fantoche] Inicializando Pygame...")
        try:
            pygame.init()
            screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Isaac - Assistente Inteligente UPA 2025 | Pressione 'q' para sair")
            clock = pygame.time.Clock()
            assets = carregar_assets(self.assets_path, self.target_height)
        except Exception as e:
            print(f"[Fantoche] ERRO CRÍTICO na inicialização do Pygame: {e}")
            stop_event.set()
            return

        estado_emocional = "neutro"
        playback_thread = None
        lip_sync_sequence = []
        audio_start_time = 0
        gaze_target = (self.width // 2, self.height // 2)

        # Posicionamento âncora dos componentes do rosto
        pos_cabeca = (self.width / 2, self.height / 2)
        pos_boca = (pos_cabeca[0], pos_cabeca[1] + config.OFFSET_BOCA_Y)
        pos_sobrancelhas = (pos_cabeca[0], pos_cabeca[1] + config.OFFSET_SOBRANCELHAS_Y)
        pos_olho_esq = (pos_cabeca[0] - config.OFFSET_OLHO_X, pos_cabeca[1] + config.OFFSET_OLHO_Y)
        pos_olho_dir = (pos_cabeca[0] + config.OFFSET_OLHO_X, pos_cabeca[1] + config.OFFSET_OLHO_Y)

        def play_audio_task(path: str):
            try:
                samplerate, data = read_wav(path)
                sd.play(data, samplerate)
                sd.wait()
            except Exception as e:
                print(f"[Fantoche Audio] Erro na reprodução de som: {e}")
            finally:
                # SOLUÇÃO PARA O LIMPEZA DE ÁUDIO ASSÍNCRONO
                try:
                    if os.path.exists(path):
                        os.remove(path)
                    json_path = path.replace(".wav", ".json")
                    if os.path.exists(json_path):
                        os.remove(json_path)
                    txt_path = path.replace(".wav", ".txt")
                    if os.path.exists(txt_path):
                        os.remove(txt_path)
                except Exception as e_cleanup:
                    print(f"[Fantoche Audio] Erro ao limpar arquivos temporários: {e_cleanup}")

        def blit_center(surf: pygame.Surface, pos: Tuple[float, float]):
            rect = surf.get_rect(center=pos)
            screen.blit(surf, rect)

        print("[Fantoche] Janela pronta e ativa.")
        while not stop_event.is_set():
            try:
                while not stop_event.is_set():
                    is_speaking = (playback_thread is not None and playback_thread.is_alive())

                    # Se não estiver falando, verifica se há novo comando na fila
                    if not is_speaking:
                        if lip_sync_sequence:
                            lip_sync_sequence.clear()
                        try:
                            comando = command_queue.get(block=False)
                            if "emocao" in comando:
                                estado_emocional = comando["emocao"]
                            if "target_coord" in comando and comando["target_coord"]:
                                gaze_target = comando["target_coord"]
                            if "audio_path" in comando and os.path.exists(comando["audio_path"]):
                                playback_thread = threading.Thread(
                                    target=play_audio_task,
                                    args=(comando["audio_path"],),
                                    name="AudioPlayback"
                                )
                                playback_thread.start()
                                audio_start_time = pygame.time.get_ticks()
                                lip_sync_sequence = comando.get("lip_sync_data", [])
                        except queue.Empty:
                            pass

                    # Captura eventos do teclado e fechamento de janela
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            stop_event.set()
                        elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                            stop_event.set()

                    # Recalcula estado de fala após verificação
                    is_speaking = (playback_thread is not None and playback_thread.is_alive())
                    boca_padrao = assets["bocas"].get("neutro", next(iter(assets["bocas"].values())))

                    # 1. Determina sprite da boca
                    if is_speaking and lip_sync_sequence:
                        current_time_s = (pygame.time.get_ticks() - audio_start_time) / 1000.0
                        boca_key = "neutro"
                        for cue in lip_sync_sequence:
                            if current_time_s >= cue['start']:
                                boca_key = config.VISEME_MAP.get(cue['value'], "neutro")
                            else:
                                break
                        boca_atual = assets["bocas"].get(boca_key, boca_padrao)
                    else:
                        boca_key = config.BOCA_EMOCAO_MAP.get(estado_emocional, "neutro")
                        boca_atual = assets["bocas"].get(boca_key, boca_padrao)

                    # 2. Determina sprite das sobrancelhas
                    sobrancelha_padrao = assets["sobrancelhas"].get("neutro", next(iter(assets["sobrancelhas"].values())))
                    sobrancelha_key = config.SOBRANCELHA_EMOCAO_MAP.get(estado_emocional, "neutro")
                    sobrancelha_atual = assets["sobrancelhas"].get(sobrancelha_key, sobrancelha_padrao)

                    # 3. Desenho no canvas do Pygame
                    screen.fill(config.BACKGROUND_COLOR)
                    blit_center(assets['cabeça'], pos_cabeca)
                    blit_center(sobrancelha_atual, pos_sobrancelhas)
                    blit_center(boca_atual, pos_boca)

                    # 4. Animação direcional das pupilas
                    pos_olhos = {'esquerda': pos_olho_esq, 'direita': pos_olho_dir}
                    for lado, pos_base in pos_olhos.items():
                        vx = gaze_target[0] - pos_base[0]
                        vy = gaze_target[1] - pos_base[1]
                        dist = (vx**2 + vy**2)**0.5

                        if dist > 0:
                            dx, dy = vx / dist, vy / dist
                            deslocamento = min(dist, config.PUPIL_ORBIT_RADIUS)
                            pos_pupila = (pos_base[0] + dx * deslocamento, pos_base[1] + dy * deslocamento)
                        else:
                            pos_pupila = pos_base

                        blit_center(assets['pupilas'][lado], pos_pupila)

                    pygame.display.flip()
                    clock.tick(self.fps)

            except Exception as e:
                print(f"[Fantoche] ERRO na renderização principal: {e}. Revivendo renderizador...")
                time.sleep(1)
        
        # Parar áudio caso esteja tocando ao fechar o app
        sd.stop()
        pygame.quit()
        print("[Fantoche] Pipeline de atuação finalizada.")
