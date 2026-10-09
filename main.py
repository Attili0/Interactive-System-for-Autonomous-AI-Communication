"""
Ponto de Entrada Principal (O Maestro) - Projeto Isaac (UPA 2025)
Orquestra as pipelines multithread de Visão, Áudio, Cognição (LLM), Síntese (TTS/Lip-Sync)
e Atuação Gráfica (Pygame).
"""

import sys
import argparse
import queue
import threading
import time

import config

def check_camera(index: int) -> bool:
    """Testa rapidamente se a câmera está disponível e retornando frames."""
    try:
        import cv2
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            return False
        ret, _ = cap.read()
        cap.release()
        return ret
    except Exception:
        return False

def parse_args():
    parser = argparse.ArgumentParser(description="Projeto Isaac - Assistente Multimodal UPA 2025")
    parser.add_argument(
        "--no-vision",
        action="store_true",
        help="Executa em modo simplificado sem a pipeline de visão computacional (apenas áudio + LLM + fantoche)."
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default=config.MODELO_WHISPER,
        help=f"Modelo do Faster-Whisper a utilizar (padrão: {config.MODELO_WHISPER})."
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=config.LLM_MODEL_NAME,
        help=f"Modelo do Ollama a utilizar (padrão: {config.LLM_MODEL_NAME})."
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=config.VOICE_MODEL_NAME,
        help=f"Nome da voz neural do Piper em tts_model/ (padrão: {config.VOICE_MODEL_NAME})."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=config.CAMERA_INDEX,
        help=f"Índice da câmera a utilizar (padrão: {config.CAMERA_INDEX})."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    print("=" * 70)
    print("      PROJETO ISAAC - ASSISTENTE INTELIGENTE MULTIMODAL (UPA 2025)")
    print("=" * 70)

    # Verifica câmera antes de continuar
    if not args.no_vision:
        print("[Sistema] Verificando conexão com a câmera...")
        if not check_camera(args.camera):
            print("[Sistema] Aviso: Câmera não detectada ou indisponível. Desativando pipeline de visão automaticamente.")
            args.no_vision = True

    print(f"Modo Visão: {'DESATIVADO (--no-vision ou sem câmera)' if args.no_vision else f'ATIVO (Câmera {args.camera})'}")
    print(f"Modelo STT (Whisper): {args.whisper_model}")
    print(f"Modelo LLM (Ollama): {args.llm_model}")
    print(f"Voz Neural (Piper): {args.voice}")
    print("=" * 70)

    # 1. Importação dos Módulos com Verificação de Dependências
    try:
        from audio import AudioPipeline
        from brain import LLMPipeline, CerebroPipeline
        if getattr(config, "ANIMATION_MODE", "sprites") == "sprites":
            from puppet import PuppetRenderer
        else:
            from organic_puppet import OrganicRenderer
    except ImportError as e:
        print(f"\n[ERRO DE DEPENDÊNCIA] Não foi possível importar um dos módulos principais: {e}")
        print("Certifique-se de instalar os requisitos com: pip install -r requirements.txt\n")
        sys.exit(1)

    # 2. Eventos e Filas de Comunicação Assíncrona
    stop_event = threading.Event()
    sensor_data_queue = queue.Queue()
    llm_output_queue = queue.Queue()
    actuator_command_queue = queue.Queue()

    # 3. Instanciação dos Módulos
    audio_pipeline = AudioPipeline(whisper_model=args.whisper_model)
    llm_pipeline = LLMPipeline(model_name=args.llm_model)

    caminho_voz = config.TTS_MODEL_DIR / f"{args.voice}.onnx"
    cerebro_pipeline = CerebroPipeline(
        voice_path=str(caminho_voz),
        rhubarb_path=str(config.RHUBARB_EXE)
    )

    if getattr(config, "ANIMATION_MODE", "sprites") == "sprites":
        renderer = PuppetRenderer(
            assets_path=str(config.ASSETS_DIR),
            target_height=config.PUPPET_TARGET_HEIGHT,
            width=config.SCREEN_WIDTH,
            height=config.SCREEN_HEIGHT,
            fps=config.FPS
        )
    else:
        renderer = OrganicRenderer(
            width=config.SCREEN_WIDTH,
            height=config.SCREEN_HEIGHT,
            fps=config.FPS
        )

    # 3. Threads em Segundo Plano
    threads = []

    # Thread de Áudio (Microfone + VAD + Whisper)
    audio_thread = threading.Thread(
        target=audio_pipeline.run,
        args=(stop_event, sensor_data_queue),
        name="AudioSensor",
        daemon=True
    )
    threads.append(audio_thread)

    # Thread de Visão (Webcam + Face + Emoções)
    if not args.no_vision:
        try:
            from vision import VisionPipeline
            vision_pipeline = VisionPipeline(camera_index=args.camera)
            vision_thread = threading.Thread(
                target=vision_pipeline.run,
                args=(stop_event, sensor_data_queue),
                name="VisionSensor",
                daemon=True
            )
            threads.append(vision_thread)
        except Exception as e_vision:
            print(f"[Aviso] Não foi possível carregar o módulo de visão: {e_vision}")
            print("[Aviso] Continuando em modo de apenas áudio.")

    # Thread de Cognição (Ollama LLM)
    llm_thread = threading.Thread(
        target=llm_pipeline.run,
        args=(stop_event, sensor_data_queue, llm_output_queue),
        name="CognicaoLLM",
        daemon=True
    )
    threads.append(llm_thread)

    # Thread de Síntese (Piper TTS + Rhubarb Lip-Sync)
    cerebro_thread = threading.Thread(
        target=cerebro_pipeline.run,
        args=(stop_event, llm_output_queue, actuator_command_queue),
        name="SinteseCerebro",
        daemon=True
    )
    threads.append(cerebro_thread)

    # 4. Inicialização das Threads
    print("[Maestro] Iniciando módulos de segundo plano...")
    for t in threads:
        t.start()

    print("\n[Maestro] Sistema em execução!")
    print(">> Para sair: feche a janela do fantoche, pressione 'q' ou aperte Ctrl+C no terminal.\n")

    # 5. Execução do Renderizador Gráfico (Pygame na Thread Principal)
    try:
        renderer.run(stop_event, actuator_command_queue)
    except KeyboardInterrupt:
        print("\n[Maestro] Interrupção por teclado (Ctrl+C). Finalizando...")
    finally:
        stop_event.set()
        print("[Maestro] Aguardando encerramento das threads...")
        for t in threads:
            t.join(timeout=2.0)
        print("[Maestro] Todos os sistemas foram desligados com sucesso. Até logo!")

if __name__ == "__main__":
    main()
