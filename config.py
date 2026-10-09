"""
Configurações Globais do Projeto Isaac (UPA 2025)
Centraliza parâmetros de hardware, seleção de modelos de IA (CPU vs GPU),
caminhos de diretório e personalização da persona do assistente.
"""

import os
from pathlib import Path
try:
    from dotenv import load_dotenv
    # Carrega as variáveis de ambiente do arquivo .env, caso ele exista
    load_dotenv()
except ImportError:
    pass  # python-dotenv não instalado, ignora silenciosamente
# ==============================================================================
# AUTENTICAÇÃO E CHAVES DE API
# ==============================================================================
HF_TOKEN = os.getenv("HF_TOKEN")

# ==============================================================================
# CAMINHOS DO SISTEMA
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets_fantoche"
TTS_MODEL_DIR = BASE_DIR / "tts_model"
RHUBARB_DIR = BASE_DIR / "Rhubarb"
RHUBARB_EXE = RHUBARB_DIR / "rhubarb.exe"

# ==============================================================================
# DISPOSITIVOS DE EXECUÇÃO (CPU vs GPU / CUDA)
# ==============================================================================
# No notebook original, alguns modelos (como análise de emoções) foram definidos
# manualmente em "cpu" para evitar disputa de VRAM com o Whisper/InsightFace.
# Aqui você pode definir explicitamente onde cada modelo será executado.

INSIGHTFACE_DEVICE = "cuda"       # "cuda" (GPU NVIDIA) ou "cpu". Se "cuda", utiliza CUDAExecutionProvider com fallback automático para CPU.
EMOTION_DEVICE = "cpu"           # "cpu" ou "cuda". Configurado em "cpu" no notebook original para preservar VRAM para o modelo de fala e visão.
WHISPER_DEVICE = "cuda"           # "cuda" (recomendado para respostas em tempo real) ou "cpu".
WHISPER_COMPUTE_TYPE = "float16"  # "float16" (ideal para GPU), "int8" (ideal para CPU ou GPUs com pouca VRAM), "float32"
VAD_DEVICE = "cpu"               # "cpu" ou "cuda". Silero VAD consome menos de 1ms na CPU, dispensando uso de GPU.

# ==============================================================================
# SELEÇÃO DE MODELOS DE INTELIGÊNCIA ARTIFICIAL
# ==============================================================================

# --- 1. Detecção Facial e Extração de Embeddings (InsightFace) ---
INSIGHTFACE_MODEL = "antelopev2"
# Opções disponíveis para INSIGHTFACE_MODEL:
# - "buffalo_l"   # Padrão original. Alta precisão (ResNet-50 + SCRFD-500M), gera embeddings de 512D. Mais pesado em VRAM/RAM.
# - "buffalo_s"   # (Recomendado para otimização) Utiliza backbone MobileNet. Reduz o uso de memória em ~70% e roda com latência inferior a 15ms até em CPU, com perda insignificante de acurácia para distâncias de webcam.
# - "antelopev2"  # Versão mais recente e robusta que o buffalo_l para poses difíceis, rostos de perfil e variações severas de iluminação.
# - "buffalo_m"   # Equilíbrio intermediário entre buffalo_s e buffalo_l.

# --- 2. Classificação de Expressões e Emoções Faciais (EmotiEffLib / ONNX) ---
EMOTION_MODEL = "enet_b0_8_best_vgaf"
# Opções disponíveis para EMOTION_MODEL:
# - "enet_b0_8_best_vgaf"  # Padrão original (EmotiEffLib). Baseado em EfficientNet-B0 treinado no dataset VGAF. Rápido e leve em CPU.
# - "enet_b2_8"            # Versão EfficientNet-B2. Maior capacidade de representação e diferenciação de nuances sutis, ligeiramente mais pesado.
# - "enet_b0_7"            # Versão treinada em 7 emoções (sem 'desprezo'), reduzindo ambiguidades entre estado neutro e desgosto.

EMOTION_MODEL_URLS = {
    "enet_b0_8_best_vgaf": "https://github.com/HSE-asavchenko/face-emotion-recognition/raw/main/models/affectnet_emotions/enet_b0_8_best_vgaf.onnx",
    "enet_b2_8": "https://github.com/HSE-asavchenko/face-emotion-recognition/raw/main/models/affectnet_emotions/enet_b2_8.onnx",
    "enet_b0_7": "https://github.com/HSE-asavchenko/face-emotion-recognition/raw/main/models/affectnet_emotions/enet_b0_7.onnx"
}

# --- 3. Transcrição de Fala para Texto (STT / Faster-Whisper) ---
MODELO_WHISPER = "large-v3-turbo"
# Opções disponíveis para MODELO_WHISPER (Faster-Whisper):
# - "base"                 # Padrão atual. Muito rápido (~70-80% acurácia PT-BR). Bom para testes rápidos, mas pode errar em sotaques regionais.
# - "large-v3-turbo"       # (Altamente Recomendado) Lançado pela OpenAI em 2024. Quase a velocidade do 'base/small' com precisão superior ao 'medium' em português.
# - "tiny"                 # O mais leve de todos. Latência mínima em CPU, mas baixa precisão e alta taxa de erro em português.
# - "small"                # Bom equilíbrio entre custo computacional e acurácia (~85% em PT-BR).
# - "medium"               # Quase estado da arte (~92% em PT-BR), mas exige GPU dedicada para tempo real sem atraso perceptível.
# - "large-v3"             # Máxima precisão da OpenAI (~95% em PT-BR), porém exige GPU forte (6GB+ VRAM) e tem maior tempo de inferência.

# --- 4. Cérebro e Raciocínio (LLM / Ollama) ---
LLM_MODEL_NAME = "qwen3.5:0.6b"
# Configuração calibrada por footprint de VRAM e densidade de parâmetros:

# FAIXA 1: Ultra-Edge & Micro-SLMs (< 2 GB VRAM ou CPU Básica)
# - "granite4:350m"         # IBM (350M). Q4: ~250MB | FP16: ~700MB. Ideal para parsing de intenção e extração JSON atômica.
# - "qwen3.5:0.6b"         # Alibaba (600M). Q4: ~450MB | FP16: ~1.2GB. Excelente vocabulário e latência mínima para edge.

# FAIXA 2: Compactos & Borda (2 GB a 4 GB VRAM)
# - "qwen3.5:2b"           # Q4: ~1.5GB pesos (VRAM total: ~2.8GB). Suporte multimodal leve e raciocínio conversacional ágil.
# - "gemma4:e2b"           # Q4: ~1.6GB pesos (VRAM total: ~3.0GB). Google Edge com encoders integrados e system prompt.

# FAIXA 3: Intermediários / Sweet Spot de Eficiência (6 GB a 8 GB VRAM)
# - "bonsai"               # PrismLM (Ternário 1.58-bit). Destilado do Qwen 3.6 27B. Consumo: ~5.5 a 6 GB VRAM. Excelente baseline para inferência rápida sem perda drástica de sintaxe.
# - "bonsai-2"             # PrismLM (Binário 1-bit / Ternário otimizado). Destilado do Qwen 3.8 27B. Consumo: ~3.5 a 4 GB VRAM. Mantém as capacidades lógicas e o controle estrutural da geração 3.8 com footprint mínimo.
# - "granite4:3b"          # Q4: ~2.1GB pesos (VRAM total: ~4.2GB). Foco corporativo estrito, schemas JSON e tool-calling sem desvios.
# - "phi4-mini"            # Microsoft (3.8B). Q4: ~2.6GB pesos (VRAM total: ~5.0GB). Alto rigor lógico e validação sintática.
# - "gemma4:e4b"           # Q4: ~2.8GB pesos (VRAM total: ~5.2GB). 4B efetivos; diálogo fluido e excelente prosódia textual em PT-BR.
# - "qwen3.5:4b"           # Q4: ~2.7GB pesos (VRAM total: ~5.5GB). Se rodar em FP16 nativo, exige placa de 10-12GB (8GB só de pesos).

# FAIXA 4: Produção Avançada & Raciocínio Robusto (10 GB a 16 GB VRAM)
# - "granite4.2:8b"        # Q4: ~5.2GB pesos (VRAM total: ~8.5GB). Suporte a thinking tokens dinâmicos (/set think low/high).
# - "qwen3.5:9b"           # Q4: ~5.8GB pesos (VRAM total: ~9.5GB a 12GB dependendo do ctx). Janela ampla de 256k e robustez em PT-BR.
# - "gemma4:12b"           # Q4: ~7.6GB pesos (VRAM total: ~12GB a 14GB). Fronteira em compreensão semântica e respostas empáticas.
# - "phi4"                 # Microsoft (14B). Q4: ~9.2GB pesos (VRAM total: ~14GB a 16GB). Desempenho analítico denso de ponta.

# FAIXA 5: Alta Escala, MoE e Agentes Complexos (20 GB a 24 GB+ VRAM)
# - "gemma4:26b"           # MoE (3.8B ativos / 25.2B totais). Q4: ~16GB de pesos alocados. Rápido na inferência, exige >=20GB VRAM.
# - "qwen3.8:27b"          # Q4: ~17.5GB pesos (VRAM total: ~22GB a 24GB). Raciocínio de fronteira para fluxos agentic multi-etapas.

# --- 5. Síntese Neural de Voz (TTS / Piper) ---
VOICE_MODEL_NAME = "pt_BR-jeff-medium"
# Opções disponíveis na pasta tts_model/:
# - "pt_BR-jeff-medium"    # Padrão atual. Voz masculina clara, boa articulação e velocidade de síntese em milissegundos na CPU.
# - "pt_BR-cadu-medium"    # Voz masculina alternativa, tom mais jovem e informal.
# - "pt_BR-faber-medium"   # Voz masculina alternativa, tom mais formal e estilo locutor profissional.
# - "pt_BR-edresson-low"   # Modelo leve treinado pelo pesquisador Edresson. Menor consumo de processamento, ideal para hardware restrito.
VOICE_MODEL_PATH = TTS_MODEL_DIR / f"{VOICE_MODEL_NAME}.onnx"

PIPER_MODELS_URLS = {
    "pt_BR-jeff-medium": "https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_BR/jeff/medium/pt_BR-jeff-medium",
    "pt_BR-cadu-medium": "https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_BR/cadu/medium/pt_BR-cadu-medium",
    "pt_BR-faber-medium": "https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_BR/faber/medium/pt_BR-faber-medium",
    "pt_BR-edresson-low": "https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_BR/edresson/low/pt_BR-edresson-low"
}

# --- 6. Detecção de Atividade de Voz (VAD / Silero) ---
VAD_MODEL = "silero_vad_v5"
# Opções disponíveis para VAD_MODEL:
# - "silero_vad"      # Padrão da indústria (v4). Excelente separação de fala e silêncio, consome <1ms de CPU por chunk.
# - "silero_vad_v5"   # Versão 5 mais recente do Silero, com filtragem acústica aprimorada contra ruídos de fundo em feiras e eventos.

# ==============================================================================
# CONFIGURAÇÕES DO FANTOCHE (RENDERIZADOR)
# ==============================================================================
ANIMATION_MODE = "sprites"  
# Opções disponíveis para ANIMATION_MODE:
# - "sprites" : (Padrão) Usa o Rhubarb para sincronia labial e o Pygame para desenhar PNGs estáticos (alta performance, CPU friendly).
# - "organic" : Usa modelos de Deep Learning (Wav2Lip) para deformar imagens em tempo real sincronizadas com o áudio.

#               AVISO: Para usar o modo orgânico, instale os pacotes extras executando: pip install torch torchvision torchaudio librosa face_alignment numba

# Configurações do Modo Orgânico (Wav2Lip)
ORGANIC_AVATAR_IMAGE = BASE_DIR / "assets_fantoche" / "avatar_base.jpg"

ORGANIC_MODEL_NAME = "wav2lip_gan"
# Opções disponíveis para ORGANIC_MODEL_NAME:
# - "wav2lip_gan" : Usa o modelo GAN. Melhor qualidade visual (alta resolução) na região da boca, mas a sincronia labial é um pouco menos rígida.
# - "wav2lip"     : Usa o modelo padrão. Sincronia labial extremamente precisa e aderente ao áudio, mas a região da boca pode ficar levemente borrada.

WAV2LIP_DIR = BASE_DIR / "Wav2Lip"
WAV2LIP_CHECKPOINT = WAV2LIP_DIR / "checkpoints" / f"{ORGANIC_MODEL_NAME}.pth"
WAV2LIP_REPO_URL = "https://github.com/Rudrabha/Wav2Lip.git"

WAV2LIP_WEIGHTS_URLS = {
    "wav2lip_gan": "https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/wav2lip_gan.pth",
    "wav2lip": "https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/wav2lip.pth"
}

WAV2LIP_S3FD_PATH = WAV2LIP_DIR / "face_detection" / "detection" / "sfd" / "s3fd.pth"
WAV2LIP_S3FD_URL = "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth"

SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 810
PUPPET_TARGET_HEIGHT = 800
FPS = 30
BACKGROUND_COLOR = (200, 190, 200)

# Raio máximo de órbita das pupilas (efeito de olhar para o usuário)
PUPIL_ORBIT_RADIUS = 10

# Offsets relativos ao centro da cabeça para posicionamento dos assets
OFFSET_BOCA_Y = 181
OFFSET_SOBRANCELHAS_Y = -34
OFFSET_OLHO_X = 109
OFFSET_OLHO_Y = 38

# Mapeamento fonético: Saída do Rhubarb (A-X) -> Nomes dos arquivos de boca em assets_fantoche/bocas
VISEME_MAP = {
    "A": "a_e",              # Sons de vogais abertas (ex: "pá")
    "B": "neutro",           # Sons neutros (ex: "pedra")
    "C": "i_y",              # Vogais esticadas (ex: "fiz")
    "D": "g_k",              # Consoantes velares entreabertas (G, K)
    "E": "o",                # Vogais arredondadas (ex: "olho", "uva")
    "F": "f_v",              # Consoantes labiodentais (F, V)
    "G": "ch_j_sh_x",        # Sons sibilantes (CH, J, SH, X)
    "H": "c_d_n_q_s_t_z",    # Consoantes dentais/alveolares (T, D, N, etc.)
    "X": "b_m_p"             # Fechamento labial completo (M, B, P)
}

# Expressões de boca quando o fantoche está em repouso (não falando)
BOCA_EMOCAO_MAP = {
    "feliz": "feliz",
    "triste": "triste",
    "raiva": "raiva",
    "chocado": "chocado",
    "pensativo": "pensativo",
    "medo": "medo",
    "desgosto": "desgosto",
    "neutro": "neutro"
}

# Expressões das sobrancelhas para cada humor
SOBRANCELHA_EMOCAO_MAP = {
    "raiva": "raiva_desgosto",
    "desgosto": "raiva_desgosto",
    "triste": "triste_medo_choque",
    "medo": "triste_medo_choque",
    "chocado": "triste_medo_choque",
    "feliz": "feliz_pensativo",
    "pensativo": "feliz_pensativo",
    "neutro": "neutro"
}

# ==============================================================================
# CONFIGURAÇÕES DE ÁUDIO (MICROFONE, VAD E TRANSCRIÇÃO)
# ==============================================================================
IDIOMA_AUDIO = "pt"
TAXA_AMOSTRAGEM = 16000
CANAIS_AUDIO = 1
TAMANHO_BLOCO_SD = 512
VAD_THRESHOLD = 0.4
MIN_SPEECH_SEGUNDOS = 0.25
SPEECH_PAUSE = 700            # Milissegundos de silêncio para fechar o segmento
SPEECH_PAD = 100              # Milissegundos de margem antes/depois da fala

# ==============================================================================
# CONFIGURAÇÕES DO MODELO DE LINGUAGEM (PROMPT DO ISAC)
# ==============================================================================
SYSTEM_PROMPT = """
Você é ISAC, um assistente pessoal interativo criado para a UPA 2025.

Sua personalidade deve ser:
- Amigável
- Divertido
- Engraçado
- Levemente sarcástico

Sua aparência é de um personagem de desenho animado 2D.
Seu objetivo é interagir com o usuário de forma natural, gerar entretenimento e responder perguntas.
Seja engraçado e evite respostas prolixas ou textos muito longos.

### Entrada:
Você receberá informações visuais dos usuários na cena e a transcrição da fala do usuário.
Cada cadastro possui:
- "id": número no sistema
- "nome": nome do usuário
- "genero": gênero aparente
- "idade_aparente": estimativa de idade
- "emocao_atual": expressão facial detectada
- "bbox": coordenadas (x1, y1) do rosto

A fala do usuário é enviada em texto simples.

Use essas informações para:
- Personalizar sua resposta
- Adaptar o tom de acordo com a emoção detectada
- Fazer ISAC olhar na direção do usuário

### Saída:
Você DEVE responder SEMPRE com um **objeto JSON válido** contendo:
- "texto_resposta": (string) Sua fala para o usuário
- "emocao": (string) Emoção de ISAC ao responder. Escolha uma de: ["neutro", "feliz", "triste", "raiva", "chocado", "pensativo", "desgosto", "medo"]
- "target_coord": (array com 2 inteiros) Coordenadas [x, y] para onde olhar

### Exemplo de saída:
{
  "texto_resposta": "E aí pessoal! Que bom ver vocês aqui na UPA! Prontos para conhecer um pouco de inteligência artificial?",
  "emocao": "feliz",
  "target_coord": [960, 400]
}

### Regras adicionais:
- Nunca responda fora do formato JSON.
- Se não houver nome, use termos amigáveis como "amigo" ou "você aí".
- Nunca adicione explicações ou markdown antes ou depois do JSON.
"""

# ==============================================================================
# CONFIGURAÇÕES DE VISÃO COMPUTACIONAL (CÂMERA E PARÂMETROS)
# ==============================================================================

# Índice do dispositivo de captura de vídeo (Webcam)
CAMERA_INDEX = 0
# Explicação e valores comuns para CAMERA_INDEX:
# - 0    # Padrão. Primeira câmera do sistema (webcam integrada do notebook ou primeira webcam USB conectada).
# - 1    # Segunda câmera conectada (ex: webcam USB externa em tripé, câmera virtual do OBS, DroidCam ou placa de captura).
# - 2+   # Terceira câmera ou dispositivos adicionais de vídeo no sistema operacional.

# Quantidade de quadros (frames) ignorados entre cada inferência pesada de IA
FRAMES_TO_SKIP = 8
# Explicação e valores recomendados para FRAMES_TO_SKIP:
# - 8    # Padrão balanceado. Em uma câmera a 30 FPS, o modelo processa IA ~3,3 vezes por segundo (30 / (8+1)).
#        # Reduz o consumo de CPU/GPU em mais de 80%, evitando travamentos no Whisper, Pygame e LLM, mantendo o rastreamento fluido.
# - 4    # Maior reatividade (~6 detecções/segundo). Responde mais rápido a mudanças súbitas de expressão, mas consome mais CPU/GPU.
# - 2    # Alta frequência (~10 detecções/segundo). Indicado apenas se você possuir GPU potente dedicada exclusiva para visão.
# - 15   # Modo econômico (~2 detecções/segundo). Ideal para laptops ou máquinas sem GPU dedicada.

# Limiar (threshold) de distância de cosseno para reconhecimento e reidentificação facial
RECOGNITION_THRESHOLD = 0.5
# Explicação e calibração de RECOGNITION_THRESHOLD:
# - 0.50 # Padrão balanceado. Define o limite de distância de cosseno entre o vetor da face atual e o banco de usuários.
#        # Distâncias menores que esse valor são consideradas a MESMA pessoa; distâncias maiores cadastram um NOVO visitante.
# - 0.40 # Mais rigoroso / estrito. Evita que pessoas parecidas sejam confundidas, mas pequenas variações de ângulo,
#        # corte de cabelo ou iluminação podem fazer o robô achar que se trata de uma pessoa diferente.
# - 0.60 # Mais tolerante / permissivo. Reconhece o mesmo visitante mesmo com óculos, boné ou posições de perfil,
#        # porém aumenta o risco de unificar duas pessoas com fisionomia semelhante no mesmo cadastro.

EMOTIONS_DICT = {
    'Neutral': 'neutro',
    'Happiness': 'feliz',
    'Sadness': 'triste',
    'Surprise': 'chocado',
    'Fear': 'medo',
    'Disgust': 'desgosto',
    'Anger': 'raiva',
    'Contempt': 'desprezo'
}

GENDER_DICT = {
    0: "mulher",
    1: "homem"
}
