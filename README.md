# Projeto Isaac - Interactive System for Autonomous AI Communication

O **Projeto Isaac** é um avatar/fantoche virtual 2D interativo com Inteligência Artificial multimodal projetado para interação detalhada com usuário em tempo real.

O sistema funciona em tempo real e de forma **100% local e offline**, combinando visão computacional, audição, modelo de linguagem, síntese de voz e sincronização labial fonética (*lip-sync*).

---

## 🏛️ Arquitetura do Sistema

O assistente adota um modelo produtor-consumidor multithread ("O Maestro") conectado por filas assíncronas (`queue.Queue`):

```
 📷 [Câmera]                  🎙️ [Microfone]
(InsightFace + EmotiEffLib)   (Silero VAD + Faster-Whisper)
       │                                │
       └──────────────┬─────────────────┘
                      ▼
             sensor_data_queue
                      │
                      ▼
               🧠 [LLMPipeline] (Ollama / Phi-3)
                      │
                      ▼
               llm_output_queue
                      │
                      ▼
           🗣️ [CerebroPipeline] (Piper TTS + Rhubarb Lip-Sync)
                      │
                      ▼
            actuator_command_queue
                      │
                      ├────────────────────────────────────┐
                      ▼                                    ▼
            🎭 [PuppetRenderer]                  🎬 [OrganicRenderer]
          (Pygame 2D Lip-Sync)                  (Wav2Lip + OpenCV)
          ANIMATION_MODE = "sprites"            ANIMATION_MODE = "organic"
```

1. **Visão (`vision.py`)**: Identifica pessoas por embeddings faciais, estima idade/gênero e detecta expressões emocionais.
2. **Audição (`audio.py`)**: Detecta atividade de fala (Silero VAD) e transcreve a conversa em português (Faster-Whisper).
3. **Cérebro (`brain.py`)**:
   - `LLMPipeline`: Gera respostas na persona do Isaac levando em conta a fala e o contexto visual.
   - `CerebroPipeline`: Sintetiza a fala via voz neural (Piper TTS) e extrai os visemas com o Rhubarb (apenas modo sprites).
4. **Fantoche (`puppet.py` e `organic_puppet.py`)**: 
   - Modo `sprites`: Renderiza a cabeça, sincroniza a troca de bocas em tempo real com o áudio, altera as sobrancelhas pelo humor e move as pupilas na direção da pessoa.
   - Modo `organic`: Usa o motor Wav2Lip para distorcer um avatar base usando Deep Learning sincronizado em tempo real.
5. **Configurações (`config.py`)**: Centraliza parâmetros, resoluções, caminhos e prompts.
6. **Orquestrador (`main.py`)**: Gerencia o ciclo de vida de todas as threads e a interface gráfica.

---

## 📂 Estrutura de Arquivos

```text
isaac/
├── main.py                     # Ponto de entrada e orquestrador principal
├── config.py                   # Configurações globais, caminhos e chaves
├── vision.py                   # Visão computacional (InsightFace + EmotiEffLib)
├── audio.py                    # Captação de áudio, VAD e transcrição (Faster-Whisper)
├── brain.py                    # Cognição (LLM Ollama) e síntese de fala (Piper + Rhubarb)
├── puppet.py                   # Renderizador 2D em Pygame com lip-sync e animação facial
├── organic_puppet.py           # Renderizador baseado em Deep Learning (Wav2Lip)
├── requirements.txt            # Dependências Python
├── README.md                   # Documentação do projeto
├── .env                        # Arquivo para chaves locais (como HF_TOKEN)
├── UPA2025-2.ipynb             # Notebook original com histórico completo (ignorado no git)
├── assets_fantoche/            # Sprites do boneco 2D (cabeça, olhos, bocas, sobrancelhas)
├── tts_model/                  # Modelos neurais ONNX do Piper em PT-BR (jeff, cadu, faber, edresson)
└── Rhubarb/                    # Binário do Rhubarb Lip Sync e modelos acústicos (res/sphinx)
```

---

## 🚀 Como Executar

### 1. Pré-requisitos
* Python 3.9+ ou 3.10+
* [Ollama](https://ollama.com/) instalado e rodando com o modelo `phi3:mini`:
  ```bash
  ollama run phi3:mini
  ```
* Microfone e Webcam conectados.

### 2. Configurações de API e Downloads (HuggingFace)
O assistente baixa modelos hospedados no HuggingFace automaticamente. Para evitar limites de taxa de download (rate limit) ou conseguir baixar modelos de acesso restrito, configure seu token de acesso:
1. Crie um arquivo `.env` na raiz do projeto (`isaac/`).
2. Adicione a seguinte linha:
   ```env
   HF_TOKEN=seu_token_hf_aqui
   ```

### 3. Instalação das Dependências Básicas
```bash
pip install -r requirements.txt
```

### 4. Configuração do Modo Orgânico (Opcional)
Se você for usar a animação `organic` (Wav2Lip / Deep Learning), será necessário instalar as dependências de IA:
```bash
pip install torch torchvision torchaudio librosa face_alignment numba
```
Alerar o modo no `config.py`: `ANIMATION_MODE = "organic"`

### 5. Execução

#### Modo Completo (Multimodal: Visão + Áudio + Fantoche):
```bash
python main.py
```

#### Modo Simplificado (Apenas Áudio + Fantoche, sem Webcam):
Ideal para testes rápidos ou ambientes sem GPU/câmera:
```bash
python main.py --no-vision
```

#### Opções Avançadas de Linha de Comando:
```bash
python main.py --help
# Opções:
#   --no-vision                Desativa a câmera
#   --whisper-model tiny|base  Escolhe o tamanho do modelo Whisper
#   --llm-model phi3:mini      Escolhe o modelo Ollama
#   --voice pt_BR-jeff-medium  Escolhe a voz do Piper
#   --camera 0                 Índice da webcam
```

Para encerrar o programa a qualquer momento, pressione **`q`** ou feche a janela do fantoche.
