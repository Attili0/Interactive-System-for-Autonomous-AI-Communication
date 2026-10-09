"""
Módulo de Visão Computacional do Projeto Isaac
Responsável por captura de webcam, detecção de faces, extração de embeddings,
reconhecimento de visitantes e análise de emoções em tempo real.
"""

import time
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from scipy.spatial.distance import cosine

import config

class User:
    """Armazena o perfil e o estado histórico de cada pessoa detectada."""
    def __init__(self, user_id: int, embedding: np.ndarray, gender: Optional[str] = None, age: Optional[int] = None):
        self.id = user_id
        self.name = f"Pessoa_{user_id}"
        self.embedding = embedding
        self.gender = gender
        self.age = age
        self.emotions: List[str] = []
        self.last_seen = time.time()
        self.last_bbox: Optional[Tuple[int, int, int, int]] = None

    def update(self, bbox: Tuple[int, int, int, int], age: Optional[int], gender: Optional[str], emotion: str):
        """Atualiza informações da pessoa no frame atual."""
        self.last_seen = time.time()
        self.last_bbox = bbox
        self.age = age
        self.gender = gender

        # Adiciona a emoção se for diferente da última registrada
        if not self.emotions or self.emotions[-1] != emotion:
            self.emotions.append(emotion)
            if len(self.emotions) > 5:
                self.emotions.pop(0)

class VisionPipeline:
    """Gerencia a captura da câmera, modelos de IA visual e publicação de eventos de cena."""
    def __init__(
        self,
        camera_index: int = config.CAMERA_INDEX,
        recognition_threshold: float = config.RECOGNITION_THRESHOLD,
        frames_to_skip: int = config.FRAMES_TO_SKIP
    ):
        self.camera_index = camera_index
        self.recognition_threshold = recognition_threshold
        self.frames_to_skip = frames_to_skip
        self.face_app = None
        self.emo_model = None

    def load_models(self):
        """Carrega modelos InsightFace e EmotiEffLib respeitando as configurações de dispositivo."""
        target_device = getattr(config, "INSIGHTFACE_DEVICE", "cuda").lower()
        print(f"[Visão] Carregando modelo InsightFace ({config.INSIGHTFACE_MODEL}) no dispositivo '{target_device.upper()}'...")
        try:
            from insightface.app import FaceAnalysis
            if target_device == "cuda":
                # Tenta carregar com CUDA; se falhar ou não houver GPU compatível, usa CPU
                try:
                    self.face_app = FaceAnalysis(name=config.INSIGHTFACE_MODEL, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
                    self.face_app.prepare(ctx_id=0, det_size=(640, 640))
                except Exception as e_cuda:
                    print(f"[Visão] Falha ao iniciar InsightFace na GPU ({e_cuda}). Alternando para CPU...")
                    self.face_app = FaceAnalysis(name=config.INSIGHTFACE_MODEL, providers=['CPUExecutionProvider'])
                    self.face_app.prepare(ctx_id=-1, det_size=(640, 640))
            else:
                self.face_app = FaceAnalysis(name=config.INSIGHTFACE_MODEL, providers=['CPUExecutionProvider'])
                self.face_app.prepare(ctx_id=-1, det_size=(640, 640))
            print("[Visão] InsightFace carregado com sucesso.")
        except ImportError:
            raise ImportError("Biblioteca 'insightface' não encontrada. Instale com: pip install insightface")

        emotion_device = getattr(config, "EMOTION_DEVICE", "cpu").lower()
        emotion_model = getattr(config, "EMOTION_MODEL", "enet_b0_8_best_vgaf")
        print(f"[Visão] Carregando modelo de emoção ({emotion_model}) no dispositivo '{emotion_device.upper()}'...")
        try:
            import os
            import requests
            
            # 1. Rotina de Download Automático do Modelo de Emoções (ONNX)
            model_file = f"{emotion_model}.onnx"
            # O EmotiEffLib geralmente busca na raiz ou pede o caminho. Garantimos que ele exista na raiz.
            if not os.path.exists(model_file):
                url = getattr(config, "EMOTION_MODEL_URLS", {}).get(emotion_model)
                if url:
                    print(f"[Visão] Modelo {model_file} não encontrado localmente.")
                    print(f"[Visão] Baixando modelo de emoções de: {url} ...")
                    response = requests.get(url, stream=True)
                    response.raise_for_status()
                    with open(model_file, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print(f"[Visão] Download de {model_file} concluído com sucesso!")
                else:
                    print(f"[Visão] Aviso: URL para o modelo de emoção {emotion_model} não encontrada no config.py.")

            # 2. Carrega a biblioteca e o modelo
            from emotiefflib.facial_analysis import EmotiEffLibRecognizer
            self.emo_model = EmotiEffLibRecognizer(
                engine="onnx",
                model_name=emotion_model,
                device=emotion_device
            )
            print("[Visão] EmotiEffLib carregado com sucesso.")
        except ImportError:
            print("[Visão] Aviso: 'emotiefflib' não encontrado. Emoções serão marcadas como 'neutro'.")
            self.emo_model = None

    def find_best_match(self, embedding_to_check: np.ndarray, users_db: Dict[int, User]) -> Tuple[Optional[User], float]:
        """Compara o embedding atual com o banco de usuários usando distância de cosseno."""
        if not users_db:
            return None, float('inf')

        best_distance = float('inf')
        best_match_user_id = None

        for user_id, user in users_db.items():
            dist = cosine(embedding_to_check, user.embedding)
            if dist < best_distance:
                best_distance = dist
                best_match_user_id = user_id

        return users_db.get(best_match_user_id), best_distance

    def process_frame(
        self,
        frame: np.ndarray,
        current_user_id: int,
        users_db: Dict[int, User]
    ) -> Tuple[List[Dict[str, Any]], np.ndarray, int]:
        """Executa a inferência de faces e emoções em um único frame."""
        if self.face_app is None:
            return [], frame, current_user_id

        try:
            faces = self.face_app.get(frame)
            h, w = frame.shape[:2]
            detections_in_frame = []

            for face in faces:
                x1, y1, x2, y2 = face.bbox.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                face_pic = frame[y1:y2, x1:x2]
                if face_pic.size == 0:
                    continue

                embedding = face.embedding
                age = int(face.age) if hasattr(face, 'age') and face.age is not None else None
                gender = config.GENDER_DICT.get(face.gender, "indefinido") if hasattr(face, 'gender') else "indefinido"

                # Predição de emoção
                if self.emo_model is not None:
                    try:
                        emotion_list, _ = self.emo_model.predict_emotions(face_pic, logits=False)
                        raw_emotion = emotion_list[0] if emotion_list else "Neutral"
                        emotion = config.EMOTIONS_DICT.get(raw_emotion, "neutro")
                    except Exception:
                        emotion = "neutro"
                else:
                    emotion = "neutro"

                # Identificação e memória do visitante
                matched_user, distance = self.find_best_match(embedding, users_db)
                if distance < self.recognition_threshold and matched_user is not None:
                    user_to_update = matched_user
                else:
                    new_user = User(current_user_id, embedding, gender, age)
                    users_db[current_user_id] = new_user
                    user_to_update = new_user
                    print(f"[Visão] Novo visitante cadastrado: {new_user.name}")
                    current_user_id += 1

                user_to_update.update((x1, y1, x2, y2), age, gender, emotion)

                detection_data = {
                    "id": user_to_update.id,
                    "nome": user_to_update.name,
                    "genero": user_to_update.gender,
                    "idade_aparente": user_to_update.age,
                    "emocao_atual": emotion,
                    "timestamp": user_to_update.last_seen,
                    "bbox": (int(x1), int(y1))
                }
                detections_in_frame.append(detection_data)

                # Anotação visual no frame para debug
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{user_to_update.name} | {emotion} | {user_to_update.age}a"
                cv2.putText(frame, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            return detections_in_frame, frame, current_user_id

        except Exception as e:
            print(f"[Visão] Erro ao processar frame: {e}")
            return [], frame, current_user_id

    def run(self, stop_event, sensor_data_queue, users_db: Optional[Dict[int, User]] = None, next_user_id: Optional[List[int]] = None):
        """Loop contínuo de captura de vídeo e notificação de mudanças de cena."""
        print("[Visão] Inicializando pipeline de imagem...")
        if users_db is None:
            users_db = {}
        if next_user_id is None:
            next_user_id = [0]

        try:
            self.load_models()
        except Exception as e:
            print(f"[Visão] ERRO ao carregar modelos de visão: {e}")
            print("[Visão] A pipeline de visão será desativada.")
            return

        ids_na_cena_anterior = set()

        while not stop_event.is_set():
            try:
                cap = cv2.VideoCapture(self.camera_index)
                if not cap.isOpened():
                    print(f"[Visão] ERRO: Não foi possível acessar a câmera de índice {self.camera_index}. Tentando novamente em 5s...")
                    time.sleep(5)
                    continue

                frame_count = 0
                while not stop_event.is_set():
                    ret, frame = cap.read()
                    if not ret:
                        time.sleep(0.01)
                        continue

                    frame_count += 1
                    processed_frame = frame

                    if frame_count % (self.frames_to_skip + 1) == 0:
                        detections, processed_frame, next_user_id[0] = self.process_frame(frame, next_user_id[0], users_db)

                        if detections:
                            ids_na_cena_atual = {d['id'] for d in detections}
                            # Emite evento para o LLM apenas se houver alteração nos visitantes presentes
                            if ids_na_cena_atual != ids_na_cena_anterior:
                                nomes = [d['nome'] for d in detections]
                                descricao_cena = f"Visitantes presentes na cena: {', '.join(nomes)}."

                                sensor_data = {
                                    "source": "vision",
                                    "data": descricao_cena,
                                    "timestamp": time.time(),
                                    "raw_detections": detections
                                }
                                sensor_data_queue.put(sensor_data)
                                print(f"[Visão] Evento de cena enviado: {descricao_cena}")
                                ids_na_cena_anterior = ids_na_cena_atual

                    cv2.imshow("Isaac - Monitor de Visao", processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        stop_event.set()
                        break

            except Exception as e:
                print(f"[Visão] ERRO CRÍTICO no loop de visão: {e}. Revivendo a thread em 2s com memória de estado mantida...")
                time.sleep(2)
            finally:
                if 'cap' in locals() and cap is not None:
                    cap.release()
                cv2.destroyAllWindows()

        print("[Visão] Pipeline de imagem finalizada.")
