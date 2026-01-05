"""
Sistema de Análise de Vídeo com IA usando MediaPipe
Reconhecimento Facial, Detecção de Emoções e Atividades Detalhadas
Versão com MediaPipe - Maior Precisão e Performance
"""

import cv2
import numpy as np
from collections import defaultdict, deque
import json
from datetime import datetime
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

class MediaPipeVideoAnalyzer:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        # Inicializa MediaPipe Solutions
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Inicializa detectores
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Métricas de análise
        self.total_frames = 0
        self.faces_detected = []
        self.emotions_detected = defaultdict(int)
        self.activities_detected = defaultdict(int)
        self.anomalies = []
        
        # Histórico para detecção
        self.motion_history = deque(maxlen=30)
        self.activity_history = deque(maxlen=60)
        self.hand_history = deque(maxlen=15)
        self.pose_history = deque(maxlen=30)
        self.prev_frame = None
        
        # Configurações
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
    def analyze_emotion_from_landmarks(self, face_landmarks, image_height, image_width):
        """
        Analisa emoção baseada em landmarks faciais do MediaPipe
        """
        # Índices de landmarks importantes para emoções
        # Sobrancelhas: 70, 63, 105, 66, 107, 336, 296, 334, 293, 300
        # Olhos: 33, 133, 362, 263
        # Boca: 13, 14, 78, 308, 61, 291
        
        landmarks = face_landmarks.landmark
        
        # Calcula distâncias para análise emocional
        # Altura da sobrancelha (quanto mais alta, mais surpreso)
        left_eyebrow = landmarks[70].y
        right_eyebrow = landmarks[300].y
        left_eye = landmarks[33].y
        right_eye = landmarks[263].y
        eyebrow_raise = ((left_eye - left_eyebrow) + (right_eye - right_eyebrow)) / 2
        
        # Abertura da boca (quanto mais aberta, mais surpreso/feliz)
        mouth_top = landmarks[13].y
        mouth_bottom = landmarks[14].y
        mouth_opening = mouth_bottom - mouth_top
        
        # Curvatura da boca (sorriso vs tristeza)
        mouth_left = landmarks[61].y
        mouth_right = landmarks[291].y
        mouth_center = landmarks[13].y
        mouth_curve = mouth_center - ((mouth_left + mouth_right) / 2)
        
        # Classificação baseada em combinação de features
        if eyebrow_raise > 0.05 or mouth_opening > 0.08:
            return "Surpreso"
        elif mouth_curve < -0.01 and mouth_opening > 0.02:
            return "Feliz"
        elif mouth_curve > 0.01:
            return "Triste"
        else:
            return "Neutro"
    
    def analyze_hand_activity(self, hand_landmarks, image_height, image_width):
        """
        Analisa atividade da mão baseada em posição e gestos
        """
        landmarks = hand_landmarks.landmark
        
        # Ponta dos dedos e base da mão
        wrist = landmarks[0]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        thumb_tip = landmarks[4]
        
        # Posição da mão (altura normalizada)
        hand_height = wrist.y
        
        # Orientação da mão
        palm_direction = np.array([
            landmarks[0].x - landmarks[9].x,
            landmarks[0].y - landmarks[9].y,
            landmarks[0].z - landmarks[9].z
        ])
        
        # Análise de gestos
        fingers_extended = self._count_extended_fingers(landmarks)
        
        hand_info = {
            'position': (wrist.x, wrist.y, wrist.z),
            'height': hand_height,
            'fingers_extended': fingers_extended,
            'palm_direction': palm_direction
        }
        
        return hand_info
    
    def _count_extended_fingers(self, landmarks):
        """Conta quantos dedos estão estendidos"""
        finger_tips = [8, 12, 16, 20]  # Indicador, médio, anelar, mindinho
        finger_pips = [6, 10, 14, 18]
        
        extended = 0
        for tip, pip in zip(finger_tips, finger_pips):
            if landmarks[tip].y < landmarks[pip].y:
                extended += 1
        
        # Polegar (lógica diferente)
        if landmarks[4].x < landmarks[3].x:
            extended += 1
            
        return extended
    
    def analyze_pose_activity(self, pose_landmarks, image_height, image_width):
        """
        Analisa atividade baseada na pose corporal
        """
        landmarks = pose_landmarks.landmark
        
        # Pontos chave
        nose = landmarks[0]
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_elbow = landmarks[13]
        right_elbow = landmarks[14]
        left_wrist = landmarks[15]
        right_wrist = landmarks[16]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        
        # Calcula ângulos e posições
        # Inclinação do tronco
        shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_center_y = (left_hip.y + right_hip.y) / 2
        trunk_inclination = shoulder_center_y - hip_center_y
        
        # Posição dos braços (altura relativa aos ombros)
        left_arm_height = left_shoulder.y - left_wrist.y
        right_arm_height = right_shoulder.y - right_wrist.y
        avg_arm_height = (left_arm_height + right_arm_height) / 2
        
        # Distância entre mãos
        wrist_distance = np.sqrt(
            (left_wrist.x - right_wrist.x)**2 + 
            (left_wrist.y - right_wrist.y)**2
        )
        
        pose_info = {
            'trunk_inclination': trunk_inclination,
            'arm_height': avg_arm_height,
            'wrist_distance': wrist_distance,
            'shoulder_y': shoulder_center_y,
            'hip_y': hip_center_y
        }
        
        return pose_info
    
    def classify_activity_mediapipe(self, faces, hands_results, pose_results, 
                                    motion_intensity, frame_rgb):
        """
        Classifica atividade usando dados do MediaPipe
        """
        image_height, image_width = frame_rgb.shape[:2]
        
        features = {
            'num_faces': len(faces) if faces else 0,
            'num_hands': 0,
            'motion_intensity': motion_intensity,
            'hand_near_face': False,
            'hands_raised': False,
            'hands_at_keyboard': False,
            'sitting_posture': False,
            'head_down': False,
            'hands_together': False
        }
        
        # Análise de mãos
        if hands_results and hands_results.multi_hand_landmarks:
            features['num_hands'] = len(hands_results.multi_hand_landmarks)
            
            hand_positions = []
            for hand_landmarks in hands_results.multi_hand_landmarks:
                hand_info = self.analyze_hand_activity(hand_landmarks, image_height, image_width)
                hand_positions.append(hand_info)
                
                # Mão levantada (próxima ao rosto)
                if hand_info['height'] < 0.4:  # Parte superior da imagem
                    features['hands_raised'] = True
                
                # Mão na altura do teclado (região média)
                if 0.4 < hand_info['height'] < 0.7:
                    features['hands_at_keyboard'] = True
            
            # Verifica se mãos estão próximas (juntas)
            if len(hand_positions) == 2:
                distance = np.sqrt(
                    (hand_positions[0]['position'][0] - hand_positions[1]['position'][0])**2 +
                    (hand_positions[0]['position'][1] - hand_positions[1]['position'][1])**2
                )
                if distance < 0.2:
                    features['hands_together'] = True
        
        # Análise de pose
        if pose_results and pose_results.pose_landmarks:
            pose_info = self.analyze_pose_activity(pose_results.pose_landmarks, 
                                                   image_height, image_width)
            
            # Postura sentada (tronco ereto)
            if -0.1 < pose_info['trunk_inclination'] < 0.1:
                features['sitting_posture'] = True
            
            # Cabeça inclinada (lendo)
            if pose_info['shoulder_y'] < 0.6:  # Ombros visíveis na parte superior
                features['head_down'] = True
        
        # Análise de face para proximidade com mãos
        if faces and hands_results and hands_results.multi_hand_landmarks:
            for face in faces:
                face_x = face.location_data.relative_bounding_box.xmin + \
                        face.location_data.relative_bounding_box.width / 2
                face_y = face.location_data.relative_bounding_box.ymin + \
                        face.location_data.relative_bounding_box.height / 2
                
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    wrist = hand_landmarks.landmark[0]
                    distance = np.sqrt((face_x - wrist.x)**2 + (face_y - wrist.y)**2)
                    
                    if distance < 0.15:  # 15% da imagem
                        features['hand_near_face'] = True
                        break
        
        # Armazena histórico
        self.activity_history.append(features)
        
        # Classificação de atividade
        activity = self._classify_from_features(features)
        
        return activity, features
    
    def _classify_from_features(self, features):
        """
        Classifica atividade baseada nas features extraídas
        """
        # 1. Usando Celular: mão próxima ao rosto, levantada
        if (features['hand_near_face'] and features['hands_raised'] and 
            features['num_hands'] >= 1):
            return "Usando Celular"
        
        # 2. Trabalhando (PC): mãos na altura do teclado, postura sentada
        if (features['hands_at_keyboard'] and features['sitting_posture'] and
            features['num_hands'] >= 1):
            
            # Confirma com histórico
            if len(self.activity_history) > 10:
                recent_keyboard = sum(1 for h in list(self.activity_history)[-10:] 
                                    if h.get('hands_at_keyboard', False))
                if recent_keyboard >= 6:  # 60% dos últimos frames
                    return "Trabalhando (PC)"
        
        # 3. Lendo/Estudando: cabeça baixa, mãos juntas ou sem movimento
        if (features['head_down'] and features['motion_intensity'] < 0.05 and
            features['num_faces'] > 0):
            
            # Verifica estabilidade
            if len(self.activity_history) > 15:
                recent_motions = [h.get('motion_intensity', 0) 
                                for h in list(self.activity_history)[-15:]]
                if np.std(recent_motions) < 0.02:
                    return "Lendo / Estudando"
        
        # 4. Conversando/Ocioso: presença de face, sem atividade específica
        if features['num_faces'] > 0:
            return "Conversando / Ocioso"
        
        # Padrão
        return "Conversando / Ocioso"
    
    def detect_motion(self, frame_gray):
        """Detecta movimento entre frames"""
        if self.prev_frame is None:
            self.prev_frame = frame_gray
            return 0.0
        
        frame_diff = cv2.absdiff(self.prev_frame, frame_gray)
        _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
        motion_intensity = np.sum(thresh) / (self.frame_width * self.frame_height * 255)
        
        self.prev_frame = frame_gray
        return motion_intensity
    
    def detect_anomaly(self, motion_intensity, num_faces, activity):
        """Detecta comportamentos anômalos"""
        self.motion_history.append(motion_intensity)
        
        if len(self.motion_history) < 10:
            return False, None
        
        mean_motion = np.mean(self.motion_history)
        std_motion = np.std(self.motion_history)
        
        is_anomaly = False
        anomaly_type = None
        
        # Movimento brusco
        if motion_intensity > mean_motion + 2.5 * std_motion and std_motion > 0.01:
            is_anomaly = True
            anomaly_type = "Movimento Brusco"
        
        # Inatividade súbita
        elif motion_intensity < 0.005 and mean_motion > 0.05:
            is_anomaly = True
            anomaly_type = "Inatividade Súbita"
        
        # Mudança no número de faces
        if len(self.faces_detected) > 0:
            recent_faces = [f['num_faces'] for f in self.faces_detected[-10:]]
            if len(recent_faces) > 0:
                mean_faces = np.mean(recent_faces)
                if abs(num_faces - mean_faces) > 2:
                    is_anomaly = True
                    anomaly_type = "Mudança de Pessoas na Cena"
        
        # Comportamento irregular
        if len(self.activity_history) > 20:
            recent_activities = [h.get('motion_intensity', 0) 
                               for h in list(self.activity_history)[-20:]]
            if np.std(recent_activities) > 0.08:
                is_anomaly = True
                anomaly_type = "Comportamento Irregular"
        
        return is_anomaly, anomaly_type
    
    def process_video(self, output_path='output_video_mediapipe.mp4', show_preview=False):
        """Processa o vídeo completo com MediaPipe"""
        print(f"Iniciando análise com MediaPipe: {self.video_path}")
        print(f"FPS: {self.fps}, Dimensões: {self.frame_width}x{self.frame_height}")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, 
                             (self.frame_width, self.frame_height))
        
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame_count += 1
            self.total_frames += 1
            
            # Converte para RGB (MediaPipe usa RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detecta movimento
            motion_intensity = self.detect_motion(frame_gray)
            
            # Processa com MediaPipe
            faces_results = self.face_detection.process(frame_rgb)
            face_mesh_results = self.face_mesh.process(frame_rgb)
            hands_results = self.hands.process(frame_rgb)
            pose_results = self.pose.process(frame_rgb)
            
            # Extrai faces detectadas
            faces = faces_results.detections if faces_results.detections else []
            
            # Classifica atividade
            activity, features = self.classify_activity_mediapipe(
                faces, hands_results, pose_results, motion_intensity, frame_rgb
            )
            
            # Detecta anomalias
            is_anomaly, anomaly_type = self.detect_anomaly(
                motion_intensity, len(faces), activity
            )
            
            # Desenha detecções no frame
            frame_emotions = []
            
            # Desenha faces e emoções
            if face_mesh_results.multi_face_landmarks:
                for idx, face_landmarks in enumerate(face_mesh_results.multi_face_landmarks):
                    # Desenha mesh facial (simplificado)
                    self.mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles
                        .get_default_face_mesh_tesselation_style())
                    
                    # Analisa emoção
                    emotion = self.analyze_emotion_from_landmarks(
                        face_landmarks, self.frame_height, self.frame_width
                    )
                    frame_emotions.append(emotion)
                    self.emotions_detected[emotion] += 1
                    
                    # Adiciona texto da emoção
                    nose = face_landmarks.landmark[1]
                    x = int(nose.x * self.frame_width)
                    y = int(nose.y * self.frame_height) - 30
                    cv2.putText(frame, f'Rosto {idx+1}: {emotion}', (x, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Desenha mãos
            if hands_results.multi_hand_landmarks:
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style())
            
            # Desenha pose (simplificado - apenas pontos principais)
            if pose_results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    pose_results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles
                    .get_default_pose_landmarks_style())
            
            # Registra atividade
            self.activities_detected[activity] += 1
            
            # Registra anomalia
            if is_anomaly:
                anomaly_data = {
                    'frame': frame_count,
                    'timestamp': frame_count / self.fps,
                    'type': anomaly_type,
                    'motion_intensity': float(motion_intensity),
                    'num_faces': len(faces),
                    'activity': activity
                }
                self.anomalies.append(anomaly_data)
                
                cv2.putText(frame, f'ANOMALIA: {anomaly_type}', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Informações no frame
            info_y = self.frame_height - 120
            cv2.putText(frame, f'Frame: {frame_count}', (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f'Rostos: {len(faces)}', (10, info_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f'Maos: {features["num_hands"]}', (10, info_y + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f'Atividade: {activity}', (10, info_y + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f'Movimento: {motion_intensity:.3f}', (10, info_y + 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, 'MediaPipe v1.0', (10, info_y + 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
            
            # Salva dados do frame
            self.faces_detected.append({
                'frame': frame_count,
                'num_faces': len(faces),
                'emotions': frame_emotions,
                'activity': activity,
                'motion_intensity': float(motion_intensity),
                'features': {k: v for k, v in features.items() 
                           if isinstance(v, (bool, int, float))}
            })
            
            # Escreve frame
            out.write(frame)
            
            if show_preview:
                cv2.imshow('MediaPipe Video Analysis', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            if frame_count % 30 == 0:
                print(f"Processados {frame_count} frames...")
        
        # Finaliza
        self.cap.release()
        out.release()
        if show_preview:
            cv2.destroyAllWindows()
        
        # Libera recursos do MediaPipe
        self.face_detection.close()
        self.face_mesh.close()
        self.hands.close()
        self.pose.close()
        
        print(f"\nAnálise concluída! Total de frames: {self.total_frames}")
        return output_path
    
    def generate_report(self, output_file='relatorio_analise_mediapipe.json'):
        """Gera relatório completo"""
        total_faces = sum(f['num_faces'] for f in self.faces_detected)
        avg_faces = total_faces / len(self.faces_detected) if self.faces_detected else 0
        
        # Calcula porcentagens
        total_activity_frames = sum(self.activities_detected.values())
        activity_percentages = {}
        for activity, count in self.activities_detected.items():
            percentage = (count / total_activity_frames * 100) if total_activity_frames > 0 else 0
            activity_percentages[activity] = {
                'frames': count,
                'porcentagem': round(percentage, 1)
            }
        
        report = {
            'metadata': {
                'video_path': self.video_path,
                'data_analise': datetime.now().isoformat(),
                'fps': self.fps,
                'resolucao': f'{self.frame_width}x{self.frame_height}',
                'tecnologia': 'MediaPipe + OpenCV'
            },
            'metricas_gerais': {
                'total_frames_analisados': self.total_frames,
                'duracao_video_segundos': self.total_frames / self.fps,
                'total_rostos_detectados': total_faces,
                'media_rostos_por_frame': round(avg_faces, 2),
                'numero_anomalias_detectadas': len(self.anomalies)
            },
            'emocoes_detectadas': dict(self.emotions_detected),
            'atividades_detectadas': activity_percentages,
            'anomalias': self.anomalies,
            'resumo': self._generate_summary(activity_percentages)
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self._generate_text_report(report, output_file.replace('.json', '.txt'))
        
        return report
    
    def _generate_summary(self, activity_percentages):
        """Gera resumo textual"""
        summary = []
        
        if self.emotions_detected:
            top_emotion = max(self.emotions_detected.items(), key=lambda x: x[1])
            summary.append(f"Emoção predominante: {top_emotion[0]} ({top_emotion[1]} detecções)")
        
        if activity_percentages:
            summary.append("\nDistribuição de Atividades (MediaPipe):")
            sorted_activities = sorted(activity_percentages.items(), 
                                     key=lambda x: x[1]['porcentagem'], 
                                     reverse=True)
            for activity, data in sorted_activities:
                summary.append(f"  • {data['porcentagem']}% - {activity} ({data['frames']} frames)")
        
        if self.anomalies:
            anomaly_types = defaultdict(int)
            for a in self.anomalies:
                anomaly_types[a['type']] += 1
            summary.append(f"\nTotal de anomalias: {len(self.anomalies)}")
            summary.append("Tipos de anomalias detectadas:")
            for atype, count in anomaly_types.items():
                summary.append(f"  - {atype}: {count} ocorrências")
        else:
            summary.append("\nNenhuma anomalia detectada")
        
        return summary
    
    def _generate_text_report(self, report, filename):
        """Gera relatório em texto"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("RELATÓRIO DE ANÁLISE DE VÍDEO - MEDIAPIPE VERSION\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("INFORMAÇÕES DO VÍDEO\n")
            f.write("-" * 80 + "\n")
            f.write(f"Vídeo: {report['metadata']['video_path']}\n")
            f.write(f"Data da Análise: {report['metadata']['data_analise']}\n")
            f.write(f"FPS: {report['metadata']['fps']}\n")
            f.write(f"Resolução: {report['metadata']['resolucao']}\n")
            f.write(f"Tecnologia: {report['metadata']['tecnologia']}\n\n")
            
            f.write("MÉTRICAS GERAIS\n")
            f.write("-" * 80 + "\n")
            for key, value in report['metricas_gerais'].items():
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")
            f.write("\n")
            
            f.write("ATIVIDADES DETECTADAS (COM PORCENTAGENS)\n")
            f.write("-" * 80 + "\n")
            sorted_activities = sorted(report['atividades_detectadas'].items(), 
                                     key=lambda x: x[1]['porcentagem'], 
                                     reverse=True)
            for activity, data in sorted_activities:
                f.write(f"• {data['porcentagem']}% - {activity} ({data['frames']} frames)\n")
            f.write("\n")
            
            f.write("EMOÇÕES DETECTADAS\n")
            f.write("-" * 80 + "\n")
            for emotion, count in sorted(report['emocoes_detectadas'].items(), 
                                        key=lambda x: x[1], reverse=True):
                f.write(f"{emotion}: {count} detecções\n")
            f.write("\n")
            
            f.write("RESUMO EXECUTIVO\n")
            f.write("-" * 80 + "\n")
            for item in report['resumo']:
                f.write(f"{item}\n")
            f.write("\n")
            
            if report['anomalias']:
                f.write("DETALHES DAS ANOMALIAS\n")
                f.write("-" * 80 + "\n")
                for i, anomaly in enumerate(report['anomalias'], 1):
                    f.write(f"\nAnomalia {i}:\n")
                    f.write(f"  Frame: {anomaly['frame']}\n")
                    f.write(f"  Timestamp: {anomaly['timestamp']:.2f}s\n")
                    f.write(f"  Tipo: {anomaly['type']}\n")
                    f.write(f"  Atividade no momento: {anomaly.get('activity', 'N/A')}\n")
                    f.write(f"  Intensidade de Movimento: {anomaly['motion_intensity']:.3f}\n")
                    f.write(f"  Número de Rostos: {anomaly['num_faces']}\n")


def main():
    """Função principal"""
    import sys
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = 'input_video.mp4'
    
    if not os.path.exists(video_path):
        print(f"ERRO: Vídeo não encontrado: {video_path}")
        print("\nUso: python video_analysis_mediapipe.py <caminho_do_video>")
        print("Ou coloque o vídeo como 'input_video.mp4' no mesmo diretório")
        return
    
    print("=" * 80)
    print("SISTEMA DE ANÁLISE DE VÍDEO COM MEDIAPIPE")
    print("=" * 80)
    print("\nInicializando MediaPipe...")
    
    # Inicializa analisador
    analyzer = MediaPipeVideoAnalyzer(video_path)
    
    print("✓ MediaPipe inicializado com sucesso!")
    print("  - Face Detection")
    print("  - Face Mesh (468 landmarks)")
    print("  - Hand Tracking (21 landmarks por mão)")
    print("  - Pose Estimation (33 landmarks)")
    print()
    
    # Processa vídeo
    output_video = analyzer.process_video(
        output_path='video_analisado_mediapipe.mp4',
        show_preview=False
    )
    
    # Gera relatório
    report = analyzer.generate_report('relatorio_analise_mediapipe.json')
    
    print("\n" + "=" * 80)
    print("ANÁLISE CONCLUÍDA COM MEDIAPIPE!")
    print("=" * 80)
    print(f"\nVídeo processado: {output_video}")
    print("Relatórios gerados:")
    print("  - relatorio_analise_mediapipe.json (formato JSON)")
    print("  - relatorio_analise_mediapipe.txt (formato texto)")
    print("\n" + "=" * 80)
    print("RESUMO:")
    print("=" * 80)
    for item in report['resumo']:
        print(f"{item}")
    print("\n" + "=" * 80)
    print(f"Total de Frames Analisados: {report['metricas_gerais']['total_frames_analisados']}")
    print(f"Número de Anomalias Detectadas: {report['metricas_gerais']['numero_anomalias_detectadas']}")
    print("=" * 80)
    print("\n✓ Análise com MediaPipe oferece maior precisão em:")
    print("  • Detecção facial (468 pontos de landmark)")
    print("  • Reconhecimento de mãos (tracking preciso)")
    print("  • Estimativa de pose corporal")
    print("  • Análise de expressões faciais")
    print("=" * 80)


if __name__ == '__main__':
    main()
