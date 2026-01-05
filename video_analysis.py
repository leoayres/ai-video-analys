"""
Sistema de Análise de Vídeo com IA
Reconhecimento Facial, Detecção de Emoções e Atividades Detalhadas
"""

import cv2
import numpy as np
from collections import defaultdict, deque
import json
from datetime import datetime
import os

class VideoAnalyzer:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        # Carrega os classificadores pré-treinados do OpenCV
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        self.upper_body_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_upperbody.xml'
        )
        
        # Métricas de análise
        self.total_frames = 0
        self.faces_detected = []
        self.emotions_detected = defaultdict(int)
        self.activities_detected = defaultdict(int)
        self.anomalies = []
        
        # Histórico para detecção de atividades e anomalias
        self.motion_history = deque(maxlen=30)
        self.activity_history = deque(maxlen=60)  # Histórico de 2 segundos
        self.prev_frame = None
        self.prev_gray = None
        
        # Detecção de objetos e padrões
        self.hand_positions = deque(maxlen=15)
        self.face_positions = deque(maxlen=15)
        
        # Configurações
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
    def detect_faces(self, frame):
        """Detecta rostos no frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return faces, gray
    
    def analyze_emotion(self, face_roi):
        """
        Análise de emoção baseada em características faciais
        Utiliza aspectos como contraste e distribuição de pixels
        """
        # Calcula histograma da região facial
        hist = cv2.calcHist([face_roi], [0], None, [256], [0, 256])
        
        # Análise de características
        mean_intensity = np.mean(face_roi)
        std_intensity = np.std(face_roi)
        
        # Detecta olhos para análise adicional
        eyes = self.eye_cascade.detectMultiScale(face_roi)
        
        # Lógica de classificação emocional
        if std_intensity > 50 and len(eyes) >= 2:
            if mean_intensity > 120:
                return "Feliz"
            elif mean_intensity < 80:
                return "Triste"
            else:
                return "Neutro"
        elif len(eyes) < 2:
            return "Surpreso"
        else:
            return "Neutro"
    
    def detect_hand_regions(self, frame, gray):
        """Detecta possíveis regiões de mãos baseado em cor de pele"""
        # Converte para HSV para detecção de cor de pele
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        #Range de cor de pele (ajustável)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Máscara de pele
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Aplica operações morfológicas
        kernel = np.ones((5, 5), np.uint8)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        
        # Encontra contornos
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtra contornos pequenos e muito grandes
        hand_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 500 < area < 50000:
                x, y, w, h = cv2.boundingRect(contour)
                hand_regions.append((x, y, w, h, area))
        
        return hand_regions
    
    def analyze_activity(self, frame, gray, faces, motion_intensity):
        """
        Análise detalhada de atividades baseada em múltiplos fatores
        """
        height, width = gray.shape
        
        # Detecta regiões de mãos/pele
        hand_regions = self.detect_hand_regions(frame, gray)
        
        # Detecta corpos superiores
        upper_bodies = self.upper_body_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=3, minSize=(50, 50)
        )
        
        # Análise de movimento em regiões específicas
        region_movements = self._analyze_regional_movement(gray)
        
        # Características para classificação
        features = {
            'num_faces': len(faces),
            'num_hands': len(hand_regions),
            'motion_intensity': motion_intensity,
            'upper_region_motion': region_movements['upper'],
            'middle_region_motion': region_movements['middle'],
            'lower_region_motion': region_movements['lower'],
            'hand_near_face': self._check_hand_near_face(faces, hand_regions),
            'concentrated_motion': region_movements['concentrated'],
            'num_bodies': len(upper_bodies)
        }
        
        # Armazena características no histórico
        self.activity_history.append(features)
        
        # Classifica atividade
        activity = self._classify_activity(features)
        
        return activity, features
    
    def _analyze_regional_movement(self, gray):
        """Analisa movimento em diferentes regiões da imagem"""
        if self.prev_gray is None:
            self.prev_gray = gray
            return {'upper': 0, 'middle': 0, 'lower': 0, 'concentrated': False}
        
        height, width = gray.shape
        
        # Divide em regiões
        upper_region = gray[0:height//3, :]
        middle_region = gray[height//3:2*height//3, :]
        lower_region = gray[2*height//3:, :]
        
        prev_upper = self.prev_gray[0:height//3, :]
        prev_middle = self.prev_gray[height//3:2*height//3, :]
        prev_lower = self.prev_gray[2*height//3:, :]
        
        # Calcula movimento em cada região
        upper_diff = cv2.absdiff(prev_upper, upper_region)
        middle_diff = cv2.absdiff(prev_middle, middle_region)
        lower_diff = cv2.absdiff(prev_lower, lower_region)
        
        _, upper_thresh = cv2.threshold(upper_diff, 25, 255, cv2.THRESH_BINARY)
        _, middle_thresh = cv2.threshold(middle_diff, 25, 255, cv2.THRESH_BINARY)
        _, lower_thresh = cv2.threshold(lower_diff, 25, 255, cv2.THRESH_BINARY)
        
        upper_motion = np.sum(upper_thresh) / (upper_thresh.size * 255)
        middle_motion = np.sum(middle_thresh) / (middle_thresh.size * 255)
        lower_motion = np.sum(lower_thresh) / (lower_thresh.size * 255)
        
        # Verifica se movimento está concentrado em uma região
        total_motion = upper_motion + middle_motion + lower_motion
        concentrated = False
        if total_motion > 0:
            max_motion = max(upper_motion, middle_motion, lower_motion)
            if max_motion / total_motion > 0.6:
                concentrated = True
        
        self.prev_gray = gray
        
        return {
            'upper': upper_motion,
            'middle': middle_motion,
            'lower': lower_motion,
            'concentrated': concentrated
        }
    
    def _check_hand_near_face(self, faces, hand_regions):
        """Verifica se há mãos próximas ao rosto"""
        if len(faces) == 0 or len(hand_regions) == 0:
            return False
        
        for (fx, fy, fw, fh) in faces:
            face_center_x = fx + fw // 2
            face_center_y = fy + fh // 2
            
            for (hx, hy, hw, hh, _) in hand_regions:
                hand_center_x = hx + hw // 2
                hand_center_y = hy + hh // 2
                
                # Distância entre mão e rosto
                distance = np.sqrt((face_center_x - hand_center_x)**2 + 
                                 (face_center_y - hand_center_y)**2)
                
                # Se mão está próxima ao rosto (menos de 150 pixels)
                if distance < 150:
                    return True
        
        return False
    
    def _classify_activity(self, features):
        """
        Classifica a atividade baseada nas características extraídas
        Categorias: Conversando/Ocioso, Trabalhando(PC), Lendo/Estudando, Usando Celular
        """
        
        # Usando Celular: mão próxima ao rosto, movimento concentrado na região superior
        if features['hand_near_face'] and features['upper_region_motion'] > 0.03:
            if features['concentrated_motion']:
                return "Usando Celular"
        
        # Trabalhando (PC): movimento concentrado na região média, presença de face, movimento moderado
        if (features['middle_region_motion'] > features['upper_region_motion'] and 
            features['middle_region_motion'] > 0.02 and
            features['num_faces'] > 0 and
            features['motion_intensity'] < 0.15):
            
            # Verifica histórico recente para confirmar
            if len(self.activity_history) > 10:
                recent_middle = [h['middle_region_motion'] for h in list(self.activity_history)[-10:]]
                if np.mean(recent_middle) > 0.02:
                    return "Trabalhando (PC)"
        
        # Lendo/Estudando: baixo movimento, face presente, movimento concentrado na parte superior
        if (features['num_faces'] > 0 and 
            features['motion_intensity'] < 0.05 and
            features['upper_region_motion'] > features['middle_region_motion']):
            
            # Posição estável do rosto indica leitura
            if len(self.activity_history) > 15:
                recent_motions = [h['motion_intensity'] for h in list(self.activity_history)[-15:]]
                if np.std(recent_motions) < 0.02:  # Movimento consistentemente baixo
                    return "Lendo / Estudando"
        
        # Conversando/Ocioso: movimento baixo a moderado, faces presentes
        if features['num_faces'] > 0:
            if features['motion_intensity'] < 0.1:
                return "Conversando / Ocioso"
            elif features['motion_intensity'] < 0.15:
                # Verifica se há variação na face (boca se movendo)
                return "Conversando / Ocioso"
        
        # Padrão: Conversando/Ocioso
        return "Conversando / Ocioso"
    
    def detect_motion(self, frame, gray):
        """Detecta e analisa movimento no frame"""
        if self.prev_frame is None:
            self.prev_frame = gray
            return 0
        
        # Calcula diferença entre frames
        frame_diff = cv2.absdiff(self.prev_frame, gray)
        _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
        
        # Calcula intensidade do movimento
        motion_intensity = np.sum(thresh) / (self.frame_width * self.frame_height * 255)
        
        self.prev_frame = gray
        
        return motion_intensity
    
    def detect_anomaly(self, motion_intensity, num_faces, activity):
        """Detecta comportamentos anômalos"""
        self.motion_history.append(motion_intensity)
        
        if len(self.motion_history) < 10:
            return False, None
        
        # Calcula média e desvio padrão do movimento recente
        mean_motion = np.mean(self.motion_history)
        std_motion = np.std(self.motion_history)
        
        # Detecta anomalias
        is_anomaly = False
        anomaly_type = None
        
        # Movimento brusco (pico repentino)
        if motion_intensity > mean_motion + 2.5 * std_motion and std_motion > 0.01:
            is_anomaly = True
            anomaly_type = "Movimento Brusco"
        
        # Movimento muito baixo quando esperava-se atividade
        elif motion_intensity < 0.005 and mean_motion > 0.05:
            is_anomaly = True
            anomaly_type = "Inatividade Súbita"
        
        # Mudança drástica no número de faces
        if len(self.faces_detected) > 0:
            recent_faces = [f['num_faces'] for f in self.faces_detected[-10:]]
            if len(recent_faces) > 0:
                mean_faces = np.mean(recent_faces)
                if abs(num_faces - mean_faces) > 2:
                    is_anomaly = True
                    anomaly_type = "Mudança de Pessoas na Cena"
        
        # Anomalia baseada em mudança súbita de atividade
        if len(self.activity_history) > 20:
            recent_activities = [h.get('motion_intensity', 0) for h in list(self.activity_history)[-20:]]
            activity_std = np.std(recent_activities)
            if activity_std > 0.08:  # Alta variabilidade
                is_anomaly = True
                anomaly_type = "Comportamento Irregular"
        
        return is_anomaly, anomaly_type
    
    def process_video(self, output_path='output_video.mp4', show_preview=False):
        """Processa o vídeo completo"""
        print(f"Iniciando análise do vídeo: {self.video_path}")
        print(f"FPS: {self.fps}, Dimensões: {self.frame_width}x{self.frame_height}")
        
        # Configuração do vídeo de saída
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
            
            # Detecta rostos
            faces, gray = self.detect_faces(frame)
            
            # Detecta movimento
            motion_intensity = self.detect_motion(frame, gray)
            
            # Analisa atividade detalhada
            activity, features = self.analyze_activity(frame, gray, faces, motion_intensity)
            
            # Detecta anomalias
            is_anomaly, anomaly_type = self.detect_anomaly(motion_intensity, len(faces), activity)
            
            # Processa cada rosto detectado
            frame_emotions = []
            for i, (x, y, w, h) in enumerate(faces):
                # Desenha retângulo ao redor do rosto
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Analisa emoção
                face_roi = gray[y:y+h, x:x+w]
                emotion = self.analyze_emotion(face_roi)
                frame_emotions.append(emotion)
                self.emotions_detected[emotion] += 1
                
                # Adiciona texto com a emoção
                cv2.putText(frame, f'Rosto {i+1}: {emotion}', (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Registra atividade
            self.activities_detected[activity] += 1
            
            # Registra anomalia se detectada
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
                
                # Marca anomalia no vídeo
                cv2.putText(frame, f'ANOMALIA: {anomaly_type}', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Adiciona informações no frame
            info_y = self.frame_height - 100
            cv2.putText(frame, f'Frame: {frame_count}', (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f'Rostos: {len(faces)}', (10, info_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f'Atividade: {activity}', (10, info_y + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f'Movimento: {motion_intensity:.3f}', (10, info_y + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f'Maos detectadas: {features["num_hands"]}', (10, info_y + 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Salva dados do frame
            self.faces_detected.append({
                'frame': frame_count,
                'num_faces': len(faces),
                'emotions': frame_emotions,
                'activity': activity,
                'motion_intensity': float(motion_intensity),
                'features': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                           for k, v in features.items()}
            })
            
            # Escreve frame processado
            out.write(frame)
            
            # Mostra preview se solicitado
            if show_preview:
                cv2.imshow('Video Analysis', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Progresso
            if frame_count % 30 == 0:
                print(f"Processados {frame_count} frames...")
        
        # Finaliza
        self.cap.release()
        out.release()
        if show_preview:
            cv2.destroyAllWindows()
        
        print(f"\nAnálise concluída! Total de frames processados: {self.total_frames}")
        return output_path
    
    def generate_report(self, output_file='relatorio_analise.json'):
        """Gera relatório completo da análise"""
        
        # Calcula estatísticas
        total_faces = sum(f['num_faces'] for f in self.faces_detected)
        avg_faces = total_faces / len(self.faces_detected) if self.faces_detected else 0
        
        # Calcula porcentagens de atividades
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
                'resolucao': f'{self.frame_width}x{self.frame_height}'
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
        
        # Salva relatório em JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Gera também versão texto
        self._generate_text_report(report, output_file.replace('.json', '.txt'))
        
        return report
    
    def _generate_summary(self, activity_percentages):
        """Gera resumo textual da análise"""
        summary = []
        
        # Emoção mais comum
        if self.emotions_detected:
            top_emotion = max(self.emotions_detected.items(), key=lambda x: x[1])
            summary.append(f"Emoção predominante: {top_emotion[0]} ({top_emotion[1]} detecções)")
        
        # Atividades com porcentagens
        if activity_percentages:
            summary.append("\nDistribuição de Atividades:")
            # Ordena por porcentagem
            sorted_activities = sorted(activity_percentages.items(), 
                                     key=lambda x: x[1]['porcentagem'], 
                                     reverse=True)
            for activity, data in sorted_activities:
                summary.append(f"  • {data['porcentagem']}% - {activity} ({data['frames']} frames)")
        
        # Anomalias
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
        """Gera relatório em formato texto"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("RELATÓRIO DE ANÁLISE DE VÍDEO COM DETECÇÃO DE ATIVIDADES\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("INFORMAÇÕES DO VÍDEO\n")
            f.write("-" * 80 + "\n")
            f.write(f"Vídeo: {report['metadata']['video_path']}\n")
            f.write(f"Data da Análise: {report['metadata']['data_analise']}\n")
            f.write(f"FPS: {report['metadata']['fps']}\n")
            f.write(f"Resolução: {report['metadata']['resolucao']}\n\n")
            
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
    
    # Caminho do vídeo
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = 'input_video.mp4'  # Vídeo padrão
    
    if not os.path.exists(video_path):
        print(f"ERRO: Vídeo não encontrado: {video_path}")
        print("\nUso: python video_analysis.py <caminho_do_video>")
        print("Ou coloque o vídeo como 'input_video.mp4' no mesmo diretório")
        return
    
    # Inicializa analisador
    analyzer = VideoAnalyzer(video_path)
    
    # Processa vídeo
    output_video = analyzer.process_video(
        output_path='video_analisado.mp4',
        show_preview=False  # Mude para True para ver preview durante processamento
    )
    
    # Gera relatório
    report = analyzer.generate_report('relatorio_analise.json')
    
    print("\n" + "=" * 80)
    print("ANÁLISE CONCLUÍDA!")
    print("=" * 80)
    print(f"\nVídeo processado salvo em: {output_video}")
    print("Relatórios gerados:")
    print("  - relatorio_analise.json (formato JSON)")
    print("  - relatorio_analise.txt (formato texto)")
    print("\n" + "=" * 80)
    print("RESUMO:")
    print("=" * 80)
    for item in report['resumo']:
        print(f"{item}")
    print("\n" + "=" * 80)
    print(f"Total de Frames Analisados: {report['metricas_gerais']['total_frames_analisados']}")
    print(f"Número de Anomalias Detectadas: {report['metricas_gerais']['numero_anomalias_detectadas']}")
    print("=" * 80)


if __name__ == '__main__':
    main()
