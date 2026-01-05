"""
Sistema de Análise de Vídeo com IA
Reconhecimento Facial, Detecção de Emoções e Atividades
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
        
        # Métricas de análise
        self.total_frames = 0
        self.faces_detected = []
        self.emotions_detected = defaultdict(int)
        self.activities_detected = defaultdict(int)
        self.anomalies = []
        
        # Histórico de movimento para detecção de anomalias
        self.motion_history = deque(maxlen=30)
        self.prev_frame = None
        
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
        
        # Lógica simplificada de classificação emocional
        # Baseada em intensidade e variação de pixels
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
    
    def detect_motion(self, frame, gray):
        """Detecta e analisa movimento no frame"""
        if self.prev_frame is None:
            self.prev_frame = gray
            return 0, "Iniciando"
        
        # Calcula diferença entre frames
        frame_diff = cv2.absdiff(self.prev_frame, gray)
        _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
        
        # Calcula intensidade do movimento
        motion_intensity = np.sum(thresh) / (self.frame_width * self.frame_height * 255)
        
        self.prev_frame = gray
        
        # Classifica atividade baseada no movimento
        if motion_intensity > 0.15:
            activity = "Movimento Intenso"
        elif motion_intensity > 0.05:
            activity = "Movimento Moderado"
        elif motion_intensity > 0.01:
            activity = "Movimento Leve"
        else:
            activity = "Estático"
        
        return motion_intensity, activity
    
    def detect_anomaly(self, motion_intensity, num_faces):
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
            
            # Detecta movimento e atividade
            motion_intensity, activity = self.detect_motion(frame, gray)
            
            # Detecta anomalias
            is_anomaly, anomaly_type = self.detect_anomaly(motion_intensity, len(faces))
            
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
                    'num_faces': len(faces)
                }
                self.anomalies.append(anomaly_data)
                
                # Marca anomalia no vídeo
                cv2.putText(frame, f'ANOMALIA: {anomaly_type}', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Adiciona informações no frame
            info_y = self.frame_height - 80
            cv2.putText(frame, f'Frame: {frame_count}', (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f'Rostos: {len(faces)}', (10, info_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f'Atividade: {activity}', (10, info_y + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f'Movimento: {motion_intensity:.3f}', (10, info_y + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Salva dados do frame
            self.faces_detected.append({
                'frame': frame_count,
                'num_faces': len(faces),
                'emotions': frame_emotions,
                'activity': activity,
                'motion_intensity': float(motion_intensity)
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
            'atividades_detectadas': dict(self.activities_detected),
            'anomalias': self.anomalies,
            'resumo': self._generate_summary()
        }
        
        # Salva relatório em JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Gera também versão texto
        self._generate_text_report(report, output_file.replace('.json', '.txt'))
        
        return report
    
    def _generate_summary(self):
        """Gera resumo textual da análise"""
        summary = []
        
        # Emoção mais comum
        if self.emotions_detected:
            top_emotion = max(self.emotions_detected.items(), key=lambda x: x[1])
            summary.append(f"Emoção predominante: {top_emotion[0]} ({top_emotion[1]} detecções)")
        
        # Atividade mais comum
        if self.activities_detected:
            top_activity = max(self.activities_detected.items(), key=lambda x: x[1])
            summary.append(f"Atividade predominante: {top_activity[0]} ({top_activity[1]} frames)")
        
        # Anomalias
        if self.anomalies:
            anomaly_types = defaultdict(int)
            for a in self.anomalies:
                anomaly_types[a['type']] += 1
            summary.append(f"Total de anomalias: {len(self.anomalies)}")
            summary.append("Tipos de anomalias detectadas:")
            for atype, count in anomaly_types.items():
                summary.append(f"  - {atype}: {count} ocorrências")
        else:
            summary.append("Nenhuma anomalia detectada")
        
        return summary
    
    def _generate_text_report(self, report, filename):
        """Gera relatório em formato texto"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("RELATÓRIO DE ANÁLISE DE VÍDEO\n")
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
            
            f.write("EMOÇÕES DETECTADAS\n")
            f.write("-" * 80 + "\n")
            for emotion, count in sorted(report['emocoes_detectadas'].items(), 
                                        key=lambda x: x[1], reverse=True):
                f.write(f"{emotion}: {count} detecções\n")
            f.write("\n")
            
            f.write("ATIVIDADES DETECTADAS\n")
            f.write("-" * 80 + "\n")
            for activity, count in sorted(report['atividades_detectadas'].items(), 
                                         key=lambda x: x[1], reverse=True):
                f.write(f"{activity}: {count} frames\n")
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
        print(f"• {item}")
    print("\n" + "=" * 80)
    print(f"Total de Frames Analisados: {report['metricas_gerais']['total_frames_analisados']}")
    print(f"Número de Anomalias Detectadas: {report['metricas_gerais']['numero_anomalias_detectadas']}")
    print("=" * 80)


if __name__ == '__main__':
    main()
