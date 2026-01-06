"""
Sistema de Análise de Vídeo com IA usando YOLO
Detecção automática de objetos e atividades
Reconhecimento Facial + YOLO para máxima precisão
"""

import cv2
import numpy as np
from collections import defaultdict, deque
import json
from datetime import datetime
import os

print("Inicializando Sistema de Análise com YOLO...")
print("="*70)

# Verifica dependências
try:
    from ultralytics import YOLO
    print("✓ Ultralytics YOLO disponível")
    YOLO_AVAILABLE = True
except ImportError:
    print("✗ Ultralytics YOLO não instalado")
    print("\nPara instalar: pip install ultralytics")
    YOLO_AVAILABLE = False

if not YOLO_AVAILABLE:
    print("\n" + "="*70)
    print("INSTALANDO YOLO...")
    print("="*70)
    import subprocess
    try:
        subprocess.check_call(['pip', 'install', 'ultralytics'])
        from ultralytics import YOLO
        print("✓ YOLO instalado com sucesso!")
        YOLO_AVAILABLE = True
    except:
        print("✗ Falha ao instalar YOLO")
        print("\nUse a versão OpenCV como alternativa:")
        print("  python video_analysis.py input_video.mp4")
        exit(1)


class YOLOVideoAnalyzer:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        # Inicializa YOLO
        print("\nCarregando modelo YOLO...")
        try:
            # YOLOv8n (nano) - rápido e leve
            self.yolo_model = YOLO('yolov8n.pt')
            print("✓ YOLOv8-nano carregado")
        except:
            print("Baixando modelo YOLOv8-nano...")
            self.yolo_model = YOLO('yolov8n.pt')
            print("✓ Modelo baixado e carregado")
        
        # Carrega detector de faces
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Classes YOLO relevantes para atividades
        self.activity_objects = {
            'laptop': 'Trabalhando (PC)',
            'cell phone': 'Usando Celular',
            'book': 'Lendo / Estudando',
            'keyboard': 'Trabalhando (PC)',
            'mouse': 'Trabalhando (PC)',
            'tv': 'Assistindo TV',
            'cup': 'Tomando Café/Chá',
            'bottle': 'Bebendo',
            'dining table': 'Em Reunião',
            'chair': 'Sentado'
        }
        
        # Métricas
        self.total_frames = 0
        self.faces_detected = []
        self.emotions_detected = defaultdict(int)
        self.activities_detected = defaultdict(int)
        self.objects_detected = defaultdict(int)
        self.anomalies = []
        
        # Histórico
        self.motion_history = deque(maxlen=30)
        self.activity_history = deque(maxlen=60)
        self.object_history = deque(maxlen=30)
        self.prev_frame = None
        
        # Configurações
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"✓ Analisador inicializado")
        print(f"  Vídeo: {video_path}")
        print(f"  FPS: {self.fps}")
        print(f"  Resolução: {self.frame_width}x{self.frame_height}")
    
    def detect_faces(self, frame):
        """Detecta rostos no frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return faces, gray
    
    def analyze_emotion(self, face_roi):
        """Análise de emoção baseada em características faciais"""
        mean_intensity = np.mean(face_roi)
        std_intensity = np.std(face_roi)
        
        if std_intensity > 50:
            if mean_intensity > 120:
                return "Feliz"
            elif mean_intensity < 80:
                return "Triste"
            else:
                return "Neutro"
        else:
            return "Neutro"
    
    def detect_objects_yolo(self, frame):
        """Detecta objetos usando YOLO"""
        results = self.yolo_model(frame, verbose=False)
        
        detected_objects = []
        boxes = []
        
        for result in results:
            for box in result.boxes:
                # Pega informações da detecção
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.yolo_model.names[class_id]
                
                # Apenas objetos com confiança > 0.5
                if confidence > 0.5:
                    detected_objects.append({
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': (int(x1), int(y1), int(x2), int(y2))
                    })
                    boxes.append((class_name, confidence, 
                                (int(x1), int(y1), int(x2), int(y2))))
        
        return detected_objects, boxes
    
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
    
    def classify_activity_yolo(self, objects_detected, num_faces, motion_intensity):
        """
        Classifica atividade baseada em objetos detectados pelo YOLO
        """
        # Armazena objetos no histórico
        current_objects = [obj['class'] for obj in objects_detected]
        self.object_history.append(current_objects)
        
        # Contabiliza objetos detectados
        for obj in objects_detected:
            self.objects_detected[obj['class']] += 1
        
        features = {
            'objects': current_objects,
            'num_faces': num_faces,
            'motion_intensity': motion_intensity
        }
        
        self.activity_history.append(features)
        
        # Prioridade de detecção baseada em objetos
        activity = None
        confidence = 0
        
        # 1. Usando Celular (detecta cell phone)
        if 'cell phone' in current_objects:
            activity = "Usando Celular"
            confidence = max([obj['confidence'] for obj in objects_detected 
                            if obj['class'] == 'cell phone'])
        
        # 2. Trabalhando no PC (detecta laptop, keyboard, mouse)
        elif any(obj in current_objects for obj in ['laptop', 'keyboard', 'mouse']):
            # Verifica histórico recente para confirmar
            recent_work_objects = 0
            for hist_objects in list(self.object_history)[-10:]:
                if any(obj in hist_objects for obj in ['laptop', 'keyboard', 'mouse']):
                    recent_work_objects += 1
            
            if recent_work_objects >= 5:  # 50% dos últimos frames
                activity = "Trabalhando (PC)"
                confidence = 0.8
        
        # 3. Lendo/Estudando (detecta book)
        elif 'book' in current_objects and motion_intensity < 0.1:
            # Verifica estabilidade (pessoa lendo fica relativamente parada)
            if len(self.activity_history) > 10:
                recent_motions = [h['motion_intensity'] 
                                for h in list(self.activity_history)[-10:]]
                if np.std(recent_motions) < 0.03:
                    activity = "Lendo / Estudando"
                    confidence = 0.7
        
        # 4. Assistindo TV
        elif 'tv' in current_objects and num_faces > 0:
            activity = "Assistindo TV"
            confidence = 0.7
        
        # 5. Em Reunião (detecta multiple pessoas + mesa)
        elif num_faces >= 2 and 'dining table' in current_objects:
            activity = "Em Reunião"
            confidence = 0.6
        
        # 6. Conversando/Ocioso (pessoa presente sem atividade específica)
        elif num_faces > 0:
            activity = "Conversando / Ocioso"
            confidence = 0.5
        
        # 7. Sem Atividade Detectada
        else:
            activity = "Sem Detecção"
            confidence = 0.3
        
        return activity, confidence, current_objects
    
    def detect_anomaly(self, motion_intensity, num_faces, activity, objects):
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
        
        # Mudança no número de pessoas
        if len(self.faces_detected) > 0:
            recent_faces = [f['num_faces'] for f in self.faces_detected[-10:]]
            if len(recent_faces) > 0:
                mean_faces = np.mean(recent_faces)
                if abs(num_faces - mean_faces) > 2:
                    is_anomaly = True
                    anomaly_type = "Mudança de Pessoas na Cena"
        
        # Objeto incomum detectado
        unusual_objects = ['knife', 'scissors', 'fire hydrant']
        if any(obj in objects for obj in unusual_objects):
            is_anomaly = True
            anomaly_type = "Objeto Incomum Detectado"
        
        return is_anomaly, anomaly_type
    
    def process_video(self, output_path='video_analisado_yolo.mp4', show_preview=False):
        """Processa o vídeo completo com YOLO"""
        print("\n" + "="*70)
        print("INICIANDO PROCESSAMENTO COM YOLO")
        print("="*70)
        print()
        
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
            motion_intensity = self.detect_motion(gray)
            
            # Detecta objetos com YOLO
            objects_detected, boxes = self.detect_objects_yolo(frame)
            
            # Classifica atividade usando YOLO
            activity, confidence, current_objects = self.classify_activity_yolo(
                objects_detected, len(faces), motion_intensity
            )
            
            # Detecta anomalias
            is_anomaly, anomaly_type = self.detect_anomaly(
                motion_intensity, len(faces), activity, current_objects
            )
            
            # Desenha detecções de rostos
            frame_emotions = []
            for i, (x, y, w, h) in enumerate(faces):
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                face_roi = gray[y:y+h, x:x+w]
                emotion = self.analyze_emotion(face_roi)
                frame_emotions.append(emotion)
                self.emotions_detected[emotion] += 1
                
                cv2.putText(frame, f'Rosto {i+1}: {emotion}', (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Desenha detecções do YOLO
            for obj_name, conf, (x1, y1, x2, y2) in boxes:
                # Cor baseada no tipo de objeto
                if obj_name in self.activity_objects:
                    color = (255, 0, 0)  # Azul para objetos de atividade
                else:
                    color = (255, 165, 0)  # Laranja para outros objetos
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f'{obj_name}: {conf:.2f}'
                cv2.putText(frame, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
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
                    'activity': activity,
                    'objects_detected': current_objects
                }
                self.anomalies.append(anomaly_data)
                
                cv2.putText(frame, f'ANOMALIA: {anomaly_type}', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Informações no frame
            info_y = self.frame_height - 140
            cv2.putText(frame, f'Frame: {frame_count}', (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f'Rostos: {len(faces)}', (10, info_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f'Objetos: {len(objects_detected)}', (10, info_y + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f'Atividade: {activity}', (10, info_y + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f'Confianca: {confidence:.2f}', (10, info_y + 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f'Movimento: {motion_intensity:.3f}', (10, info_y + 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, 'Powered by YOLOv8', (10, info_y + 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Salva dados do frame
            self.faces_detected.append({
                'frame': frame_count,
                'num_faces': len(faces),
                'emotions': frame_emotions,
                'activity': activity,
                'confidence': float(confidence),
                'objects': current_objects,
                'motion_intensity': float(motion_intensity)
            })
            
            # Escreve frame processado
            out.write(frame)
            
            # Mostra preview
            if show_preview:
                cv2.imshow('YOLO Video Analysis', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Progresso
            if frame_count % 30 == 0:
                print(f"Processados {frame_count} frames... "
                      f"(Atividade atual: {activity})")
        
        # Finaliza
        self.cap.release()
        out.release()
        if show_preview:
            cv2.destroyAllWindows()
        
        print(f"\n✓ Análise concluída! Total de frames processados: {self.total_frames}")
        return output_path
    
    def generate_report(self, output_file='relatorio_analise_yolo.json'):
        """Gera relatório completo da análise com YOLO"""
        
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
        
        # Top objetos detectados
        top_objects = dict(sorted(self.objects_detected.items(), 
                                 key=lambda x: x[1], reverse=True)[:10])
        
        report = {
            'metadata': {
                'video_path': self.video_path,
                'data_analise': datetime.now().isoformat(),
                'fps': self.fps,
                'resolucao': f'{self.frame_width}x{self.frame_height}',
                'tecnologia': 'YOLOv8 + OpenCV',
                'modelo_yolo': 'YOLOv8-nano'
            },
            'metricas_gerais': {
                'total_frames_analisados': self.total_frames,
                'duracao_video_segundos': self.total_frames / self.fps,
                'total_rostos_detectados': total_faces,
                'media_rostos_por_frame': round(avg_faces, 2),
                'numero_anomalias_detectadas': len(self.anomalies),
                'total_objetos_unicos_detectados': len(self.objects_detected)
            },
            'emocoes_detectadas': dict(self.emotions_detected),
            'atividades_detectadas': activity_percentages,
            'objetos_mais_detectados': top_objects,
            'anomalias': self.anomalies,
            'resumo': self._generate_summary(activity_percentages, top_objects)
        }
        
        # Salva relatório em JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Gera também versão texto
        self._generate_text_report(report, output_file.replace('.json', '.txt'))
        
        return report
    
    def _generate_summary(self, activity_percentages, top_objects):
        """Gera resumo textual da análise"""
        summary = []
        
        # Emoção mais comum
        if self.emotions_detected:
            top_emotion = max(self.emotions_detected.items(), key=lambda x: x[1])
            summary.append(f"Emoção predominante: {top_emotion[0]} ({top_emotion[1]} detecções)")
        
        # Atividades com porcentagens
        if activity_percentages:
            summary.append("\nDistribuição de Atividades (detectadas por YOLO):")
            sorted_activities = sorted(activity_percentages.items(), 
                                     key=lambda x: x[1]['porcentagem'], 
                                     reverse=True)
            for activity, data in sorted_activities:
                summary.append(f"  • {data['porcentagem']}% - {activity} ({data['frames']} frames)")
        
        # Objetos mais detectados
        if top_objects:
            summary.append("\nObjetos mais detectados:")
            for obj, count in list(top_objects.items())[:5]:
                summary.append(f"  • {obj}: {count} detecções")
        
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
            f.write("RELATÓRIO DE ANÁLISE DE VÍDEO COM YOLO\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("INFORMAÇÕES DO VÍDEO\n")
            f.write("-" * 80 + "\n")
            for key, value in report['metadata'].items():
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")
            f.write("\n")
            
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
            
            f.write("OBJETOS MAIS DETECTADOS\n")
            f.write("-" * 80 + "\n")
            for obj, count in report['objetos_mais_detectados'].items():
                f.write(f"• {obj}: {count} detecções\n")
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
                    f.write(f"  Objetos detectados: {', '.join(anomaly.get('objects_detected', []))}\n")
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
        print("\nUso: python video_analysis_yolo.py <caminho_do_video>")
        print("Ou coloque o vídeo como 'input_video.mp4' no mesmo diretório")
        return
    
    print("\n" + "=" * 80)
    print("SISTEMA DE ANÁLISE DE VÍDEO COM YOLO")
    print("="*80)
    
    # Inicializa analisador
    analyzer = YOLOVideoAnalyzer(video_path)
    
    # Processa vídeo
    output_video = analyzer.process_video(
        output_path='video_analisado_yolo.mp4',
        show_preview=False
    )
    
    # Gera relatório
    report = analyzer.generate_report('relatorio_analise_yolo.json')
    
    print("\n" + "=" * 80)
    print("ANÁLISE CONCLUÍDA COM YOLO!")
    print("=" * 80)
    print(f"\nVídeo processado salvo em: {output_video}")
    print("Relatórios gerados:")
    print("  - relatorio_analise_yolo.json (formato JSON)")
    print("  - relatorio_analise_yolo.txt (formato texto)")
    print("\n" + "=" * 80)
    print("RESUMO:")
    print("=" * 80)
    for item in report['resumo']:
        print(f"{item}")
    print("\n" + "=" * 80)
    print(f"Total de Frames Analisados: {report['metricas_gerais']['total_frames_analisados']}")
    print(f"Número de Anomalias Detectadas: {report['metricas_gerais']['numero_anomalias_detectadas']}")
    print(f"Objetos Únicos Detectados: {report['metricas_gerais']['total_objetos_unicos_detectados']}")
    print("=" * 80)
    print("\n✓ Análise com YOLO oferece:")
    print("  • Detecção automática de 80+ tipos de objetos")
    print("  • Identificação precisa de atividades baseada em contexto")
    print("  • Confiança percentual para cada detecção")
    print("  • Rastreamento de objetos específicos (laptop, celular, livro, etc.)")
    print("=" * 80)


if __name__ == '__main__':
    main()
