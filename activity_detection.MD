# 🔬 Documentação Técnica - Sistema de Detecção de Atividades

## Visão Geral

O sistema de detecção de atividades implementa uma abordagem multi-fatorial que combina análise de movimento, detecção de objetos e reconhecimento de padrões para classificar ações humanas em categorias específicas.

---

## Arquitetura do Sistema

### Fluxo de Processamento

```
Frame de Vídeo
    ↓
┌─────────────────────────────────────┐
│   1. Pré-processamento              │
│   - Conversão para escala de cinza  │
│   - Detecção de rostos              │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│   2. Extração de Features           │
│   - Análise de movimento regional   │
│   - Detecção de regiões de pele     │
│   - Detecção de corpos superiores   │
│   - Cálculo de intensidade          │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│   3. Análise Contextual             │
│   - Posição relativa mão-rosto      │
│   - Distribuição espacial movimento │
│   - Análise temporal (histórico)    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│   4. Classificação de Atividade     │
│   - Usando Celular                  │
│   - Trabalhando (PC)                │
│   - Lendo / Estudando               │
│   - Conversando / Ocioso            │
└─────────────────────────────────────┘
```

---

## Componentes Técnicos

### 1. Análise de Movimento Regional

O frame é dividido em três regiões horizontais para análise independente:

```python
Região Superior (0-33%):    # Cabeça, mãos elevadas
Região Média (33-66%):      # Tronco, teclado, mesa
Região Inferior (66-100%):  # Pernas, parte baixa
```

**Algoritmo:**
```python
def analyze_regional_movement(frame_atual, frame_anterior):
    # Divide frame em 3 regiões
    superior = frame[0:h//3, :]
    media = frame[h//3:2*h//3, :]
    inferior = frame[2*h//3:, :]
    
    # Calcula diferença absoluta para cada região
    diff_superior = cv2.absdiff(prev_superior, superior)
    diff_media = cv2.absdiff(prev_media, media)
    diff_inferior = cv2.absdiff(prev_inferior, inferior)
    
    # Aplica threshold
    _, thresh_sup = cv2.threshold(diff_superior, 25, 255, BINARY)
    _, thresh_med = cv2.threshold(diff_media, 25, 255, BINARY)
    _, thresh_inf = cv2.threshold(diff_inferior, 25, 255, BINARY)
    
    # Normaliza (0.0 a 1.0)
    motion_sup = sum(thresh_sup) / (pixels_sup * 255)
    motion_med = sum(thresh_med) / (pixels_med * 255)
    motion_inf = sum(thresh_inf) / (pixels_inf * 255)
    
    return {
        'superior': motion_sup,
        'media': motion_med,
        'inferior': motion_inf
    }
```

### 2. Detecção de Regiões de Pele/Mãos

Utiliza espaço de cores HSV para identificar tons de pele:

```python
def detect_hand_regions(frame):
    # Converte BGR → HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Range de cor de pele (calibrado)
    lower_skin = [0, 20, 70]    # H, S, V mínimos
    upper_skin = [20, 255, 255]  # H, S, V máximos
    
    # Cria máscara binária
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Operações morfológicas (remove ruído)
    kernel = np.ones((5,5), uint8)
    skin_mask = cv2.morphologyEx(skin_mask, MORPH_CLOSE, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, MORPH_OPEN, kernel)
    
    # Encontra contornos
    contours = cv2.findContours(skin_mask, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
    
    # Filtra por área (500 < área < 50000 pixels)
    hand_regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 500 < area < 50000:
            x, y, w, h = cv2.boundingRect(contour)
            hand_regions.append((x, y, w, h, area))
    
    return hand_regions
```

### 3. Análise de Proximidade Mão-Rosto

Calcula distância euclidiana entre centros de regiões:

```python
def check_hand_near_face(faces, hands):
    for (fx, fy, fw, fh) in faces:
        face_center = (fx + fw//2, fy + fh//2)
        
        for (hx, hy, hw, hh, area) in hands:
            hand_center = (hx + hw//2, hy + hh//2)
            
            # Distância euclidiana
            distance = sqrt((face_center[0] - hand_center[0])² + 
                          (face_center[1] - hand_center[1])²)
            
            # Threshold de proximidade: 150 pixels
            if distance < 150:
                return True
    
    return False
```

---

## Algoritmos de Classificação

### 🔵 1. Usando Celular

**Critérios de Detecção:**
- ✅ Mão próxima ao rosto (< 150px)
- ✅ Movimento concentrado na região superior (> 0.03)
- ✅ Movimento é localizado (não distribuído)

```python
if hand_near_face AND upper_motion > 0.03 AND concentrated_motion:
    → "Usando Celular"
```

**Características Visuais:**
- Braço elevado
- Mão na altura da face
- Movimento sutil mas constante
- Foco visual direcionado para baixo

---

### 💻 2. Trabalhando (PC)

**Critérios de Detecção:**
- ✅ Movimento predominante na região média
- ✅ Intensidade moderada (0.02 - 0.15)
- ✅ Face presente na cena
- ✅ Padrão consistente ao longo do tempo

```python
if middle_motion > upper_motion AND 
   middle_motion > 0.02 AND 
   motion_intensity < 0.15 AND 
   face_detected:
    
    # Confirmação com histórico (últimos 10 frames)
    if mean(last_10_middle_motions) > 0.02:
        → "Trabalhando (PC)"
```

**Características Visuais:**
- Movimento de mãos na altura do teclado/mouse
- Postura frontal
- Movimento rítmico e consistente
- Baixa variabilidade temporal

---

### 📖 3. Lendo / Estudando

**Critérios de Detecção:**
- ✅ Face presente
- ✅ Movimento muito baixo (< 0.05)
- ✅ Movimento superior > movimento médio
- ✅ Alta estabilidade temporal

```python
if face_detected AND 
   motion_intensity < 0.05 AND 
   upper_motion > middle_motion:
    
    # Verificação de estabilidade (últimos 15 frames)
    if std_deviation(last_15_motions) < 0.02:
        → "Lendo / Estudando"
```

**Características Visuais:**
- Postura estática
- Cabeça ligeiramente inclinada
- Movimento mínimo dos olhos
- Ausência de gestos

---

### 💬 4. Conversando / Ocioso

**Critérios de Detecção:**
- ✅ Face presente
- ✅ Movimento baixo a moderado (< 0.15)
- ✅ Não atende critérios de outras atividades

```python
if face_detected AND motion_intensity < 0.15:
    # Padrão default para presença humana
    → "Conversando / Ocioso"
```

**Características Visuais:**
- Pessoa presente mas sem atividade específica
- Movimento natural do corpo
- Pode incluir conversação
- Gestos ocasionais

---

## Features Extraídas

Para cada frame, o sistema extrai:

```python
features = {
    'num_faces': int,              # Quantidade de rostos
    'num_hands': int,              # Quantidade de mãos detectadas
    'motion_intensity': float,     # Intensidade geral (0.0-1.0)
    'upper_region_motion': float,  # Movimento superior (0.0-1.0)
    'middle_region_motion': float, # Movimento médio (0.0-1.0)
    'lower_region_motion': float,  # Movimento inferior (0.0-1.0)
    'hand_near_face': bool,        # Mão próxima ao rosto?
    'concentrated_motion': bool,   # Movimento localizado?
    'num_bodies': int              # Corpos superiores detectados
}
```

---

## Histórico Temporal

O sistema mantém históricos para análise temporal:

```python
# Janela temporal: 60 frames (2 segundos a 30 FPS)
activity_history = deque(maxlen=60)

# Janela para movimento: 30 frames (1 segundo)
motion_history = deque(maxlen=30)
```

**Uso do Histórico:**
- Suavização de classificações
- Detecção de padrões consistentes
- Redução de falsos positivos
- Análise de tendências

---

## Métricas de Performance

### Thresholds Calibrados

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| `motion_threshold` | 0.02 | Distingue movimento mínimo de ruído |
| `pc_work_motion` | 0.02-0.15 | Faixa típica de trabalho em PC |
| `reading_motion` | < 0.05 | Leitura requer estabilidade |
| `hand_face_distance` | 150px | Distância típica de celular na mão |
| `skin_color_range` | [0-20, 20-255, 70-255] | HSV calibrado para tons de pele |
| `hand_area_min` | 500px² | Elimina pequenos artefatos |
| `hand_area_max` | 50000px² | Evita detecção de corpo inteiro |

---

## Limitações e Melhorias Futuras

### Limitações Atuais

1. **Iluminação**: Sensível a mudanças bruscas de luz
2. **Ângulo**: Melhor performance com câmera frontal
3. **Cor de pele**: Range HSV pode não cobrir todas as etnias
4. **Resolução**: Melhor em vídeos HD (720p+)

### Melhorias Propostas

1. **Deep Learning**: Integrar modelos CNN para reconhecimento de pose
2. **Tracking**: Implementar rastreamento de objetos entre frames
3. **Calibração Automática**: Ajuste dinâmico de thresholds
4. **Multi-pessoa**: Análise individual em cenas com várias pessoas

---

## Exemplo de Uso Avançado

### Ajustando Sensibilidade

```python
# No código video_analysis.py, método _classify_activity

# Para detectar mais facilmente "Trabalhando (PC)"
if middle_motion > 0.015:  # Era 0.02, agora mais sensível
    return "Trabalhando (PC)"

# Para ser mais rigoroso com "Usando Celular"
if hand_near_face and upper_motion > 0.05:  # Era 0.03, agora mais rigoroso
    return "Usando Celular"
```

### Adicionando Nova Categoria

```python
def _classify_activity(self, features):
    # Nova categoria: "Fazendo Exercício"
    if (features['motion_intensity'] > 0.20 and
        features['num_faces'] > 0 and
        features['upper_region_motion'] > 0.15):
        return "Fazendo Exercício"
    
    # Categorias existentes...
```

---

## Validação e Testes

### Casos de Teste Recomendados

1. **Vídeo com pessoa trabalhando em laptop**
   - Esperado: 60-80% "Trabalhando (PC)"

2. **Vídeo com pessoa lendo livro**
   - Esperado: 70-90% "Lendo / Estudando"

3. **Vídeo com pessoa usando smartphone**
   - Esperado: 50-70% "Usando Celular"

4. **Vídeo de conversação**
   - Esperado: 60-80% "Conversando / Ocioso"

### Métricas de Avaliação

- **Acurácia**: Taxa de acertos na classificação
- **Precisão**: Verdadeiros positivos / (VP + Falsos Positivos)
- **Recall**: Verdadeiros positivos / (VP + Falsos Negativos)
- **F1-Score**: Média harmônica de Precisão e Recall

---

## Referências Técnicas

- **OpenCV**: Biblioteca de Computer Vision
- **Haar Cascades**: Detecção de objetos
- **HSV Color Space**: Representação de cores
- **Frame Differencing**: Técnica de detecção de movimento
- **Morphological Operations**: Processamento de imagens binárias

---

**Desenvolvido para análise avançada de comportamento humano em vídeos**
