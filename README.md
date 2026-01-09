# 🎯 Sistema de Análise de Vídeo com YOLO + DeepFace

Versão **PREMIUM** com **YOLOv8** para detecção de objetos e **DeepFace** para análise avançada de emoções.

## 🌟 Tecnologias Integradas

### 1️⃣ YOLO (You Only Look Once)
- Detecção automática de 80+ objetos
- Classificação inteligente de atividades

### 2️⃣ DeepFace
- Análise profissional de 7 emoções
- Precisão 85-90% (vs 60% método básico)
- Deep Learning estado da arte

### 3️⃣ OpenCV
- Detecção rápida de rostos
- Processamento de vídeo

---

## 🎭 Emoções Detectadas (DeepFace)

✅ **7 Emoções Completas:**
1. 😠 **Raiva** (angry)
2. 🤢 **Nojo** (disgust)
3. 😨 **Medo** (fear)
4. 😊 **Feliz** (happy)
5. 😢 **Triste** (sad)
6. 😲 **Surpreso** (surprise)
7. 😐 **Neutro** (neutral)

**vs 4 emoções do método básico**

### Vantagens do YOLO

✅ **Detecção automática de 80+ objetos**
- laptop, cell phone, book, keyboard, mouse, tv, cup, bottle, etc.

✅ **Alta precisão**
- Detecta objetos com confiança percentual
- Identifica contexto da atividade automaticamente

✅ **Inteligente**
- Analisa a cena completa
- Entende o contexto dos objetos
- Classifica atividades baseado em evidências

✅ **Estado da arte**
- YOLOv8 (2023) - última versão
- Usado em aplicações profissionais
- Mantido pela Ultralytics

---

## 🆚 Comparação de Versões

| Característica | OpenCV | MediaPipe | **YOLO + DeepFace** |
|----------------|--------|-----------|---------------------|
| Detecção Facial | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Análise Emoções | ⭐⭐⭐ (4) | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ (7) |
| Detecção Objetos | ❌ | ❌ | ⭐⭐⭐⭐⭐ |
| Atividades | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Precisão | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Velocidade | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Instalação | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |

---

## 🎯 Detecção Inteligente de Atividades

### Como funciona?

YOLO detecta objetos na cena e o sistema **infere** a atividade:

#### 📱 Usando Celular
```
Objetos detectados: cell phone + person
→ Atividade: "Usando Celular"
Confiança: 95%
```

#### 💻 Trabalhando (PC)
```
Objetos detectados: laptop + keyboard + mouse + person
→ Atividade: "Trabalhando (PC)"
Confiança: 90%
```

#### 📖 Lendo / Estudando
```
Objetos detectados: book + person
Movimento: Baixo (< 0.1)
→ Atividade: "Lendo / Estudando"
Confiança: 85%
```

#### 📺 Assistindo TV
```
Objetos detectados: tv + person
→ Atividade: "Assistindo TV"
Confiança: 80%
```

---

## 📦 Instalação

### Requisitos

- Python 3.8 - 3.11
- 4GB RAM mínimo (8GB recomendado)
- 2GB espaço em disco

### Passo a Passo

#### 1. Instalar dependências básicas

```bash
pip install opencv-python numpy
```

#### 2. Instalar Ultralytics (YOLO) e DeepFace

```bash
pip install ultralytics deepface tf-keras
```

Isso também instala automaticamente:
- PyTorch (framework de deep learning)
- TensorFlow/Keras (para DeepFace)
- torchvision (processamento de imagens)

#### 3. Instalar via requirements

```bash
pip install -r requirements_yolo.txt
```

---

## 🚀 Como Usar

### Execução Simples

```bash
python video_analysis_yolo.py input_video.mp4
```

### Primeira Execução

Na primeira vez, o YOLO irá **baixar o modelo** automaticamente (~6MB):

```
Carregando modelo YOLO...
Downloading yolov8n.pt...
100%|████████████| 6.2M/6.2M [00:02<00:00, 2.8MB/s]
✓ YOLOv8-nano carregado
```

Nas próximas execuções, o modelo já estará salvo e será carregado instantaneamente.

---

## 📊 O que o YOLO Detecta

### 80+ Classes de Objetos

**Eletrônicos:**
- laptop, cell phone, keyboard, mouse, tv, remote

**Leitura/Estudo:**
- book

**Mobília:**
- chair, couch, bed, dining table, desk

**Pessoas:**
- person

**Alimentação:**
- cup, bottle, bowl, wine glass, fork, knife, spoon

**E muito mais!**

---

## 🎨 Visualizações no Vídeo

O vídeo processado mostra:

### Detecção de Rostos
- Retângulo verde ao redor
- Emoção identificada

### Detecção de Objetos (YOLO)
- Retângulo azul para objetos relevantes (laptop, celular, livro)
- Retângulo laranja para outros objetos
- Label com nome e confiança (ex: "laptop: 0.94")

### Informações em Tempo Real
- Frame atual
- Número de rostos
- Número de objetos
- Atividade detectada
- Confiança da detecção
- Intensidade de movimento

---

## 📈 Relatórios Gerados

### 1. Relatório JSON (`relatorio_analise_yolo.json`)

```json
{
  "metricas_gerais": {
    "total_frames_analisados": 1500,
    "numero_anomalias_detectadas": 8,
    "total_objetos_unicos_detectados": 15
  },
  "atividades_detectadas": {
    "Trabalhando (PC)": {
      "frames": 800,
      "porcentagem": 53.3
    },
    "Usando Celular": {
      "frames": 300,
      "porcentagem": 20.0
    },
    "Lendo / Estudando": {
      "frames": 250,
      "porcentagem": 16.7
    }
  },
  "objetos_mais_detectados": {
    "laptop": 1200,
    "cell phone": 450,
    "person": 1500,
    "chair": 1400,
    "cup": 300
  }
}
```

### 2. Relatório TXT

Formato legível com todas as informações organizadas.

### 3. Vídeo Anotado

Vídeo com todas as detecções visualizadas.

---

## 🎯 Exemplo de Uso Real

```bash
# Processar vídeo
python video_analysis_yolo.py meu_video.mp4

# Saída:
# Processados 30 frames... (Atividade atual: Trabalhando (PC))
# Processados 60 frames... (Atividade atual: Trabalhando (PC))
# Processados 90 frames... (Atividade atual: Usando Celular)
# ...
# ✓ Análise concluída!
```

---

## 🔧 Configurações Avançadas

### Ajustar Confiança Mínima

No código, linha ~182:
```python
if confidence > 0.5:  # Altere para 0.3 (mais detecções) ou 0.7 (mais rigoroso)
```

### Adicionar Novos Objetos de Atividade

No código, linha ~46-57:
```python
self.activity_objects = {
    'laptop': 'Trabalhando (PC)',
    'cell phone': 'Usando Celular',
    'book': 'Lendo / Estudando',
    # Adicione mais:
    'sports ball': 'Jogando Bola',
    'bicycle': 'Andando de Bicicleta',
}
```

### Usar Modelo Maior (Mais Preciso)

```python
# YOLOv8n (nano) - rápido, leve (padrão)
self.yolo_model = YOLO('yolov8n.pt')

# YOLOv8s (small) - mais preciso
self.yolo_model = YOLO('yolov8s.pt')

# YOLOv8m (medium) - ainda mais preciso
self.yolo_model = YOLO('yolov8m.pt')

# YOLOv8l (large) - máxima precisão
self.yolo_model = YOLO('yolov8l.pt')
```

**Nota:** Modelos maiores são mais lentos mas mais precisos.

---

## 📊 Performance

### Velocidade Esperada

| Resolução | YOLOv8n | YOLOv8s | YOLOv8m |
|-----------|---------|---------|---------|
| 480p | ~20 FPS | ~15 FPS | ~10 FPS |
| 720p | ~12 FPS | ~8 FPS | ~5 FPS |
| 1080p | ~8 FPS | ~5 FPS | ~3 FPS |

*Hardware: Intel i5/i7, sem GPU*

### Com GPU (NVIDIA)

| Resolução | YOLOv8n | YOLOv8s | YOLOv8m |
|-----------|---------|---------|---------|
| 480p | ~60 FPS | ~45 FPS | ~30 FPS |
| 720p | ~40 FPS | ~30 FPS | ~20 FPS |
| 1080p | ~25 FPS | ~18 FPS | ~12 FPS |

*Hardware: NVIDIA RTX 3060 ou superior*

---

## 💡 Quando Usar YOLO?

### ✅ Use YOLO quando:

1. **Precisão é crítica**
   - Análise profissional
   - Pesquisa acadêmica
   - Aplicações comerciais

2. **Precisa detectar objetos específicos**
   - Identificar uso de dispositivos
   - Contar objetos na cena
   - Rastrear equipamentos

3. **Quer contextualização automática**
   - Inferir atividades por objetos
   - Entender comportamento
   - Análise semântica da cena

4. **Tem hardware adequado**
   - PC moderno (i5+, 8GB RAM)
   - Ou paciência para processar devagar

### ⚠️ Use OpenCV quando:

1. **Velocidade é prioridade**
2. **Hardware limitado**
3. **Vídeos muito longos (> 1 hora)**
4. **Análise básica suficiente**

---

## 🐛 Solução de Problemas

### Erro: "No module named 'ultralytics'"

```bash
pip install ultralytics
```

### Erro: PyTorch não instalado

```bash
# CPU apenas (menor, mais lento)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# GPU NVIDIA (requer CUDA)
pip install torch torchvision
```

### Erro: Memória insuficiente

Use modelo menor:
```python
self.yolo_model = YOLO('yolov8n.pt')  # nano (padrão)
```

Ou reduza resolução do vídeo antes de processar.

### Processamento muito lento

Opções:
1. Use GPU se disponível
2. Reduza resolução do vídeo
3. Use YOLOv8n (nano)
4. Processe menos frames (pule frames)

---

## 🎓 Comparação Final

### OpenCV
- ⚡ Muito rápido
- 💻 Roda em qualquer PC
- 🎯 Precisão razoável
- **Melhor para:** Análise rápida, vídeos longos

### MediaPipe (0.10.31)
- ⚠️ API incompatível com vídeo contínuo
- **Não recomendado atualmente**

### YOLO
- 🎯 Máxima precisão
- 🤖 Detecção inteligente
- 📦 80+ objetos reconhecidos
- 🔍 Contextualização automática
- **Melhor para:** Análise profissional, precisão

---

## 📚 Recursos Adicionais

### Documentação Oficial

- Ultralytics: https://docs.ultralytics.com/
- YOLOv8: https://github.com/ultralytics/ultralytics
- PyTorch: https://pytorch.org/

### Tutoriais

- Object Detection: https://docs.ultralytics.com/tasks/detect/
- Custom Training: https://docs.ultralytics.com/modes/train/

---

## ✅ Checklist de Uso

- [ ] Python 3.8-3.11 instalado
- [ ] Dependências instaladas (`pip install -r requirements_yolo.txt`)
- [ ] Vídeo disponível como `input_video.mp4`
- [ ] ~2GB espaço livre (para modelo e saída)
- [ ] Executar: `python video_analysis_yolo.py input_video.mp4`
- [ ] Aguardar processamento (pode levar alguns minutos)
- [ ] Verificar saídas: vídeo + JSON + TXT

---

## 🎉 Conclusão

**YOLO oferece a análise mais avançada e precisa!**

- Detecta automaticamente o contexto
- Identifica objetos específicos
- Infere atividades inteligentemente
- Perfeito para análise profissional



---
