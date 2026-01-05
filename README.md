# Sistema de Análise de Vídeo com IA

Sistema completo para análise de vídeos com **Reconhecimento Facial**, **Detecção de Emoções**, **Análise de Atividades** e **Detecção de Anomalias**.

## 📋 Índice

- [Funcionalidades](#funcionalidades)
- [Requisitos](#requisitos)
- [Instalação](#instalação)
- [Como Usar](#como-usar)
- [Saídas Geradas](#saídas-geradas)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Detalhes Técnicos](#detalhes-técnicos)

## ✨ Funcionalidades

### 1. Reconhecimento Facial
- Detecta e marca todos os rostos presentes em cada frame do vídeo
- Utiliza Haar Cascades do OpenCV para detecção robusta
- Desenha retângulos verdes ao redor de cada rosto identificado

### 2. Análise de Expressões Emocionais
- Analisa emoções baseadas em características faciais
- Categorias detectadas: Feliz, Triste, Neutro, Surpreso
- Exibe a emoção de cada rosto identificado no vídeo processado

### 3. Detecção de Atividades Detalhadas
Sistema avançado que classifica atividades específicas baseado em múltiplos fatores:
- **Análise de movimento regional**: Avalia movimento em diferentes áreas do frame
- **Detecção de mãos/pele**: Identifica regiões de mãos para análise contextual
- **Postura corporal**: Detecta corpos superiores e posicionamento

**Categorias de Atividades Detectadas:**
- **Conversando / Ocioso**: Pessoa presente com baixo a moderado movimento, sem atividade específica
- **Trabalhando (PC)**: Movimento concentrado na região média, indicando uso de computador
- **Lendo / Estudando**: Movimento muito baixo e estável, foco visual concentrado
- **Usando Celular**: Mão próxima ao rosto, movimento concentrado na região superior

### 4. Detecção de Anomalias
Sistema inteligente que identifica comportamentos atípicos:
- **Movimento Brusco**: Picos repentinos de atividade
- **Inatividade Súbita**: Queda drástica de movimento
- **Mudança de Pessoas**: Variação significativa no número de rostos

### 5. Geração de Resumo Automático
- Relatório JSON com todas as métricas
- Relatório em texto formatado
- Estatísticas completas de emoções e atividades

## 🔧 Requisitos

### Software Necessário
- Python 3.7 ou superior
- pip (gerenciador de pacotes Python)

### Bibliotecas Python
```
opencv-python==4.8.1.78
numpy==1.24.3
```

## 📦 Instalação

### Passo 1: Clone ou baixe o projeto

```bash
git clone <seu-repositorio>
cd video-analysis-system
```

### Passo 2: Crie um ambiente virtual (recomendado)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Passo 3: Instale as dependências

```bash
pip install -r requirements.txt
```

**Conteúdo do requirements.txt:**
```
opencv-python==4.8.1.78
numpy==1.24.3
```

Ou instale manualmente:
```bash
pip install opencv-python numpy
```

## 🚀 Como Usar

### Preparação do Vídeo

1. Baixe o vídeo do Google Drive fornecido
2. Coloque o vídeo na pasta do projeto com o nome `input_video.mp4`

Ou use qualquer vídeo especificando o caminho:

### Execução Básica

```bash
python video_analysis.py
```

Este comando processa o arquivo `input_video.mp4` no diretório atual.

### Execução com Caminho Personalizado

```bash
python video_analysis.py /caminho/para/seu/video.mp4
```

### Visualização em Tempo Real (Opcional)

Para ver o processamento em tempo real, edite a linha no código:

```python
show_preview=True  # Altere de False para True
```

Pressione `Q` para encerrar a visualização antecipadamente.

## 📊 Saídas Geradas

Após a execução, o sistema gera três arquivos principais:

### 1. `video_analisado.mp4`
Vídeo processado contendo:
- Retângulos verdes ao redor dos rostos detectados
- Labels com emoções identificadas
- Marcadores de anomalias (em vermelho)
- Informações de frame, número de rostos e atividade

### 2. `relatorio_analise.json`
Relatório completo em formato JSON com:

```json
{
  "metadata": {
    "video_path": "input_video.mp4",
    "data_analise": "2026-01-05T...",
    "fps": 30.0,
    "resolucao": "1920x1080"
  },
  "metricas_gerais": {
    "total_frames_analisados": 1500,
    "duracao_video_segundos": 50.0,
    "total_rostos_detectados": 3200,
    "media_rostos_por_frame": 2.13,
    "numero_anomalias_detectadas": 8
  },
  "emocoes_detectadas": {
    "Neutro": 1800,
    "Feliz": 1200,
    "Triste": 150,
    "Surpreso": 50
  },
  "atividades_detectadas": {
    "Conversando / Ocioso": {
      "frames": 1042,
      "porcentagem": 69.5
    },
    "Trabalhando (PC)": {
      "frames": 234,
      "porcentagem": 15.6
    },
    "Lendo / Estudando": {
      "frames": 81,
      "porcentagem": 5.4
    },
    "Usando Celular": {
      "frames": 50,
      "porcentagem": 3.3
    }
  },
  "anomalias": [
    {
      "frame": 245,
      "timestamp": 8.17,
      "type": "Movimento Brusco",
      "motion_intensity": 0.234,
      "num_faces": 3
    }
  ],
  "resumo": [...]
}
```

### 3. `relatorio_analise.txt`
Relatório formatado em texto para fácil leitura:

```
================================================================================
RELATÓRIO DE ANÁLISE DE VÍDEO
================================================================================

INFORMAÇÕES DO VÍDEO
--------------------------------------------------------------------------------
Vídeo: input_video.mp4
Data da Análise: 2026-01-05T14:30:00
FPS: 30.0
Resolução: 1920x1080

MÉTRICAS GERAIS
--------------------------------------------------------------------------------
Total Frames Analisados: 1500
Duração Video Segundos: 50.0
Total Rostos Detectados: 3200
Média Rostos Por Frame: 2.13
Número Anomalias Detectadas: 8

EMOÇÕES DETECTADAS
--------------------------------------------------------------------------------
Neutro: 1800 detecções
Feliz: 1200 detecções
Triste: 150 detecções
Surpreso: 50 detecções

ATIVIDADES DETECTADAS (COM PORCENTAGENS)
--------------------------------------------------------------------------------
• 69.5% - Conversando / Ocioso (1042 frames)
• 15.6% - Trabalhando (PC) (234 frames)
• 5.4% - Lendo / Estudando (81 frames)
• 3.3% - Usando Celular (50 frames)

RESUMO EXECUTIVO
--------------------------------------------------------------------------------
Emoção predominante: Neutro (1800 detecções)

Distribuição de Atividades:
  • 69.5% - Conversando / Ocioso (1042 frames)
  • 15.6% - Trabalhando (PC) (234 frames)
  • 5.4% - Lendo / Estudando (81 frames)
  • 3.3% - Usando Celular (50 frames)

Total de anomalias: 8
Tipos de anomalias detectadas:
  - Movimento Brusco: 5 ocorrências
  - Mudança de Pessoas na Cena: 3 ocorrências

DETALHES DAS ANOMALIAS
--------------------------------------------------------------------------------
Anomalia 1:
  Frame: 245
  Timestamp: 8.17s
  Tipo: Movimento Brusco
  Intensidade de Movimento: 0.234
  Número de Rostos: 3
```

## 📁 Estrutura do Projeto

```
video-analysis-system/
│
├── video_analysis.py          # Código principal
├── requirements.txt           # Dependências Python
├── README.md                  # Este arquivo
│
├── input_video.mp4           # Vídeo de entrada (você adiciona)
│
└── Saídas geradas:
    ├── video_analisado.mp4    # Vídeo processado
    ├── relatorio_analise.json # Relatório JSON
    └── relatorio_analise.txt  # Relatório texto
```

## 🔬 Detalhes Técnicos

### Reconhecimento Facial
- **Método**: Haar Cascade Classifier (OpenCV)
- **Modelo**: `haarcascade_frontalface_default.xml`
- **Parâmetros**: 
  - Scale Factor: 1.1
  - Min Neighbors: 5
  - Min Size: 30x30 pixels

### Análise de Emoções
Baseada em:
- Análise de histograma da região facial
- Intensidade média e desvio padrão de pixels
- Detecção de olhos para contexto adicional

**Algoritmo de Classificação:**
```python
if desvio_padrão > 50 and olhos >= 2:
    if intensidade_média > 120: → Feliz
    elif intensidade_média < 80: → Triste
    else: → Neutro
elif olhos < 2: → Surpreso
else: → Neutro
```

### Detecção de Atividades Detalhadas

O sistema usa uma combinação de técnicas para identificar atividades específicas:

**1. Conversando / Ocioso** - Detectado quando:
- Pessoa presente na cena (rosto detectado)
- Movimento baixo a moderado (intensidade < 0.15)
- Sem padrões específicos de outras atividades
- Movimento distribuído pela cena

**2. Trabalhando (PC)** - Detectado quando:
- Movimento concentrado na região média do frame
- Intensidade de movimento moderada (0.02-0.15)
- Rosto presente na cena
- Padrão consistente de movimento na mesma região

**3. Lendo / Estudando** - Detectado quando:
- Movimento muito baixo e estável (< 0.05)
- Rosto presente e em posição fixa
- Movimento concentrado na região superior
- Baixa variabilidade no movimento ao longo do tempo

**4. Usando Celular** - Detectado quando:
- Mão detectada próxima ao rosto (< 150 pixels)
- Movimento concentrado na região superior
- Padrão característico de mão elevada

**Algoritmo de Classificação:**
```python
Prioridade de detecção:
1. Usando Celular (mão próxima ao rosto + movimento superior)
2. Trabalhando PC (movimento região média + consistência)
3. Lendo/Estudando (baixo movimento + estabilidade)
4. Conversando/Ocioso (padrão default com rosto presente)
```

### Detecção de Anomalias

**Critérios de Detecção:**

1. **Movimento Brusco**:
   - Intensidade atual > média + 2.5 × desvio padrão
   - Considera janela temporal de 30 frames

2. **Inatividade Súbita**:
   - Intensidade < 0.005 quando média recente > 0.05
   - Indica parada repentina

3. **Mudança de Pessoas**:
   - Variação > 2 no número de rostos detectados
   - Comparado com média dos últimos 10 frames

### Performance
- Processamento em tempo real possível em hardware moderno
- ~30 FPS em vídeo 1080p (processador i5 ou superior)
- Uso de memória: ~200-500 MB

## ⚙️ Configurações Avançadas

### Ajustar Sensibilidade de Anomalias

No código `video_analysis.py`, método `detect_anomaly`:

```python
# Movimento brusco - aumentar 2.5 para menos sensível
if motion_intensity > mean_motion + 2.5 * std_motion:

# Mudança de pessoas - aumentar 2 para menos sensível  
if abs(num_faces - mean_faces) > 2:
```

### Ajustar Detecção Facial

```python
faces = self.face_cascade.detectMultiScale(
    gray, 
    scaleFactor=1.1,    # Reduzir para mais detecções (ex: 1.05)
    minNeighbors=5,     # Reduzir para mais detecções (ex: 3)
    minSize=(30, 30)    # Reduzir para detectar rostos menores
)
```

## 🐛 Solução de Problemas

### Erro: "Vídeo não encontrado"
- Verifique se o arquivo existe no caminho especificado
- Certifique-se de usar o caminho completo ou relativo correto

### Erro: "No module named cv2"
```bash
pip install --upgrade opencv-python
```

### Vídeo de saída não reproduz
- Instale um player compatível com codec H.264 (VLC, MPC-HC)
- Ou altere o codec no código:
```python
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Tente XVID
```

### Poucos rostos detectados
- Aumente a sensibilidade da detecção facial (veja Configurações Avançadas)
- Verifique a qualidade e iluminação do vídeo

### Muitas anomalias falsas
- Aumente o threshold no método `detect_anomaly`
- Aumente a janela temporal (`maxlen` do `motion_history`)

## 📝 Notas Importantes

1. **Observação sobre Anomalias**: O sistema define como anômalo qualquer movimento que não segue o padrão geral de atividades, incluindo:
   - Gestos bruscos ou repentinos
   - Comportamentos atípicos comparados ao histórico recente
   - Mudanças drásticas na cena

2. **Privacidade**: Este sistema não armazena identificações faciais, apenas detecta presença e analisa emoções.

3. **Precisão**: A análise de emoções é baseada em características visuais básicas. Para maior precisão, considere integrar modelos de deep learning específicos.

4. **Performance**: O tempo de processamento depende da duração e resolução do vídeo, além das capacidades do hardware.

## 📞 Suporte

Para problemas ou dúvidas:
1. Verifique a seção de Solução de Problemas acima
2. Revise os logs de erro gerados
3. Consulte a documentação do OpenCV: https://docs.opencv.org

## 📄 Licença

Este projeto foi desenvolvido para fins educacionais como parte do Tech Challenge.

---

**Desenvolvido para análise avançada de vídeo com técnicas de Computer Vision e IA**
