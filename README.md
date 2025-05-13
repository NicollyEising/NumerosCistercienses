# Conversor de Números Cistercienses

Este projeto implementa um sistema de conversão bidirecional entre números arábicos (1 a 9999) e números cistercienses, com suporte a entrada via imagem e geração gráfica.

## Funcionalidades

- **Conversão de imagem cisterciense para número arábico**  
- **Geração de imagem cisterciense a partir de número arábico**  
- **Extração de número arábico de imagem usando OCR (Tesseract)**  
- **Interface gráfica simples para seleção de arquivos**  
- **Visualização com OpenCV**  

## Requisitos

- Python 3.7+  
- OpenCV 
- NumPy  
- pytesseract  
- Tesseract OCR

## Instalação

1. Clone o repositório:  
   ```bash
   git clone https://github.com/NicollyEising/NumerosCistercienses.git
   ```
2. Instale as dependências 

3. Instale o [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) e configure o caminho no script:  
   ```python
   pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
   ```

4. Rodar uvicorn: 
   ```uvicorn api:app --reload
   ```
5. Ao rodar o codigo python siga a interface gráfica para seleção de arquivos
