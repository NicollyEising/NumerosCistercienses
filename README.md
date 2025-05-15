
# Conversor de Números Cistercienses

Este projeto implementa um sistema de conversão bidirecional entre números arábicos (1 a 9999) e números cistercienses, com suporte a entrada via imagem e geração gráfica.

## Participantes
Nicolly Munhoz Eising e Nahuel Isaias Ayala Molinas

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

### 1. Clone o repositório
```bash
git clone https://github.com/NicollyEising/NumerosCistercienses.git
cd NumerosCistercienses
```

### 2. Instale as dependências Python
```bash
pip install -r requirements.txt
```

###  3. Acesso ao diretorio: 
   ```
   cd .\NumerosCistercienses
   ```

### 4. Rodar uvicorn: 
   ```
   uvicorn api:app --reload
   ```


### 5. Instale o [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) caso houver erros pytesseract:  
   ```python
   pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
   ```
   ou Configure o caminho no script principal, se necessário:
   ```python
   pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
   ```

### 6. Acessar my-app:
   ```
   cd my-app
   ```

### 7. Rodar npm:
   ```
   npm start
   ```

### 8 Caso de erro de script:
   ```
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   ```
### 9. Ao rodar o codigo python siga a interface gráfica para seleção de arquivos
