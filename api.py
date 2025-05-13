import base64
from typing import Dict
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import numpy as np
import cv2
import io
import shutil
import os
from main import cistercian_to_arabic, arabic_to_cistercian_image, extract_number_from_image, arabic_to_cistercian_quadrant_images, gerar_imagem_numero_arabico

app = FastAPI()


class ArabicNumberRequest(BaseModel):
    number: str
    
@app.post("/upload-image-arabic")
async def process_cistercian_image(imagem: UploadFile = File(...)):
    extensoes_permitidas = (".png", ".jpg", ".jpeg", ".bmp", ".gif")

    if not imagem.filename.lower().endswith(extensoes_permitidas):
        raise HTTPException(status_code=400, detail="Formato de imagem não suportado.")

    temp_path = f"temp_{imagem.filename}"

    try:
        # Salva o arquivo temporariamente
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(imagem.file, buffer)

        # Lê e decodifica a imagem com OpenCV
        with open(temp_path, "rb") as f:
            img_bytes = f.read()
        image_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Falha ao decodificar a imagem.")

        # Converte imagem cisterciense em número arábico
        numero_extraido = cistercian_to_arabic(img)
        num = str(numero_extraido)
        # Gera imagem cisterciense completa
        numero_arabico = gerar_imagem_numero_arabico(numero_extraido)

        # Gera imagens dos quadrantes
        imgs_quadrantes = arabic_to_cistercian_quadrant_images(num)

        def encode_image_to_base64(image: np.ndarray) -> str:
            _, buffer = cv2.imencode('.png', image)
            return base64.b64encode(buffer).decode('utf-8')

        # Codifica as imagens em base64
        imagens_base64: Dict[str, str] = {
            "completa": encode_image_to_base64(img),
            "numero_arabico": encode_image_to_base64(numero_arabico)
        }
        for nome, img_q in imgs_quadrantes.items():
            imagens_base64[str(nome)] = encode_image_to_base64(img_q)

        return JSONResponse(content={
            "numero": str(numero_extraido),
            "imagens": imagens_base64
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/arabic-to-cistercian")
def convert_image_to_cistercian_local(imagem: UploadFile = File(...)):
    if not imagem.filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
        raise HTTPException(status_code=400, detail="Formato de imagem não suportado.")

    caminho_temp = f"temp_{imagem.filename}"
    with open(caminho_temp, "wb") as f:
        shutil.copyfileobj(imagem.file, f)

    numero_extraido = extract_number_from_image(caminho_temp)
    os.remove(caminho_temp)

    if not numero_extraido.isdigit():
        raise HTTPException(status_code=400, detail="Nenhum número válido detectado na imagem.")

    imagem_cisterciense = arabic_to_cistercian_image(numero_extraido)
    imgs_quadrantes = arabic_to_cistercian_quadrant_images(numero_extraido)

    def encode_image_to_base64(image) -> str:
        _, buffer = cv2.imencode(".png", image)
        return base64.b64encode(buffer).decode("utf-8")

    imagens_base64: Dict[str, str] = {
        "completa": encode_image_to_base64(imagem_cisterciense)
    }

    for nome, img in imgs_quadrantes.items():
        imagens_base64[nome] = encode_image_to_base64(img)

    return JSONResponse(content={"numero": numero_extraido, "imagens": imagens_base64})

class NumeroEntrada(BaseModel):
    numero: str

@app.post("/convertString")
def convert_arabic_to_cistercian_string(entrada: NumeroEntrada):
    numero = entrada.numero

    if not numero.isdigit():
        raise HTTPException(status_code=400, detail="Número inválido.")

    imagem_cisterciense = arabic_to_cistercian_image(numero)
    imgs_quadrantes = arabic_to_cistercian_quadrant_images(numero)

    def encode_image_to_base64(image) -> str:
        _, buffer = cv2.imencode(".png", image)
        return base64.b64encode(buffer).decode("utf-8")

    imagens_base64: Dict[str, str] = {
        "completa": encode_image_to_base64(imagem_cisterciense)
    }

    for nome, img in imgs_quadrantes.items():
        imagens_base64[nome] = encode_image_to_base64(img)

    return JSONResponse(content={"numero": numero, "imagens": imagens_base64})