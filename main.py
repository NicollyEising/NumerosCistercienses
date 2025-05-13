import cv2
import numpy as np
import pytesseract
import tkinter as tk
from tkinter import filedialog


def is_segment_present(binary, x1, y1, x2, y2, unit, roi=None):
    line_width = max(1, int(unit * 0.4))
    line_mask = np.zeros_like(binary, dtype=np.uint8)
    cv2.line(line_mask, (x1, y1), (x2, y2), 255, line_width)

    if roi is not None:
        rx1, ry1, rx2, ry2 = roi
        roi_mask = np.zeros_like(binary, dtype=np.uint8)
        cv2.rectangle(roi_mask, (rx1, ry1), (rx2, ry2), 255, -1)
        line_mask = cv2.bitwise_and(line_mask, roi_mask)

    overlap = cv2.bitwise_and(binary, line_mask)

    overlap_pixels   = np.count_nonzero(overlap)
    expected_pixels  = np.count_nonzero(line_mask)
    coverage_ratio   = overlap_pixels / expected_pixels if expected_pixels > 0 else 0

    ys, xs = np.where(overlap > 0)
    detected_length = np.hypot(xs.max() - xs.min(), ys.max() - ys.min()) if xs.size else 0
    expected_length = np.hypot(x2 - x1, y2 - y1)

    coverage_th = 0.35
    length_th   = 0.40
    return coverage_ratio > coverage_th and detected_length >= length_th * expected_length


def detect_central_line(binary_img, debug=False):
    """
    Tenta localizar a linha vertical central.
    1º  – HoughLinesP (mais robusto a quebras);
    2º  – contornos como fallback.
    Retorna (x1,y1,x2,y2) ou None.
    """
    h_img, w_img = binary_img.shape

    # ---------- Hough ----------
    lines = cv2.HoughLinesP(binary_img, 1, np.pi/180,
                            threshold=120,
                            minLineLength=int(h_img * 0.6),
                            maxLineGap=20)
    best = None
    if lines is not None:
        best_score = 1e9
        for l in lines:
            x1,y1,x2,y2 = l[0]
            slope = abs((x2 - x1) / (y2 - y1 + 1e-6))
            if slope > 0.20:        # não muito vertical
                continue
            cx_line = (x1 + x2) / 2
            score   = abs(cx_line - w_img / 2)  # perto do meio = score baixo
            if score < best_score:
                best_score = score
                best       = (x1, y1, x2, y2)

    if best is not None:
        if debug:
            print("Linha central por Hough.")
        return best

    # ---------- Fallback: contornos ----------
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_score = -1e9
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        aspect  = w / h
        if aspect > 0.35:
            continue
        cx_cnt  = x + w/2
        score   = h - 2*abs(cx_cnt - w_img/2)
        if score > best_score:
            best_score = score
            best       = (x, y, x+w, y+h)

    if best and debug:
        print("Linha central por contorno.")
    return best


def cistercian_to_arabic(image: np.ndarray, debug: bool = False) -> int:
    """
    Converte uma imagem de número cisterciense para inteiro arábico.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # --- pré-processamento básico ---
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    _, binary = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # dilatação leve para unir falhas
    binary = cv2.dilate(binary, np.ones((3,3), np.uint8), 1)

    # fechamento morfológico “alto” para reconstruir a linha vertical
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                         (1, int(gray.shape[0]*0.15)))
    binary_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, v_kernel)

    # --- localizar linha central ---
    central = detect_central_line(binary_closed, debug)
    if central is None:
        if debug:
            print("Nenhuma linha central detectada – retornando 0.")
        return 0

    x1,y1,x2,y2 = central
    cx = image.shape[1] // 2
    cy = int((y1 + y2) / 2)
    height = abs(y2 - y1)
    unit   = height / 8.0
    dbg = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.line(dbg, (cx, 0), (cx, image.shape[0]), (0, 0, 255), 3)  # Vermelho
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if debug:
        dbg = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.line(dbg, (x1,y1), (x2,y2), (0,0,255), 3)
        cv2.waitKey(0); cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    #  padrões dos segmentos / decodificação (igual ao que você já tinha)
    # ------------------------------------------------------------------
    patterns = { 
        0: [],
        1: [((0, -3.8), (1, -3.8))],
        2: [((0, -2.5), (1, -2.5))],
        3: [((0, -3.8), (1, -2.5))],
        4: [((0, -2.5), (1, -3.8))],
        5: [((0, -2.5), (1, -3.8)), ((0, -3.8), (1, -3.8))],
        6: [((1, -3.8), (1, -2.5))],
        7: [((0, -3.8), (1, -3.8)), ((1, -3.8), (1, -2.5))],
        8: [((0, -2.5), (1, -2.5)), ((1, -3.8), (1, -2.5))],
        9: [((0, -2.5), (1, -2.5)), ((1, -3.8), (1, -2.5)),
            ((0, -3.8), (1, -3.8))]
    }

    sorted_patterns = sorted(patterns.items(), key=lambda x: -len(x[1]))
    digits = {1:0,2:0,3:0,4:0}

    quad_size = int(4 * unit)
    quadrant_names = {1:"unidade",2:"centena",3:"milhar",4:"dezena"}

    for quadrant in (3,2,4,1):
        mx = 1 if quadrant in (1,4) else -1
        my = 1 if quadrant in (1,2) else -1

        # ROI do quadrante
        if quadrant in (1,2):  y_min,y_max = cy - quad_size, cy
        else:                  y_min,y_max = cy,            cy + quad_size
        if quadrant in (1,4):  x_min,x_max = cx,            cx + quad_size
        else:                  x_min,x_max = cx - quad_size, cx

        x_min = max(0, x_min); x_max = min(binary.shape[1]-1, x_max)
        y_min = max(0, y_min); y_max = min(binary.shape[0]-1, y_max)
        roi = (int(x_min), int(y_min), int(x_max), int(y_max))

        best_digit, best_score = 0, -1.0
        for digit, segs in sorted_patterns:
            if not segs:
                continue
            found = 0
            for (dx1,dy1),(dx2,dy2) in segs:
                x_s = int(cx + mx*dx1*unit)
                y_s = int(cy + my*dy1*unit)
                x_e = int(cx + mx*dx2*unit)
                y_e = int(cy + my*dy2*unit)
                if is_segment_present(binary_closed, x_s,y_s,x_e,y_e, unit, roi):
                    found += 1
            score = found / len(segs)
            if score > best_score:
                best_score,best_digit = score,digit
            if best_score == 1.0:
                break

        digits[quadrant] = best_digit if best_score == 1.0 else 0

    arabic_number = (digits[3]*1000 +
                     digits[4]*100  +
                     digits[2]*10   +
                     digits[1])

    if debug:
        print("Quadrantes →", digits)
        print("Número final =", arabic_number)
    return arabic_number

def get_quadrant_roi(cx, cy, unit, quadrant, img_shape, margin=0.3):
    """
    Retorna (xmin, ymin, xmax, ymax) que limita a área do quadrante.
    margin –– parcela extra de unit para “folga” (0.3 ≈ 30 %)
    """
    h_half = 4 * unit                                  # altura/largura de um quadrante
    x_min = cx - h_half if quadrant in [2, 3] else cx
    x_max = cx        if quadrant in [2, 3] else cx + h_half
    y_min = cy - h_half if quadrant in [1, 2] else cy
    y_max = cy        if quadrant in [1, 2] else cy + h_half

    # aplica margem e garante que não sairá do frame
    x_min = max(0, int(x_min - margin * unit))
    y_min = max(0, int(y_min - margin * unit))
    x_max = min(img_shape[1]-1, int(x_max + margin * unit))
    y_max = min(img_shape[0]-1, int(y_max + margin * unit))
    return x_min, y_min, x_max, y_max


## Arabico para cisterciense

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def arabic_to_cistercian_image(number: str, img_height=400, img_width=200) -> np.ndarray:
    """
    Converte um número arábico (string) de 1 a 9999 em imagem com representação cisterciense.
    """
    if not number.isdigit():
        raise ValueError("Entrada inválida: apenas números inteiros positivos são permitidos.")
    
    valor = int(number)
    if not (1 <= valor <= 9999):
        raise ValueError("Número fora do intervalo suportado (1 a 9999).")

    # Criar imagem branca
    img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
    center_x, center_y = img_width // 2, img_height // 2

    # Traço vertical central
    cv2.line(img, (center_x, 50), (center_x, img_height - 50), (0, 0, 0), 3)

    # Escala proporcional
    unit = img_height // 10

    # Padrões de traços para cada dígito (em coordenadas relativas)
    patterns = { 
        0: [],
        1: [((0, -3.8), (1, -3.8))],
        2: [((0, -2.5), (1, -2.5))],
        3: [((0, -3.8), (1, -2.5))],
        4: [((0, -2.5), (1, -3.8))],
        5: [((0, -2.5), (1, -3.8)), ((0, -3.8), (1, -3.8))],
        6: [((1, -3.8), (1, -2.5))],
        7: [((0, -3.8), (1, -3.8)), ((1, -3.8), (1, -2.5))],
        8: [((0, -2.5), (1, -2.5)), ((1, -3.8), (1, -2.5))],
        9: [((0, -2.5), (1, -2.5)), ((1, -3.8), (1, -2.5)),
            ((0, -3.8), (1, -3.8))]
    }


    # Quadrantes: 1: sup dir, 2: sup esq, 3: inf esq, 4: inf dir
    mirrors = {
        1: (-1, -1),  # sup. direita → milhar
        2: (1, -1), # sup. esquerda → centena
        3: (-1, 1),  # inf. esquerda → dezena
        4: (1, 1),   # inf. direita → unidade
    }

    def draw_segment(quadrant: int, digit: int):
        if digit == 0:
            return

        mx, my = mirrors[quadrant]
        for (dx1, dy1), (dx2, dy2) in patterns.get(digit, []):
            x1 = center_x + mx * dx1 * unit
            y1 = center_y + my * dy1 * unit
            x2 = center_x + mx * dx2 * unit
            y2 = center_y + my * dy2 * unit
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), 5)

    # Processar os 4 dígitos
    num_str = number.zfill(4)
    milhares, centenas, dezenas, unidades = map(int, num_str)

    draw_segment(1, milhares)
    draw_segment(2, centenas)
    draw_segment(3, dezenas)
    draw_segment(4, unidades)
    
    nome_arquivo = f"cistercian_{num_str}.png"
    cv2.imwrite(nome_arquivo, img)
    print(f"Imagem salva como: {nome_arquivo}")

    return img


def preprocess_image(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(contrast, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def extract_number_from_image(image_path: str) -> str:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")
    
    processed = preprocess_image(image)
    config = '--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789'
    extracted_text = pytesseract.image_to_string(processed, config=config)
    return extracted_text.strip()


def gerar_imagem_numero_arabico(numero: int):
    # Define tamanho da imagem
    largura, altura = 400, 200
    imagem = np.ones((altura, largura, 3), dtype=np.uint8) * 255  # fundo branco

    # Define a fonte e posicionamento do número
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    escala = 4
    cor = (0, 0, 0)  # preto
    espessura = 5
    texto = str(numero)
    tamanho_texto = cv2.getTextSize(texto, fonte, escala, espessura)[0]
    pos_x = (largura - tamanho_texto[0]) // 2
    pos_y = (altura + tamanho_texto[1]) // 2

    # Renderiza o texto na imagem
    cv2.putText(imagem, texto, (pos_x, pos_y), fonte, escala, cor, espessura, cv2.LINE_AA)

    # Exibe a imagem
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return imagem





# (mantém aqui as importações e configurações já existentes, até antes de arabic_to_cistercian_image)

def arabic_to_cistercian_quadrant_images(number: str,
                                         img_height=400,
                                         img_width=200) -> dict:
    """
    Converte um número arábico (string) em quatro imagens cistercienses,
    uma para cada quadrante (milhar, centena, dezena, unidade).
    Retorna um dict: {'milhar': img1, 'centena': img2, ...}.
    """
    if not number.isdigit():
        raise ValueError("Entrada inválida: apenas dígitos.")
    valor = int(number)
    if not (0 <= valor <= 9999):
        raise ValueError("Fora do intervalo 0–9999.")
    
    # Padrões e espelhamentos (mesmos da função original)
    patterns = { 
        0: [],
        1: [((0, -3.8), (1, -3.8))],
        2: [((0, -2.5), (1, -2.5))],
        3: [((0, -3.8), (1, -2.5))],
        4: [((0, -2.5), (1, -3.8))],
        5: [((0, -2.5), (1, -3.8)), ((0, -3.8), (1, -3.8))],
        6: [((1, -3.8), (1, -2.5))],
        7: [((0, -3.8), (1, -3.8)), ((1, -3.8), (1, -2.5))],
        8: [((0, -2.5), (1, -2.5)), ((1, -3.8), (1, -2.5))],
        9: [((0, -2.5), (1, -2.5)), ((1, -3.8), (1, -2.5)),
            ((0, -3.8), (1, -3.8))]
    }
    mirrors = {
        1: (-1, -1),  # sup. direita → milhar
        2: (1, -1), # sup. esquerda → centena
        3: (-1, 1),  # inf. esquerda → dezena
        4: (1, 1),   # inf. direita → unidade
    }
    
    # Extrai dígitos com zero à esquerda
    m, c, d, u = map(int, number.zfill(4))
    digit_map = {1: m, 2: c, 3: d, 4: u}
    label = {1: 'milhar', 2: 'centena', 3: 'dezena', 4: 'unidade'}
    
    # Função auxiliar para desenhar linha central e segmentos
    def make_base():
        img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
        cx, cy = img_width // 2, img_height // 2
        cv2.line(img, (cx, 50), (cx, img_height - 50), (0, 0, 0), 3)
        unit = img_height // 10
        return img, cx, cy, unit
    
    def draw_segment(img, cx, cy, unit, quadrant, digit):
        if digit == 0:
            return
        mx, my = mirrors[quadrant]
        for (dx1, dy1), (dx2, dy2) in patterns.get(digit, []):
            x1 = int(cx + mx * dx1 * unit)
            y1 = int(cy + my * dy1 * unit)
            x2 = int(cx + mx * dx2 * unit)
            y2 = int(cy + my * dy2 * unit)
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), 3)
    
    # Gera as quatro imagens
    images = {}
    for quadrant in (1, 2, 3, 4):
        img, cx, cy, unit = make_base()
        draw_segment(img, cx, cy, unit, quadrant, digit_map[quadrant])
        images[label[quadrant]] = img
    
    return images



def main():
    num = ""
    root = tk.Tk()
    root.withdraw()

    print("Selecione:\n[1] Arabic -> Cisterciense\n[2] Cisterciense -> Arabic")
    escolha = input("Digite sua escolha (1 ou 2): ")

    if escolha == '1':
        print("Selecione:\n[1] Digitar número\n[2] Inserir imagem")
        escolha1 = input("Digite sua escolha (1 ou 2): ")

        if escolha1 == '1':
            numDigitado = input("Digite o número: ")
            if not numDigitado.isdigit():
                print("Número inválido.")
                return
            num = numDigitado
        elif escolha1 == '2':
            caminho_imagem = filedialog.askopenfilename(
                title="Selecione uma imagem",
                filetypes=[("Arquivos de imagem", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
            )
            if not caminho_imagem:
                print("Nenhuma imagem selecionada.")
                return
            print("Caminho da imagem selecionada:", caminho_imagem)

            number_str = extract_number_from_image(caminho_imagem)
            if not number_str.isdigit():
                print("Nenhum número detectado na imagem.")
                return
            print(f"Número detectado por OCR: {number_str}")
            num = number_str
        else:
            print("Selecione uma opção válida.")
            return

        print("\nGerando imagem cisterciense correspondente...")
        cistercian_img = arabic_to_cistercian_image(num)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif escolha == '2':
        caminho_imagem = filedialog.askopenfilename(
            title="Selecione uma imagem",
            filetypes=[("Arquivos de imagem", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
        )
        if not caminho_imagem:
            print("Nenhuma imagem selecionada.")
            return
        print(f"Carregando imagem: {caminho_imagem}")
        image = cv2.imread(caminho_imagem)

        print("\nProcessando como número cisterciense...")
        arabic_number = cistercian_to_arabic(image)
        print(f"Número detectado: {arabic_number}")
        num = str(arabic_number)
        gerar_imagem_numero_arabico(arabic_number)

    else:
        print("Digite uma opção válida (1 ou 2).")
        return

    # Geração das imagens por quadrante
    quad_imgs = arabic_to_cistercian_quadrant_images(num)
    for nome, img in quad_imgs.items():
        filename = f"{nome}_{num}.png"
        cv2.imwrite(filename, img)
        print(f"Salvo: {filename}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\n--- Fim da Execução ---")


if __name__ == "__main__":
    main()
