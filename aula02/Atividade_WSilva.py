import cv2
import numpy as np
import auxiliar as aux
import time

cap = cv2.VideoCapture(0)

'''Parte 1'''
h = 258 #px
H = 14 #cm
D = 30 #cm

f = (h*D)/H
print(f'A distância focal é {f} px')

# Cores a serem identificadas
pink = '#ff004d'
blue = '#0173fe'

brisk = cv2.BRISK_create()

# Configura o algoritmo de casamento de features que vê *como* o objeto que deve ser encontrado aparece na imagem
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

# Define o mínimo de pontos similares
MINIMO_SEMELHANCAS = 18


def find_good_matches(descriptor_image1, frame_gray):
    """
        Recebe o descritor da imagem a procurar e um frame da cena, e devolve os keypoints e os good matches
    """
    des1 = descriptor_image1
    kp2, des2 = brisk.detectAndCompute(frame_gray,None)

    # Tenta fazer a melhor comparacao usando o algoritmo
    matches = bf.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    return kp2, good

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    return edged

def center_of_contour(contorno):
    M = cv2.moments(contorno)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return (int(cX), int(cY))
    else:
        return (0,150)

def crosshair(img, point, size, color):
    """ Desenha um crosshair centrado no point.
        point deve ser uma tupla (x,y)
        color é uma tupla R,G,B uint8
    """
    x,y = point
    cv2.line(img,(x - size,y),(x + size,y),color,5)
    cv2.line(img,(x,y - size),(x, y + size),color,5)

def texto(img, a, p):
    """Escreve na img RGB dada a string a na posição definida pela tupla p"""
    cv2.putText(img, str(a), p, font,1,(0,50,100),2,cv2.LINE_AA)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if ret == False:
        print("Codigo de retorno FALSO - problema para capturar o frame")

    # Our operations on the frame come here
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
    
    blur = cv2.GaussianBlur(gray,(5,5),0) 
    bordas = auto_canny(blur)
    circles = []
    bordas_color = cv2.cvtColor(bordas, cv2.COLOR_GRAY2BGR)
   
    # Rosa
    cap_pink1, cap_pink2 = aux.ranges(pink)
    mask_pink = cv2.inRange(hsv, cap_pink1, cap_pink2)

    # Blue 
    cap_blue1, cap_blue2 = aux.ranges(blue)
    mask_blue = cv2.inRange(hsv, cap_blue1, cap_blue2)
    
    # Construindo a máscara de ambos
    mask = mask_blue + mask_pink
    
    # Colocando a cor AZUL
    segmentado_cor_blue = cv2.morphologyEx(mask_blue,cv2.MORPH_CLOSE,np.ones((10, 10)))
    selecao_blue = cv2.bitwise_and(frame, frame, mask=segmentado_cor_blue)
    
    # Colocando a cor ROSA
    segmentado_cor_rosa = cv2.morphologyEx(mask_pink,cv2.MORPH_CLOSE,np.ones((10, 10)))
    selecao_pink = cv2.bitwise_and(frame, frame, mask=segmentado_cor_rosa)
    
    # Fazendo a seleção de ambos
    selecao = selecao_pink + selecao_blue  
    
    # HoughCircles
    circles = None
    circles=cv2.HoughCircles(bordas,cv2.HOUGH_GRADIENT,2,40,param1=50,param2=105,minRadius=5,maxRadius=60)

    if circles is not None:        
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            print(i)
            cv2.circle(selecao,(i[0],i[1]),i[2],(0,255,0),2)
            # Desenha o centro do círculo
            cv2.circle(selecao,(i[0],i[1]),2,(0,0,255),3)
    
    # Colocando a imagem das seleções
    cv2.imshow("mask", selecao)
    
    # CONTORNO do ROSA
    segmentado_pink = cv2.morphologyEx(mask_pink,cv2.MORPH_CLOSE,np.ones((4, 4)))
    img_out_pink, contornos_pink, arvore_rosa = cv2.findContours(segmentado_pink.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contorno_img_pink = frame.copy()
    cv2.drawContours(contorno_img_pink, contornos_pink, -1, [0, 0, 255], 3);
    
    maior_pink = None
    maior_area_pink = 0
    for c in contornos_pink:
        area_pink = cv2.contourArea(c)
        if area_pink > maior_area_pink:
            maior_area_pink = area_pink
            maior_pink = c
    try:        
        cv2.drawContours(contorno_img_pink, [maior_pink], -1, [0, 0, 255], 3);
    except:
        pass
    
    # CONTORNO do AZUL
    segmentado_blue = cv2.morphologyEx(mask_blue,cv2.MORPH_CLOSE,np.ones((4, 4)))
    img_out_blue, contornos_blue, arvore_rosa = cv2.findContours(segmentado_blue.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contorno_img_blue = frame.copy()
    cv2.drawContours(contorno_img_blue, contornos_blue, -1, [0, 0, 255], 3);
    
    maior_blue = None
    maior_area_blue = 0
    for c in contornos_blue:
        area_blue = cv2.contourArea(c)
        if area_blue > maior_area_blue:
            maior_area_blue = area_blue
            maior_blue = c
    try:
        cv2.drawContours(contorno_img_blue, [maior_blue], -1, [0, 0, 255], 3);
    except:
        pass
            
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.imshow("contornos azul", contorno_img_blue)
    cv2.imshow("contornos pink", contorno_img_pink)
    
    '''Código para identificar a imagem do Insper'''
    insper = cv2.imread('insper.png')
    insper_gray = cv2.cvtColor(insper, cv2.COLOR_BGR2GRAY)
    
    # Encontra os pontos únicos (keypoints) nas duas imagems
    kp1, des1 = brisk.detectAndCompute(insper, None)
    try:
        kp2, good_matches = find_good_matches(des1, gray)
    except:
        pass
        
    if len(good_matches) > MINIMO_SEMELHANCAS:
        img3 = cv2.drawMatches(insper, kp1, frame, kp2, good_matches, None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow('BRISK features', img3)
    else:
        cv2.imshow("BRISK features", frame)
      
    maskframepink = cv2.cvtColor(mask_pink,cv2.COLOR_GRAY2RGB)
    contorno_img_pink = maskframepink.copy()
    
    maskframeblue = cv2.cvtColor(mask_blue,cv2.COLOR_GRAY2RGB)
    contorno_img_blue = maskframeblue.copy()
    
    '''Parte 3'''
    for i in contornos_pink:
        area_rosa = cv2.contourArea(i)
        centro_pink = center_of_contour(i)
        crosshair(contorno_img_pink, centro_pink, 20, (128,128,0))
        texto(contorno_img_pink, np.round(area_rosa,2), centro_pink)
        
    for i in contornos_blue:
        area_blue = cv2.contourArea(i)
        centro_blue = center_of_contour(i)
        crosshair(contorno_img_blue, centro_blue, 20, (128,128,0))
        texto(contorno_img_blue, np.round(area_blue,2), centro_blue)
    
    def distancia_entre(centro1, centro2):
        distancia = np.sqrt((centro1[0]-centro2[0])**2 + (centro1[1]-centro2[1])**2)
        return distancia

    distancia = distancia_entre(centro_blue, centro_pink)
    
    '''Parte 1 da atividade novamente'''
    f = 552.85
    h = distancia
    H = 14

    if h!= 0:
        D = f*H/h
        
    def angulo(centro1, centro2):
        if (centro1[0] - centro2[0]) != 0:
            angulo = np.arctan((centro1[1]-centro2[1])/(centro1[0]-centro2[0]))
            angulo2 = np.degrees(angulo)
            return np.fabs(angulo2)
        else:
            return "-"
    
    contorno_total = contorno_img_blue + contorno_img_pink
    
    contorno2 = cv2.cvtColor(contorno_total, cv2.COLOR_BGR2GRAY)
    angulo_entre = angulo(centro_blue, centro_pink)
    cv2.line(contorno2,centro_blue,centro_pink,(255,0,0),5)
    
     
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(contorno2,'Distancia folha tela eh {0:.2f}cm'.format(D),(0,50), font, 1,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(contorno2,'Angulo eh {0} graus'.format(angulo_entre),(0,100), font, 1,(255,255,255),2,cv2.LINE_AA)
    
    cv2.imshow('contorno_total', contorno2)
        
    # Fechar as abas se apertar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()