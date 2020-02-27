import cv2
import numpy as np
import auxiliar as aux

cap = cv2.VideoCapture(0)

# Cores a serem identificadas
pink = '#ff004d'
blue = '#0173fe'

while(True):
    
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if ret == False:
        print("Codigo de retorno FALSO - problema para capturar o frame")

    # Our operations on the frame come here
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
   
    # Rosa
    cap_pink1, cap_pink2 = aux.ranges(pink)
    mask_pink = cv2.inRange(hsv, cap_pink1, cap_pink2)

    # Blue 
    cap_blue1, cap_blue2 = aux.ranges(blue)
    mask_blue = cv2.inRange(hsv, cap_blue1, cap_blue2)
    
    # Construindo a máscara de ambos
    mask = mask_blue + mask_pink
    
    # Colocando a cor AZUL
    segmentado_cor_azul = cv2.morphologyEx(mask_blue,cv2.MORPH_CLOSE,np.ones((10, 10)))
    selecao_blue = cv2.bitwise_and(frame, frame, mask=segmentado_cor_azul)
    
    # Colocando a cor ROSA
    segmentado_cor_rosa = cv2.morphologyEx(mask_pink,cv2.MORPH_CLOSE,np.ones((10, 10)))
    selecao_pink = cv2.bitwise_and(frame, frame, mask=segmentado_cor_rosa)
    
    # Fazendo a seleção de ambos
    selecao = selecao_pink + selecao_blue
    
    # CONTORNO ROSA
    segmentado_pink = cv2.morphologyEx(mask_pink,cv2.MORPH_CLOSE,np.ones((4, 4)))
    img_out_pink, contornos_pink, arvore_rosa = cv2.findContours(segmentado_pink.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contornos_img_pink = frame.copy()
    cv2.drawContours(contornos_img_pink, contornos_pink, -1, [0, 0, 255], 3);
    
    maior_pink = None
    maior_area_pink = 0
    for c in contornos_pink:
        area_pink = cv2.contourArea(c)
        if area_pink > maior_area_pink:
            maior_area_pink = area_pink
            maior_pink = c
            
    cv2.drawContours(contornos_img_pink, [maior_pink], -1, [0, 255, 255], 5);
    
    # CONTORNO AZUL
    segmentado_blue = cv2.morphologyEx(mask_blue,cv2.MORPH_CLOSE,np.ones((4, 4)))
    img_out_blue, contornos_blue, arvore_rosa = cv2.findContours(segmentado_blue.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contornos_img_blue = frame.copy()
    cv2.drawContours(contornos_img_blue, contornos_blue, -1, [0, 0, 255], 3);
    
    maior_blue = None
    maior_area_blue = 0
    for c in contornos_blue:
        area_blue = cv2.contourArea(c)
        if area_blue > maior_area_blue:
            maior_area_blue = area_blue
            maior_blue = c
            
    cv2.drawContours(contornos_img_blue, [maior_blue], -1, [0, 255, 255], 5);
    
    cv2.imshow("contornos_img_blue", contornos_img_blue)  
    
    # Display the resulting frame
    cv2.imshow('gray', gray)
    cv2.imshow("mask", mask)
    cv2.imshow("selecao", selecao)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()