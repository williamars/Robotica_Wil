import cv2
import numpy as np
import auxiliar as aux

cap = cv2.VideoCapture(0)

# Cores a serem identificadas
pink = '#ff004d'
blue = '#0173fe'

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


while(True):
    
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if ret == False:
        print("Codigo de retorno FALSO - problema para capturar o frame")

    # Our operations on the frame come here
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
    
    #
    blur = cv2.GaussianBlur(gray,(5,5),0) 
    bordas = auto_canny(blur)

    circles = []

    # Obtains a version of the edges image where we can draw in color
    bordas_color = cv2.cvtColor(bordas, cv2.COLOR_GRAY2BGR)

    # HoughCircles - detects circles using the Hough Method. For an explanation of
    circles = None
    circles=cv2.HoughCircles(bordas,cv2.HOUGH_GRADIENT,2,40,param1=50,param2=100,minRadius=5,maxRadius=60)

    if circles is not None:        
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            print(i)
            # draw the outer circle
            # cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]])
            cv2.circle(bordas_color,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(bordas_color,(i[0],i[1]),2,(0,0,255),3)
   
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
    
    cv2.imshow("selecao", selecao)
    
    # CONTORNO do ROSA
    segmentado_pink = cv2.morphologyEx(mask_pink,cv2.MORPH_CLOSE,np.ones((4, 4)))
    img_out_pink, contornos_pink, arvore_rosa = cv2.findContours(segmentado_pink.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contornos = frame.copy()
    cv2.drawContours(contornos, contornos_pink, -1, [0, 0, 255], 3);
    
    maior_pink = None
    maior_area_pink = 0
    for c in contornos_pink:
        area_pink = cv2.contourArea(c)
        if area_pink > maior_area_pink:
            maior_area_pink = area_pink
            maior_pink = c
    
    # CONTORNO do AZUL
    segmentado_blue = cv2.morphologyEx(mask_blue,cv2.MORPH_CLOSE,np.ones((4, 4)))
    img_out_blue, contornos_blue, arvore_rosa = cv2.findContours(segmentado_blue.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contornos, contornos_blue, -1, [0, 0, 255], 3);
    
    maior_blue = None
    maior_area_blue = 0
    for c in contornos_blue:
        area_blue = cv2.contourArea(c)
        if area_blue > maior_area_blue:
            maior_area_blue = area_blue
            maior_blue = c
           
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(contornos,'q to quit',(0,50), font, 1,(255,255,255),2,cv2.LINE_AA)
    
    cv2.imshow("contornos", contornos)
    cv2.imshow("circles", bordas_color)

    # Fechar as abas se apertar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()