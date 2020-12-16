import pygame
import sys
import math
pygame.init()

ЦветФона = (50, 50, 50)
Д = 1500
В = 800
Угол = -73
Скорость = 0
pos = [260,160]
Трасса = [[140, 290], [245, 180], [680, 55], [970, 40], [1290, 90],
          [1425, 230], [1450, 380], [1410, 470], [1300, 500], [1180, 490],
          [1120, 440], [1050, 310], [950, 230], [845, 240], [800, 300],
          [880, 460], [1010, 525], [1270, 530], [1360, 600], [1280, 780],
          [810, 730], [574, 440], [470, 410], [300, 720], [220, 730],
          [110, 700], [90, 580]]
Трасса1 = [[320, 230], [690, 125], [970, 115], [1240, 170], [1330, 290],
          [1335, 375], [1220, 380], [1140, 230], [1000, 130], [790,130],
          [680, 240], [685, 340], [850, 570], [980, 630], [1240,630],
          [1200, 680], [900, 640], [630, 300], [460, 260], [350, 350],
          [200, 580], [240, 300]]

МаксСкВп = 6
МаксСкНаз = -1
Ускорение = 0.01
Трение = 0.005
#Создание окна
Окно = pygame.display.set_mode((Д,В))
pygame.display.set_caption("Машинка")
Картинка = pygame.image.load('Машинка.png')
Картинка2 = pygame.image.load('Машинка.png').convert()

ш, в = Картинка.get_size()

def Поворот(Картинка,Угол):
    pivot = pygame.math.Vector2(ш/2, -в/2)
    pivot_rotate = pivot.rotate(Угол)
    pivot_move   = pivot_rotate - pivot
    Машина = pygame.transform.rotate(Картинка, Угол)
    box = [pygame.math.Vector2(p) for p in [(0, 0), (ш, 0), (ш, -в), (0, -в)]]
    box_rotate = [p.rotate(Угол) for p in box]
    min_box = (min(box_rotate, key=lambda p: p[0])[0], min(box_rotate, key=lambda p: p[1])[1])
    max_box = (max(box_rotate, key=lambda p: p[0])[0], max(box_rotate, key=lambda p: p[1])[1])
    Координаты = (pos[0] + min_box[0] - pivot_move[0], pos[1] - max_box[1] + pivot_move[1])
    return (Машина,Координаты)

def intersection(A, B, C, d):
    L1 = [0,0,0]
    L2 = [0,0,0]
    L2[0] = (A[1] - B[1])
    L2[1] = (B[0] - A[0])
    L2[2] = -(A[0]*B[1] - B[0]*A[1])
    L1[0] = (C[1] - d[1])
    L1[1] = (d[0] - C[0])
    L1[2] = -(C[0]*d[1] - d[0]*C[1])

    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        
        if min(A[0], B[0]) <= x <= max(A[0], B[0]) and min(C[0], d[0]) <= x <= max(C[0], d[0]) and \
           min(A[1], B[1]) <= y <= max(A[1], B[1]) and min(C[1], d[1]) <= y <= max(C[1], d[1]):
            return x,y
        else:
            return False
    else:
        return False

def Врезались():
    цeнтр = (pos[0]+ш/2,pos[1]+в/2)
    A = (цeнтр[0] - в/2*math.sin(math.radians(Угол)) - ш/2*math.cos(math.radians(Угол)),цeнтр[1] - в/2*math.cos(math.radians(Угол)) + ш/2*math.sin(math.radians(Угол)))
    B = (цeнтр[0] - в/2*math.sin(math.radians(Угол)) + ш/2*math.cos(math.radians(Угол)),цeнтр[1] - в/2*math.cos(math.radians(Угол)) - ш/2*math.sin(math.radians(Угол)))
    C = (цeнтр[0] + в/2*math.sin(math.radians(Угол)) + ш/2*math.cos(math.radians(Угол)),цeнтр[1] + в/2*math.cos(math.radians(Угол)) - ш/2*math.sin(math.radians(Угол)))
    D = (цeнтр[0] + в/2*math.sin(math.radians(Угол)) - ш/2*math.cos(math.radians(Угол)),цeнтр[1] + в/2*math.cos(math.radians(Угол)) + ш/2*math.sin(math.radians(Угол)))

    for i in range(len(Трасса)-1):
        if intersection(Трасса[i], Трасса[i+1], A, B) or intersection(Трасса[i], Трасса[i+1], B, C) or\
           intersection(Трасса[i], Трасса[i+1], C, D) or intersection(Трасса[i], Трасса[i+1], D, C):
           return True
    if intersection(Трасса[len(Трасса)-1], Трасса[0], A, B) or intersection(Трасса[len(Трасса)-1], Трасса[0], B, C) or\
           intersection(Трасса[len(Трасса)-1], Трасса[0], C, D) or intersection(Трасса[len(Трасса)-1], Трасса[0], D, C):
           return True
    for i in range(len(Трасса1)-1):
        if intersection(Трасса1[i], Трасса1[i+1], A, B) or intersection(Трасса1[i], Трасса1[i+1], B, C) or\
           intersection(Трасса1[i], Трасса1[i+1], C, D) or intersection(Трасса1[i], Трасса1[i+1], D, C):
           return True
    if intersection(Трасса1[len(Трасса1)-1], Трасса1[0], A, B) or intersection(Трасса1[len(Трасса1)-1], Трасса1[0], B, C) or\
           intersection(Трасса1[len(Трасса1)-1], Трасса1[0], C, D) or intersection(Трасса1[len(Трасса1)-1], Трасса1[0], D, C):
           return True
    
    return False

pygame.display.update()

Играем = 1
while Играем:
    #Пробек по всем сабытиям
    for event in pygame.event.get():
        #Нажата кнопка выход
        if event.type == pygame.QUIT:
            print('Закрыли')
            pygame.quit()
            Играем = 0
            sys.exit()
        keys = pygame.key.get_pressed()

    if keys[pygame.K_LEFT] or keys[pygame.K_a]:
        Угол+=1*abs(Скорость)*0.5+0.2
    if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        Угол-=1*abs(Скорость)*0.5+0.2

    
    
    if (keys[pygame.K_DOWN] or keys[pygame.K_s]) and Скорость > МаксСкНаз:
       Скорость-=Ускорение*3
    elif (keys[pygame.K_UP] or keys[pygame.K_w]) and Скорость < МаксСкВп:
       Скорость+=Ускорение

    elif abs(Скорость) > 0:  
        if Скорость > 0:
            Скорость-=Трение
        else:
            Скорость+=Трение
    
        
    pos[0] -= Скорость*math.sin(math.radians(Угол))
    pos[1] -= Скорость*math.cos(math.radians(Угол))

    #Закраска окна
    Окно.fill(ЦветФона)
    pygame.draw.aalines(Окно, (255,255,255), True, Трасса)
    pygame.draw.aalines(Окно, (255,255,255), True, Трасса1)
    
    if Врезались():
        Окно.blit(Поворот(Картинка2,Угол)[0],Поворот(Картинка2,Угол)[1])
    else:
        Окно.blit(Поворот(Картинка,Угол)[0],Поворот(Картинка,Угол)[1])


    pygame.display.flip()
    pygame.time.delay(5)
