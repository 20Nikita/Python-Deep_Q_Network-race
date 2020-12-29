import pygame
import sys
import math
pygame.init()

class Игра(object):
    def __init__(self):
        self.Попыток = 0
        self.Трасса = [[140, 290], [245, 180], [680, 55], [970, 40], [1290, 90],
                  [1425, 230], [1450, 380], [1410, 470], [1300, 500], [1180, 490],
                  [1120, 440], [1050, 310], [950, 230], [845, 240], [800, 300],
                  [880, 460], [1010, 525], [1270, 530], [1360, 600], [1280, 780],
                  [810, 730], [574, 440], [470, 410], [300, 720], [220, 730],
                  [110, 700], [90, 580]]
        self.Трасса1 = [[320, 230], [690, 125], [970, 115], [1240, 170], [1330, 290],
                  [1335, 375], [1220, 380], [1140, 230], [1000, 130], [790,130],
                  [680, 240], [685, 340], [850, 570], [980, 630], [1240,630],
                  [1200, 680], [900, 640], [630, 300], [460, 260], [350, 350],
                  [200, 580], [240, 300]]
        self.pos = [260,160]
        self.Угол = -73
        self.Скорость = 0
        self.Растояние = 0
        self.Попыток = 0
        self.Среда = [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]
        self.Лучей = 16

        pass

    def Обновить(self):
        self.Трасса = [[140, 290], [245, 180], [680, 55], [970, 40], [1290, 90],
                  [1425, 230], [1450, 380], [1410, 470], [1300, 500], [1180, 490],
                  [1120, 440], [1050, 310], [950, 230], [845, 240], [800, 300],
                  [880, 460], [1010, 525], [1270, 530], [1360, 600], [1280, 780],
                  [810, 730], [574, 440], [470, 410], [300, 720], [220, 730],
                  [110, 700], [90, 580]]
        self.Трасса1 = [[320, 230], [690, 125], [970, 115], [1240, 170], [1330, 290],
                  [1335, 375], [1220, 380], [1140, 230], [1000, 130], [790,130],
                  [680, 240], [685, 340], [850, 570], [980, 630], [1240,630],
                  [1200, 680], [900, 640], [630, 300], [460, 260], [350, 350],
                  [200, 580], [240, 300]]
        self.pos = [260,160]
        self.Угол = -73
        self.Скорость = 0
        self.Растояние = 0

    def Поворот(self, Картинка, ш, в):
        pivot = pygame.math.Vector2(ш/2, -в/2)
        pivot_rotate = pivot.rotate(self.Угол)
        pivot_move   = pivot_rotate - pivot
        Машина = pygame.transform.rotate(Картинка, self.Угол)
        box = [pygame.math.Vector2(p) for p in [(0, 0), (ш, 0), (ш, -в), (0, -в)]]
        box_rotate = [p.rotate(self.Угол) for p in box]
        min_box = (min(box_rotate, key=lambda p: p[0])[0], min(box_rotate, key=lambda p: p[1])[1])
        max_box = (max(box_rotate, key=lambda p: p[0])[0], max(box_rotate, key=lambda p: p[1])[1])
        Координаты = (self.pos[0] + min_box[0] - pivot_move[0], self.pos[1] - max_box[1] + pivot_move[1])
        return (Машина,Координаты)

    def intersection(self, A, B, C, d):
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

    def Врезались(self, ш, в):
        Лучей = 16
        цeнтр = (self.pos[0]+ш/2,self.pos[1]+в/2)
        A = (цeнтр[0] - в/2*math.sin(math.radians(self.Угол)) - ш/2*math.cos(math.radians(self.Угол)),цeнтр[1] - в/2*math.cos(math.radians(self.Угол)) + ш/2*math.sin(math.radians(self.Угол)))
        B = (цeнтр[0] - в/2*math.sin(math.radians(self.Угол)) + ш/2*math.cos(math.radians(self.Угол)),цeнтр[1] - в/2*math.cos(math.radians(self.Угол)) - ш/2*math.sin(math.radians(self.Угол)))
        C = (цeнтр[0] + в/2*math.sin(math.radians(self.Угол)) + ш/2*math.cos(math.radians(self.Угол)),цeнтр[1] + в/2*math.cos(math.radians(self.Угол)) - ш/2*math.sin(math.radians(self.Угол)))
        D = (цeнтр[0] + в/2*math.sin(math.radians(self.Угол)) - ш/2*math.cos(math.radians(self.Угол)),цeнтр[1] + в/2*math.cos(math.radians(self.Угол)) + ш/2*math.sin(math.radians(self.Угол)))

        for i in range(len(self.Трасса)-1):
            if self.intersection(self.Трасса[i], self.Трасса[i+1], A, B) or self.intersection(self.Трасса[i], self.Трасса[i+1], B, C) or\
               self.intersection(self.Трасса[i], self.Трасса[i+1], C, D) or self.intersection(self.Трасса[i], self.Трасса[i+1], D, C):
               return True
           #for i in range(Лучей):


        if self.intersection(self.Трасса[len(self.Трасса)-1], self.Трасса[0], A, B) or self.intersection(self.Трасса[len(self.Трасса)-1], self.Трасса[0], B, C) or\
               self.intersection(self.Трасса[len(self.Трасса)-1], self.Трасса[0], C, D) or self.intersection(self.Трасса[len(self.Трасса)-1], self.Трасса[0], D, C):
               return True
        for i in range(len(self.Трасса1)-1):
            if self.intersection(self.Трасса1[i], self.Трасса1[i+1], A, B) or self.intersection(self.Трасса1[i], self.Трасса1[i+1], B, C) or\
               self.intersection(self.Трасса1[i], self.Трасса1[i+1], C, D) or self.intersection(self.Трасса1[i], self.Трасса1[i+1], D, C):
               return True
        if self.intersection(self.Трасса1[len(self.Трасса1)-1], self.Трасса1[0], A, B) or self.intersection(self.Трасса1[len(self.Трасса1)-1], self.Трасса1[0], B, C) or\
               self.intersection(self.Трасса1[len(self.Трасса1)-1], self.Трасса1[0], C, D) or self.intersection(self.Трасса1[len(self.Трасса1)-1], self.Трасса1[0], D, C):
               return True    
        return False

    def Лучи (self, ш, в):
        self.Среда = [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]
        цeнтр = (self.pos[0]+ш/2,self.pos[1]+в/2)
        for j in range(self.Лучей):
            for i in range(len(self.Трасса)-1):
                t = self.intersection(self.Трасса[i], self.Трасса[i+1], (цeнтр[0],цeнтр[1]),
                                      (цeнтр[0] - 200 * math.sin(math.radians(self.Угол + j*360/self.Лучей)),
                                      цeнтр[1] - 200 * math.cos(math.radians(self.Угол + j*360/self.Лучей))))
                if t and (((self.Среда[j][0]-цeнтр[0])*(self.Среда[j][0]-цeнтр[0])+(self.Среда[j][1]-цeнтр[1])*(self.Среда[j][1]-цeнтр[1]))>((цeнтр[0]-t[0])*(цeнтр[0]-t[0])+(цeнтр[1]-t[1])*(цeнтр[1]-t[1]))):
                    self.Среда[j] = t
            for i in range(len(self.Трасса1)-1):
                t = self.intersection(self.Трасса1[i], self.Трасса1[i+1], (цeнтр[0],цeнтр[1]),
                                      (цeнтр[0] - 200 * math.sin(math.radians(self.Угол + j*360/self.Лучей)),
                                      цeнтр[1] - 200 * math.cos(math.radians(self.Угол + j*360/self.Лучей))))
                if t and (((self.Среда[j][0]-цeнтр[0])*(self.Среда[j][0]-цeнтр[0])+(self.Среда[j][1]-цeнтр[1])*(self.Среда[j][1]-цeнтр[1]))>((цeнтр[0]-t[0])*(цeнтр[0]-t[0])+(цeнтр[1]-t[1])*(цeнтр[1]-t[1]))):
                    self.Среда[j] = t

            t = self.intersection(self.Трасса[len(self.Трасса)-1], self.Трасса[0], (цeнтр[0],цeнтр[1]),
                                      (цeнтр[0] - 200 * math.sin(math.radians(self.Угол + j*360/self.Лучей)),
                                      цeнтр[1] - 200 * math.cos(math.radians(self.Угол + j*360/self.Лучей))))
            if t and (((self.Среда[j][0]-цeнтр[0])*(self.Среда[j][0]-цeнтр[0])+(self.Среда[j][1]-цeнтр[1])*(self.Среда[j][1]-цeнтр[1]))>((цeнтр[0]-t[0])*(цeнтр[0]-t[0])+(цeнтр[1]-t[1])*(цeнтр[1]-t[1]))):
                self.Среда[j] = t            
            t = self.intersection(self.Трасса1[len(self.Трасса1)-1], self.Трасса1[0], (цeнтр[0],цeнтр[1]),
                                      (цeнтр[0] - 200 * math.sin(math.radians(self.Угол + j*360/self.Лучей)),
                                      цeнтр[1] - 200 * math.cos(math.radians(self.Угол + j*360/self.Лучей))))
            if t and (((self.Среда[j][0]-цeнтр[0])*(self.Среда[j][0]-цeнтр[0])+(self.Среда[j][1]-цeнтр[1])*(self.Среда[j][1]-цeнтр[1]))>((цeнтр[0]-t[0])*(цeнтр[0]-t[0])+(цeнтр[1]-t[1])*(цeнтр[1]-t[1]))):
                self.Среда[j] = t

    def Управление(self,w,s,a,d):
        МаксСкВп = 12
        МаксСкНаз = -2
        Ускорение = 0.1
        Трение = 0.01
        if a:
            self.Угол+=1*abs(self.Скорость)*0.5+0.2
        if d:
            self.Угол-=1*abs(self.Скорость)*0.5+0.2      
        if s and self.Скорость > МаксСкНаз:
           self.Скорость-=Ускорение*3
        elif w and self.Скорость < МаксСкВп:
           self.Скорость += Ускорение
        elif abs(self.Скорость) > 0:  
            if self.Скорость > 0:
                self.Скорость-=Трение
            else:
                self.Скорость+=Трение
        
    def Графика(self, Окно, Картинка, Начало, Видно_Дорогу = True, Видно_Лилии = True, Видно_Круги = True, Видно_Дданные = True):
        ЦветФона = (50, 50, 50)
        ЦветЛиний = (255,255,255)
        ш, в = Картинка.get_size()
        цeнтр = (self.pos[0]+ш/2,self.pos[1]+в/2)

        pygame.font.init()
        Шрифт = pygame.font.SysFont('courier',24)

        Окно.fill(ЦветФона)

        if Видно_Дорогу:
            pygame.draw.aalines(Окно, ЦветЛиний, True, self.Трасса)
            pygame.draw.aalines(Окно, ЦветЛиний, True, self.Трасса1)

        if Видно_Лилии:
            for i in range(self.Лучей):
                pygame.draw.aaline(Окно, ЦветЛиний, (цeнтр[0],цeнтр[1]), (цeнтр[0] - 200 * math.sin(math.radians(self.Угол + i*360/self.Лучей)),цeнтр[1] - 200*math.cos(math.radians(self.Угол + i*360/self.Лучей))))
        if Видно_Круги:
            for i in range(len(self.Среда)):
                pygame.draw.circle(Окно, ЦветЛиний, self.Среда[i], 5)

        Конец = pygame.time.get_ticks()            
        self.Растояние+=self.Скорость*(Конец-Начало)/1000
        Текст = Шрифт.render(f"Пройденный путь: {int(self.Растояние)} Попыток: {self.Попыток}",0,ЦветЛиний)
        Окно.blit(Текст, (20, 20))
        if Видно_Дданные:            
            for i in range(len(self.Среда)):
                t = int((((self.Среда[i][0]-цeнтр[0])*(self.Среда[i][0]-цeнтр[0])+(self.Среда[i][1]-цeнтр[1])*(self.Среда[i][1]-цeнтр[1])))/1000)
                if t > 40:
                    t = 40
                Текст = Шрифт.render(f"{i} = {t}",0,ЦветЛиний)
                Окно.blit(Текст, (20, 20*i+40))

        Окно.blit(self.Поворот(Картинка, ш, в)[0],self.Поворот(Картинка, ш, в)[1])
        pygame.display.flip()
    
    def Запуск(self, Д = 1500,В = 800):
        #Создание окна
        Окно = pygame.display.set_mode((Д,В))
        pygame.display.set_caption("Машинка")
        Картинка = pygame.image.load('Машинка1.png')
        ш, в = Картинка.get_size()
        
        Играем = 1
        while Играем:
            Начало = pygame.time.get_ticks()
            #Пробек по всем сабытиям
            for event in pygame.event.get():
                #Нажата кнопка выход
                if event.type == pygame.QUIT:
                    Играем = 0
                keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                self.Управление(0,0,1,0)
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                self.Управление(0,0,0,1) 
            if (keys[pygame.K_DOWN] or keys[pygame.K_s]):
               self.Управление(0,1,0,0)
            if (keys[pygame.K_UP] or keys[pygame.K_w]):
               self.Управление(1,0,0,0)
            self.Управление(0,0,0,0)
            self.pos[0] -= self.Скорость*math.sin(math.radians(self.Угол))
            self.pos[1] -= self.Скорость*math.cos(math.radians(self.Угол))
           
            self.Лучи(ш, в)
            if self.Врезались(ш, в):
                print('Проиграли')
                self.Обновить()
                self.Попыток+=1
            else:
                self.Графика(Окно,Картинка,Начало)

            Конец = pygame.time.get_ticks()            
            pygame.time.delay(30-Конец+Начало)

            if Играем == 0:
                print('Закрыли')
                pygame.quit()

Играем = 1
игра = Игра()
while Играем:
    a = int(input())
    if a == 1:
        Играем = 0
        sys.exit()
    if a==2:
        print('Запуск')
        игра.Обновить()
        игра.Запуск()
"""
Играем1 = 1
игра = Игра()
while Играем1:
    a = int(input())
    if a == 1:
        Играем1 = 0
        sys.exit()
    if a==2:
        print('Запуск')
        игра.Обновить()
        Играем = 1
        while Играем:
            Д = 1500
            В = 800
            Окно = pygame.display.set_mode((Д,В))
            pygame.display.set_caption("Машинка")
            Картинка = pygame.image.load('Машинка.png')
            ш, в = Картинка.get_size()
            Начало = pygame.time.get_ticks()
            #Пробек по всем сабытиям
            for event in pygame.event.get():
                #Нажата кнопка выход
                if event.type == pygame.QUIT:
                    Играем = 0
                keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                игра.Управление(0,0,1,0)
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                игра.Управление(0,0,0,1) 
            if (keys[pygame.K_DOWN] or keys[pygame.K_s]):
               игра.Управление(0,1,0,0)
            if (keys[pygame.K_UP] or keys[pygame.K_w]):
               игра.Управление(1,0,0,0)
            игра.Управление(0,0,0,0)
            игра.pos[0] -= игра.Скорость*math.sin(math.radians(игра.Угол))
            игра.pos[1] -= игра.Скорость*math.cos(math.radians(игра.Угол))
           
            игра.Лучи(ш, в)
            if игра.Врезались(ш, в):
                print('Проиграли')
                игра.Обновить()
                игра.Попыток+=1
            else:
                игра.Графика(Окно,Картинка,Начало)

            Конец = pygame.time.get_ticks()            
            pygame.time.delay(30-Конец+Начало)

            if Играем == 0:
                print('Закрыли')
                pygame.quit()
"""

