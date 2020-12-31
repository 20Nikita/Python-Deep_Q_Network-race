import pygame
import sys
import math
pygame.init()

import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import pickle

#Если видеокарта доступна используем её
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Дивайс: ", device)
print("esc - меню")

class Игра(object):

    def __init__(self):
        self.Попыток = 0
        self.Трасса = [[140, 290], [245, 180], [680, 55], [970, 40], [1290, 90],
                  [1425, 230], [1450, 380], [1410, 470], [1300, 500], [1180, 490],
                  [1120, 440], [1050, 310], [950, 230], [845, 240],[830, 310],[1050, 510],
                  [1100, 620], [1050, 720],[950, 760],[810, 720],
                  [574, 440], [470, 410], [370, 450], [390, 600],[350, 700], [180, 730],
                  [70, 700], [60, 580]]
        self.Трасса1 = [[320, 230], [690, 125], [970, 115], [1240, 170], [1330, 290],
                  [1335, 375], [1220, 380], [1140, 230], [1000, 130], [790,130],
                  [700, 240], [700, 300],[790, 430],  [980,600],
                  [950, 640], [900, 640], [630, 300], [460, 260], [250, 350],
                  [210, 580], [200, 580], [240, 300]]
        self.pos = [260,160]
        self.Угол = -73
        self.Скорость = 0
        self.Растояние = 0
        self.rr = 0
        self.Попыток = 0
        self.Лучей = 16
        self.Среда = [0,0]*self.Лучей
        self.Картинка = pygame.image.load('Машинка.png')
        self.ш, self.в = self.Картинка.get_size()
        pass

    def Обновить(self):
        
        self.pos = [260,170]
        self.Угол = -73
        self.Скорость = 0
        self.Растояние = 0
        self.Длинна = 1000
        self.Чекпоинт = 0

        Окружение = []
        #Окружение.append(self.Скорость)
        #Окружение.append(self.Угол)
        #Окружение.append(self.Растояние)
        #Окружение.append(self.Чекпоинт)
        self.Лучи()
        ш, в = self.Картинка.get_size()
        цeнтр = (self.pos[0]+ш/2,self.pos[1]+в/2)
        for i in range(len(self.Среда)):
                t = (((self.Среда[i][0]-цeнтр[0])*(self.Среда[i][0]-цeнтр[0])+(self.Среда[i][1]-цeнтр[1])*(self.Среда[i][1]-цeнтр[1])))/1000
                if t > self.Длинна:
                    t = self.Длинна
                Окружение.append(t)

        return Окружение

    def Поворот(self):
        pivot = pygame.math.Vector2(self.ш/2, -self.в/2)
        pivot_rotate = pivot.rotate(self.Угол)
        pivot_move   = pivot_rotate - pivot
        Машина = pygame.transform.rotate(self.Картинка, self.Угол)
        box = [pygame.math.Vector2(p) for p in [(0, 0), (self.ш, 0), (self.ш, -self.в), (0, -self.в)]]
        box_rotate = [p.rotate(self.Угол) for p in box]
        min_box = (min(box_rotate, key=lambda p: p[0])[0], min(box_rotate, key=lambda p: p[1])[1])
        max_box = (max(box_rotate, key=lambda p: p[0])[0], max(box_rotate, key=lambda p: p[1])[1])
        Координаты = (self.pos[0] + min_box[0] - pivot_move[0], self.pos[1] - max_box[1] + pivot_move[1])
        return Машина, Координаты

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

    def Врезались(self):
        цeнтр = (self.pos[0]+self.ш/2,self.pos[1]+self.в/2)
        A = (цeнтр[0] - self.в/2*math.sin(math.radians(self.Угол)) - self.ш/2*math.cos(math.radians(self.Угол)),цeнтр[1] - self.в/2*math.cos(math.radians(self.Угол)) + self.ш/2*math.sin(math.radians(self.Угол)))
        B = (цeнтр[0] - self.в/2*math.sin(math.radians(self.Угол)) + self.ш/2*math.cos(math.radians(self.Угол)),цeнтр[1] - self.в/2*math.cos(math.radians(self.Угол)) - self.ш/2*math.sin(math.radians(self.Угол)))
        C = (цeнтр[0] + self.в/2*math.sin(math.radians(self.Угол)) + self.ш/2*math.cos(math.radians(self.Угол)),цeнтр[1] + self.в/2*math.cos(math.radians(self.Угол)) - self.ш/2*math.sin(math.radians(self.Угол)))
        D = (цeнтр[0] + self.в/2*math.sin(math.radians(self.Угол)) - self.ш/2*math.cos(math.radians(self.Угол)),цeнтр[1] + self.в/2*math.cos(math.radians(self.Угол)) + self.ш/2*math.sin(math.radians(self.Угол)))

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

    def Лучи (self):
        self.Среда = [[10000,10000]]*self.Лучей
        цeнтр = (self.pos[0]+self.ш/2,self.pos[1]+self.в/2)
        for j in range(self.Лучей):
            for i in range(len(self.Трасса)-1):
                t = self.intersection(self.Трасса[i], self.Трасса[i+1], (цeнтр[0],цeнтр[1]),
                                      (цeнтр[0] - self.Длинна * math.sin(math.radians(self.Угол-90 + j*360/self.Лучей/2)),
                                      цeнтр[1] - self.Длинна * math.cos(math.radians(self.Угол-90 + j*360/self.Лучей/2))))
                if t and (((self.Среда[j][0]-цeнтр[0])*(self.Среда[j][0]-цeнтр[0])+(self.Среда[j][1]-цeнтр[1])*(self.Среда[j][1]-цeнтр[1]))>((цeнтр[0]-t[0])*(цeнтр[0]-t[0])+(цeнтр[1]-t[1])*(цeнтр[1]-t[1]))):
                    self.Среда[j] = t
            for i in range(len(self.Трасса1)-1):
                t = self.intersection(self.Трасса1[i], self.Трасса1[i+1], (цeнтр[0],цeнтр[1]),
                                      (цeнтр[0] - self.Длинна * math.sin(math.radians(self.Угол-90 + j*360/self.Лучей/2)),
                                      цeнтр[1] - self.Длинна * math.cos(math.radians(self.Угол-90 + j*360/self.Лучей/2))))
                if t and (((self.Среда[j][0]-цeнтр[0])*(self.Среда[j][0]-цeнтр[0])+(self.Среда[j][1]-цeнтр[1])*(self.Среда[j][1]-цeнтр[1]))>((цeнтр[0]-t[0])*(цeнтр[0]-t[0])+(цeнтр[1]-t[1])*(цeнтр[1]-t[1]))):
                    self.Среда[j] = t

            t = self.intersection(self.Трасса[len(self.Трасса)-1], self.Трасса[0], (цeнтр[0],цeнтр[1]),
                                      (цeнтр[0] - self.Длинна * math.sin(math.radians(self.Угол-90 + j*360/self.Лучей/2)),
                                      цeнтр[1] - self.Длинна * math.cos(math.radians(self.Угол-90 + j*360/self.Лучей/2))))
            if t and (((self.Среда[j][0]-цeнтр[0])*(self.Среда[j][0]-цeнтр[0])+(self.Среда[j][1]-цeнтр[1])*(self.Среда[j][1]-цeнтр[1]))>((цeнтр[0]-t[0])*(цeнтр[0]-t[0])+(цeнтр[1]-t[1])*(цeнтр[1]-t[1]))):
                self.Среда[j] = t            
            t = self.intersection(self.Трасса1[len(self.Трасса1)-1], self.Трасса1[0], (цeнтр[0],цeнтр[1]),
                                      (цeнтр[0] - self.Длинна * math.sin(math.radians(self.Угол-90 + j*360/self.Лучей/2)),
                                      цeнтр[1] - self.Длинна * math.cos(math.radians(self.Угол-90 + j*360/self.Лучей/2))))
            if t and (((self.Среда[j][0]-цeнтр[0])*(self.Среда[j][0]-цeнтр[0])+(self.Среда[j][1]-цeнтр[1])*(self.Среда[j][1]-цeнтр[1]))>((цeнтр[0]-t[0])*(цeнтр[0]-t[0])+(цeнтр[1]-t[1])*(цeнтр[1]-t[1]))):
                self.Среда[j] = t

    def Управление(self,Действие,Начало,Окно,Видить,Время):
        МаксСкВп = 6
        МаксСкНаз = -2
        Ускорение = 0.5
        Трение = 0
        Награда = 0
        t = []
        
        
        #if Действие==0 and self.Скорость < МаксСкВп:
        if self.Скорость < МаксСкВп:
            self.Скорость += Ускорение
        elif Действие==1 and self.Скорость > МаксСкНаз:
           self.Скорость-=Ускорение

        if abs(self.Скорость) > 0:  
            if self.Скорость > 0:
                self.Скорость-=Трение
            else:
                self.Скорость+=Трение

        if Действие==2:
            self.Угол+=1*abs(self.Скорость)*0.5+0.2
        if Действие==3:
            self.Угол-=1*abs(self.Скорость)*0.5+0.2      

        if self.Угол < 0:
            self.Угол += 360
        elif self.Угол > 360:
            self.Угол -= 360
     
        self.pos[0] -= self.Скорость*math.sin(math.radians(self.Угол))
        self.pos[1] -= self.Скорость*math.cos(math.radians(self.Угол)) 
        Закончил = self.Врезались()
        self.Лучи()
        if Видить:
            self.Графика(Окно,Время,Долго)

        #if int(self.Растояние)>self.Чекпоинт:
       #     self.Чекпоинт = int(self.Растояние)
        Награда = 1
        #else:
        #    Награда = -0.1

        Окружение = []
        #Окружение.append(self.Скорость)
        #Окружение.append(self.Угол)
        #Окружение.append(self.Растояние)
        #Окружение.append(self.Чекпоинт)
        ш, в = self.Картинка.get_size()
        цeнтр = (self.pos[0]+ш/2,self.pos[1]+в/2)
        for i in range(len(self.Среда)):
                t = (((self.Среда[i][0]-цeнтр[0])*(self.Среда[i][0]-цeнтр[0])+(self.Среда[i][1]-цeнтр[1])*(self.Среда[i][1]-цeнтр[1])))/1000
                if t > self.Длинна:
                    t = self.Длинна
                Окружение.append(t)

        Конец = Время.get_time() 
        self.Растояние+=self.Скорость*(Конец-Начало)/1000
        
        return Окружение, Награда, Закончил
                              
    def Графика(self, Окно,Время, Видно_Дорогу = True, Видно_Лилии = False, Видно_Круги = False, Видно_Дданные = False):
        ЦветФона = (50, 50, 50)
        ЦветЛиний = (255,255,255)
        ш, в = self.Картинка.get_size()
        цeнтр = (self.pos[0]+ш/2,self.pos[1]+в/2)

        pygame.font.init()
        Шрифт = pygame.font.SysFont('courier',24)

        Окно.fill(ЦветФона)

        if Видно_Дорогу:
            pygame.draw.aalines(Окно, ЦветЛиний, True, self.Трасса)
            pygame.draw.aalines(Окно, ЦветЛиний, True, self.Трасса1)

        if Видно_Лилии:
            for i in range(self.Лучей):
                pygame.draw.aaline(Окно, ЦветЛиний, (цeнтр[0],цeнтр[1]), (цeнтр[0] - self.Длинна * math.sin(math.radians(self.Угол-90 + i*360/self.Лучей/2)),цeнтр[1] - self.Длинна*math.cos(math.radians(self.Угол-90 + i*360/self.Лучей/2))))
        if Видно_Круги:
            for i in range(len(self.Среда)):
                pygame.draw.circle(Окно, ЦветЛиний, self.Среда[i], 5)

        if Видно_Дданные:             
            #t = int(self.Скорость)
            #Текст = Шрифт.render(f"Скорость = {t}",0,ЦветЛиний)
            #Окно.blit(Текст, (20, 20*2))
            #t = int(self.Угол)
            #Текст = Шрифт.render(f"Угол = {t}",0,ЦветЛиний)
            #Окно.blit(Текст, (20, 20*2))
            #t = int(self.Растояние)
            #Текст = Шрифт.render(f"Расстояние = {t}",0,ЦветЛиний)
            #Окно.blit(Текст, (20, 20*3))
            #t = (self.Чекпоинт)
            #Текст = Шрифт.render(f"Чекпоинт = {t}",0,ЦветЛиний)
            #Окно.blit(Текст, (20, 20*5))
            for i in range(len(self.Среда)):
                t = int((((self.Среда[i][0]-цeнтр[0])*(self.Среда[i][0]-цeнтр[0])+(self.Среда[i][1]-цeнтр[1])*(self.Среда[i][1]-цeнтр[1])))/1000)
                if t > self.Длинна:
                    t = self.Длинна
                Текст = Шрифт.render(f"{i+1} = {t}",0,ЦветЛиний)
                Окно.blit(Текст, (20, 20*i+20*2))
        с = int(pygame.time.get_ticks()/1000)
        м = int(с/60)
        ч = int(с/3600)
        if ч: м = м % (ч*60)
        if ч or м: с = с % (м*60+ч*3600)
        Текст = Шрифт.render(f"Пройденный путь: {self.Растояние:.2f} Время кадра: {Время.get_time()}мс Попыток: {self.Попыток} Прошло: {ч}ч {м}м {с}с", 0, ЦветЛиний)
        Окно.blit(Текст, (20, 20))

        Окно.blit(self.Поворот()[0],self.Поворот()[1])
        pygame.display.flip()
    
    def Запуск(self, Д = 1500,В = 800):
        #Создание окна
        Окно = pygame.display.set_mode((Д,В))

        Играем = 1
        while Играем:
            Начало = pygame.time.get_ticks()
            t = [0,0,0,0]
            #Пробек по всем сабытиям
            for event in pygame.event.get():
                #Нажата кнопка выход
                if event.type == pygame.QUIT:
                    Играем = 0
                keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                t[2]=1
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                t[3]=1
            if (keys[pygame.K_DOWN] or keys[pygame.K_s]):
                t[1]=1
            if (keys[pygame.K_UP] or keys[pygame.K_w]):
                t[0]=1
            self.Управление(t)
            self.pos[0] -= self.Скорость*math.sin(math.radians(self.Угол))
            self.pos[1] -= self.Скорость*math.cos(math.radians(self.Угол))
           
            self.Лучи()
            if self.Врезались():
                print('Проиграли')
                self.Обновить()
                self.Попыток+=1
            else:
                self.Графика(Окно,Начало)

            Конец = pygame.time.get_ticks()            
            pygame.time.delay(30-Конец+Начало)

            if Играем == 0:
                print('Закрыли')
                pygame.quit()

class QNetwork(nn.Module):
    def __init__(self,вход,выход):
        super().__init__()

        self.сл1 = nn.Linear(вход,256)        
        self.сл10 = nn.Linear(256,выход)

    def forward(self, x):
        x = F.relu(self.сл1(x))
        x = self.сл10(x)
        return x

    def семплируем_действие(self, obs, Eps):
        вход = self.forward(obs)
        знаение = random.random()
        if знаение < Eps:
             return random.randint(2, 3)
        else:
            return вход.argmax().item() + 2

class ReplayBuffer():
    def __init__(self, max_size):
        self.Структура = []
        self.Макс_Размер=max_size

    def put(self, transition):
        if(self.Макс_Размер > len(self.Структура)):
            self.Структура.append(transition)
        else:
            t = random.randint(1, len(self.Структура)-1)
            self.Структура.pop(t)
            self.Структура.insert(t, transition)


    def sample(self, n):

        Состояние, Действие, Награда, Новое_Состояние = [],[],[],[]
        
        if(len(self.Структура)>n):
            t = random.sample(self.Структура,n)
        else:
            t = self.Структура

        for i in range(len(t)):
            Состояние.append(t[i][0])
            Действие.append([t[i][1]])
            Награда.append([t[i][2]])
            Новое_Состояние.append(t[i][3])

        return torch.tensor(Состояние, dtype = torch.float), torch.tensor(Действие), \
            torch.tensor(Награда), torch.tensor(Новое_Состояние,dtype = torch.float)

    def __len__(self):
        return len(self.Структура)

def Тренеровка (Сеть, ТаргетСеть, Буфер, Оптимизатьр, РазмерБуфера, гамма, Колличество_обновлений = 10):
   
    for i in range(Колличество_обновлений):
        #Сэмплируем мини бач из Буфера
        Состояние, Действие, Награда, Новое_состояние = Буфер.sample(РазмерБуфера)

        # Получаем полезность для выбраного действия q сети
        q_out = Сеть(Состояние)
        q_a = q_out.gather(1,Действие)
        
        #Получаем значения max_q Нагрвду сети и считаем значение Награды
        max_q_prime = ТаргетСеть(Новое_состояние).max(1)[0].unsqueeze(1)
        target = Награда + гамма*max_q_prime
                
        #Определяем Loss функцию для q
        loss = F.smooth_l1_loss(q_a, target.detach())

        Оптимизатьр.zero_grad()
        loss.backward()
        Оптимизатьр.step()

def run(learning_rate, гамма, размер_буфера, размер_бача, интервал_обновления,
стартовый_размер_буфера, печать_интервалов = 20, кол_эпизодоа = 1001):
    #Создаём окружение
    игра = gym.make('CartPole-v1')

    #Сзздаём сети
    Сеть = QNetwork()
    ТаргетСеть = QNetwork()

    #Копируем веса сети в таргет сеть
    ТаргетСеть.load_state_dict(Сеть.state_dict())

    Буфер = ReplayBuffer(max_size = размер_буфера)

    ОбщаяНаграда = 0.0

    #Инициализируем щптимизатор, полученным Lr
    Оптимизатор = optim.Adam(Сеть.parameters(), lr = learning_rate)

    for Эпизод in range(кол_эпизодоа):

        #Постепенно изменяем Eps с 8% до 1%
        Eps = max(0.01, 0.08 - 0.01*(Эпизод/200))

        Окружение = игра.reset()

        #Выполняем 600 шагов в оеружении и сохраняем полученные данные
        for t in range(600):

          #Получаем действие используя сеть
          Действие = Сеть.семплируем_действие(torch.from_numpy(Окружение).float(),Eps)

          #Выполняем действие в окружении
          НовоеОкружение, Награда, Закончил, Инфо = игра.step(Действие)
          
          #Добовляем данные в буфер
          done_mask = 0.0 if Закончил else 1.0

          #Сжимаем вознограждение и добавляем в буфер
          Буфер.put((Окружение,Действие,Награда/100.0, НовоеОкружение,done_mask))

          Окружение = НовоеОкружение

          ОбщаяНаграда+=Награда

          if Закончил:
              break

        if len(Буфер) > стартовый_размер_буфера:
            Тренеровка(Сеть, ТаргетСеть, Буфер, Оптимизатор, размер_бача, гамма)

        if Эпизод % интервал_обновления == 0 and Эпизод!=0:
            ТаргетСеть.load_state_dict(Сеть.state_dict())

        if Эпизод % печать_интервалов == 0 and Эпизод!=0:
          f = len(Буфер) 
          print("# Эпизод: {}, Награда: {:.1f}, размер буфера: {}, Eps: {:.1f}%".format(
               Эпизод,ОбщаяНаграда/печать_интервалов,len(Буфер),Eps*100))
          ОбщаяНаграда = 0.0
        
        #Отрисуем игру
        if Эпизод % 100 == 0:
            Окружение = игра.reset()
            for t in range(600):

              #Получаем действие используя сеть
              Действие = Сеть.семплируем_действие(torch.from_numpy(Окружение).float(),Eps)

              #Выполняем действие в окружении
              НовоеОкружение, Награда, Закончил, Инфо = игра.step(Действие)

              #Добовляем данные в буфер
              done_mask = 0.0 if Закончил else 1.0

              #Сжимаем вознограждение и добавляем в буфер
              Буфер.put((Окружение,Действие,Награда/100.0, НовоеОкружение,done_mask))

              Окружение = НовоеОкружение

              ОбщаяНаграда+=Награда
              игра.render()
              if Закончил:
                  break
    игра.close()

"""
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

learning_rate = 0.005
гамма = 0.98
размер_буфера = 80000
размер_бача = 100
интервал_обновления=10
стартовый_размер_буфера = 500
печать_интервалов = 20
кол_эпизодоа = 100001
Д = 1500
В = 800
Окно = pygame.display.set_mode((Д,В))
Время = pygame.time.Clock()

#Создаём окружение
игра = Игра()
#Сзздаём сети
Сеть = QNetwork(игра.Лучей,2)
ТаргетСеть = QNetwork(игра.Лучей,2)

#Копируем веса сети в таргет сеть
ТаргетСеть.load_state_dict(Сеть.state_dict())
Буфер = ReplayBuffer(max_size = размер_буфера)
Буфер2 = ReplayBuffer(max_size = размер_буфера)
#Инициализируем оптимизатор, полученным Lr
Оптимизатор = optim.Adam(Сеть.parameters(), lr = learning_rate)

ОбщаяНаграда = 0.0
Видить = True
Рекорд = -100000
Закрыли = False
Эпизод = -1
Действие = 0
сменаEps = 0
while True:
    Эпизод +=1

    #Постепенно изменяем Eps с 8% до 1%
    if not сменаEps:
        Eps = max(0.01, 0.12 - 0.00022*(Эпизод))

    Окружение = игра.Обновить()
    
    Играем = 1
    Долго = 0
    Буфер2.Структура = []
    if not Закрыли:
        #Выполняем 600 шагов в окружении и сохраняем полученные данные
        while Играем:

            Начало = 0
            #Получаем действие используя сеть
            keys = pygame.key.get_pressed()

            if not (keys[pygame.K_LEFT] or keys[pygame.K_a] or keys[pygame.K_RIGHT] or keys[pygame.K_d] or keys[pygame.K_DOWN] or keys[pygame.K_s] or keys[pygame.K_UP] or keys[pygame.K_w]):
                Действие = Сеть.семплируем_действие(torch.from_numpy(np.array(Окружение)).float(),Eps)
            
            if (keys[pygame.K_UP] or keys[pygame.K_w]):
                Действие=0
            if (keys[pygame.K_DOWN] or keys[pygame.K_s]):
                Действие=1
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                Действие=2
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                Действие=3

            #Выполняем действие в окружении
            НовоеОкружение, Награда, Закончил = игра.Управление(Действие,Начало,Окно,Видить,Время)

            if (игра.Растояние > Рекорд):
                Рекорд = игра.Растояние

            if Закончил:
                игра.Обновить()
                игра.Попыток+=1
                Награда = -300
                Играем = 0                
           
            #Добовляем данные в буфер
       
            Конец = Время.tick() 
            Долго += Конец
            if Видить:
                if Конец < 30:
                    pygame.time.delay(30-Конец+Начало)

           # if Долго > 50000:
           #     игра.Обновить()
            #    игра.Попыток+=1
            #    Играем = 0
            #Пробек по всем сабытиям
            for event in pygame.event.get():
                keys = pygame.key.get_pressed()
                #Нажата кнопка выход
                if event.type == pygame.QUIT or keys[pygame.K_ESCAPE]:
                    Играем = 0
                    Закрыли = True
                if keys[pygame.K_0]:
                    Видить = False
                if keys[pygame.K_9]:
                    Видить = True
            #Сжимаем вознограждение и добавляем в буфер
            if not (keys[pygame.K_DOWN] or keys[pygame.K_s] or keys[pygame.K_UP] or keys[pygame.K_w]):
                Буфер.put((Окружение,Действие-2,Награда, НовоеОкружение))
                Буфер2.put((Окружение,Действие-2,Награда, НовоеОкружение))
            if Награда < 0:
                for i in range(50):
                    Буфер.put((Окружение,Действие-2,Награда, НовоеОкружение))
            
            Окружение = НовоеОкружение

            ОбщаяНаграда+=Награда

    else:
        print("Пауза")
        print("Чтобы продолжить введите 1")
        print("Сохранить данные 3")
        print("Загрузить данные 4")
        print("Провести обучение 5")
        print("Поменять Eps 6")
        a = int(input())
        if a == 1:
            Закрыли = False
            Окно = pygame.display.set_mode((Д,В))
            Эпизод -=1
        if a == 2:
            Закрыли = False
            Окно = pygame.display.set_mode((Д,В))
        if a == 3:
            Эпизод -=1
            Закрыли = False
            torch.save(Сеть, 'Сеть.txt')
            torch.save(ТаргетСеть, 'ТаргетСеть.txt')
            with open('Буфер.txt', 'wb') as fp:
                pickle.dump(Буфер.Структура, fp)
            Окно = pygame.display.set_mode((Д,В))
        if a == 4:
            Эпизод =0
            Закрыли = False
            Сеть = torch.load('Сеть.txt')
            ТаргетСеть = torch.load('ТаргетСеть.txt')
            with open ('Буфер.txt', 'rb') as fp:
                Буфер.Структура = pickle.load(fp)
            Окно = pygame.display.set_mode((Д,В))
        if a == 5:
            print("Введите размер бача")
            размер_бача1 = int(input())
            print("Введите Колличество обновлений")
            Колличество_обновлений1 = int(input())
            Эпизод =0
            Закрыли = False
            Тренеровка (Сеть, ТаргетСеть, Буфер, Оптимизатор, размер_бача1, гамма, Колличество_обновлений1)
            Окно = pygame.display.set_mode((Д,В))
        if a == 6:
            print("Введите Eps")
            Eps = float(input())
            сменаEps = 1
    пп = len(Буфер2)
    if len(Буфер2)!=0:
        #кол_обн = min(500,max(10,int((len(Буфер)/len(Буфер2))/50)))
        if len(Буфер)>стартовый_размер_буфера:
            Тренеровка (Сеть, ТаргетСеть, Буфер, Оптимизатор, len(Буфер2), гамма)
        #if len(Буфер)>50000:
         #   Тренеровка (Сеть, ТаргетСеть, Буфер2, Оптимизатор, размер_бача, гамма)
    
    if Эпизод % интервал_обновления == 0 and Эпизод!=0:
        ТаргетСеть.load_state_dict(Сеть.state_dict())

    if Эпизод % печать_интервалов == 0 and Эпизод!=0:
        f = len(Буфер) 
        print("# Эпизод: {}, Награда: {:.1f}, размер буфера: {}, Eps: {:.2f}%, Рекорд: {:.2f}".format(
             Эпизод,ОбщаяНаграда/печать_интервалов,len(Буфер),Eps*100,Рекорд))
        ОбщаяНаграда = 0.0
        Рекорд = 0
