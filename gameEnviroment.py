import pygame
import neat
import time
import os
import random
import numpy as np
from gym import spaces

# import gym
pygame.font.init()

# z wielkich liter bo stałe, wielkość wyświetlanego obrazu
WIN_WIDTH = 500
WIN_HIGHT = 800

BIRD_IMGS = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird1.png"))),
             pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird2.png"))), pygame.transform.scale2x(
        pygame.image.load(os.path.join("imgs",
                                       "bird3.png")))]  # tu będą przechowywane obrzy ptaków, funkcja scale powiększa x2 wielkość ptaków przy pozostawieniu grafiki
PIPE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "pipe.png")))
BASE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "base.png")))
BG_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bg.png")))
STAT_FONT = pygame.font.SysFont('comicsans', 50)  # Czcionka i rozmiar napisu


class Bird():
    IMGS = BIRD_IMGS
    MAX_ROTATION = 25  # Obraca się do chmur i do ziemi
    ROT_VEL = 20  # Ile będziemy rotować przy każdej klatce
    ANIMATION_TIME = 5  # Ile czasu będzie trwała każda animacja ptaka

    def __init__(self, x, y):  # Pozycja startowa naszego ptaka
        self.x = x
        self.y = y
        self.tilt = 0  # to czy patzy w górę czy w dół
        self.tick_count = 0  # Jak dużo się poruszyliśmy od ostatniego skoku
        self.vel = 0  # Prędkość
        self.hight = self.y
        self.img_count = 0  # Wybór którego z ptaków chcemy użyć
        self.img = self.IMGS[0]  # Pierwszy ptak

    def jump(self):
        self.vel = -10.5  # ujemne dla tego że na dół jest dodatnie, do góry ujemne, tutaj prędkość poruszania w górę
        self.tick_count = 0  # Sprawdza kiedy ostatnio skakaliśmy
        self.hight = self.y  # Sprawdza skąd skakał ptak, z kąd wykonał ruch

    def move(self):  # Każda klatka z którą porusza się ptak
        self.tick_count += 1  # Jak dużo się poruszliśmy od ostatniego skoku

        d = self.vel * self.tick_count + 1.5 * self.tick_count ** 2  # sprawia że lecimy coraz szybciej w dół, po skoku się thick_count się zeruje

        if d >= 16:  # Graniczna wartość, jeżeli poruszamy się szybciej niż 16 w dół to ustawiamy na 16
            d = 16

        if d < 0:  # Jeżeli poruszamy się do góry, poruszmy się bardziej do góry
            d -= 2

        self.y = self.y + d  # Poruszanie się w górę i w dół

        if d < 0 or self.y < self.hight + 50:  # Sprawdza czy ptak wykonuje skok, leci w górę, po ostatiej pozycji lub prędkości y
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        else:
            if self.tilt > -90:  # Obraca w dół
                self.tilt -= self.ROT_VEL

    def draw(self, win):  # win to okno w którym jest gra
        self.img_count += 1

        # Animacja ruszania skrzydłami w zależności od czasu
        if self.img_count < self.ANIMATION_TIME:  # Jeżeli poniżej 5 jednostek czasu to obrazek bird1
            self.img = self.IMGS[0]
        elif self.img_count < self.ANIMATION_TIME * 2:
            self.img = self.IMGS[1]
        elif self.img_count < self.ANIMATION_TIME * 3:
            self.img = self.IMGS[2]
        elif self.img_count < self.ANIMATION_TIME * 4:
            self.img = self.IMGS[1]
        elif self.img_count == self.ANIMATION_TIME * 4 + 1:
            self.img = self.IMGS[0]
            self.img_count = 0

        if self.tilt <= -80:  # Kiedy skczaemy to żeby nie pominąć klatki, zmiana obrazka
            self.img = self.IMGS[1]
            self.img_count = self.ANIMATION_TIME * 2

        rotated_image = pygame.transform.rotate(self.img, self.tilt)  # ze stack overflowa, obraca obrazek na środku
        new_rect = rotated_image.get_rect(center=self.img.get_rect(topleft=(self.x, self.y)).center)
        win.blit(rotated_image, new_rect.topleft)

    def get_mask(self):  # Do kolizji obiektów
        return pygame.mask.from_surface(self.img)


class Pipe:
    GAP = 200  # odlegość między górnią a donlną rurą
    VEL = 5  # Prędkość z jaką porusza się rura s tronę ptaka

    def __init__(self, x):
        self.x = x
        self.height = 0
        self.gap = 100

        self.top = 0  # Sprawdzamy w którym miejscu znajduje się górna rura
        self.bottom = 0  # Sprawdzamy w którym miejscu znajduje się dolna rura
        self.PIPE_TOP = pygame.transform.flip(PIPE_IMG, False, True)  # obracamy rurę, żeby wisiała o góry
        self.PIPE_BOTTOM = PIPE_IMG  # Tą po prostu ładujemy

        self.passed = False  # Czy udało się ominąć rurę
        self.set_height()  # Dzięki temu będziemy rozmieszczać rury, ustalać ich przerwę wyosokość itp

    def set_height(self):
        self.height = random.randrange(50, 450)  # przedział odległość od krawędzi w którym będzie umieszona rura
        self.top = self.height - self.PIPE_TOP.get_height()  # Musimy przesunąć do góry więc wartość ujemna
        self.bottom = self.height + self.GAP  # Rura dolna + przerwa
        # Przedział self.top -> od -590 do -190
        # Przedział self.bottom -> 250 do 650

    def move(self):
        self.x -= self.VEL  # Przesuwamy rurę do lewej klatka po klatce

    def draw(self, win):
        win.blit(self.PIPE_TOP, (self.x, self.top))  # self.x, self.top pokazują gdzie na ekranie ma się to rysować
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    def collide(self, bird):
        bird_mask = bird.get_mask()  # Pobieramy maskę
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)  # Tworzymy maski obiektów

        top_offset = (self.x - bird.x, self.top - round(bird.y))  # przsuwa się pozycja względem osi x i y górnej rury
        bottom_offset = (
        self.x - bird.x, self.bottom - round(bird.y))  # przsuwa się pozycja względem osi x i y dolnej rury

        b_poin = bird_mask.overlap(bottom_mask, bottom_offset)  # Pokazuje punkt kolizji, jeżeli nie ma to zwraca NONE
        t_poin = bird_mask.overlap(top_mask, top_offset)

        if t_poin or b_poin:
            return True

        return False


class Base:
    VEL = 5  # Musi być takie samo jak prędkoć rury, żeby poruszały się w tym samym czasie
    WIDTH = BASE_IMG.get_width()
    IMG = BASE_IMG

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):  # Ta funkcja służy przesuwaniu się w nieskończoność ziemi
        self.x1 -= self.VEL
        self.x2 -= self.VEL

        if self.x1 + self.WIDTH < 0:  # jeżeli przejdzie całą szerokość, wskakuje na tył x.2
            self.x1 = self.x2 + self.WIDTH

        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self, win):
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))


def draw_window(win, bird, pipes, base, score):
    win.blit(BG_IMG, (0, 0))  # blit rysuje cokolwiek się poda 0,0 to top left position Lewy górny
    for pipe in pipes:
        pipe.draw(win)

    text = STAT_FONT.render('Score: ' + str(score), 1, (255, 255, 255))
    win.blit(text, (WIN_WIDTH - 10 - text.get_width(), 10))  # Jak będzię rosnąć text to się będzie przesuwał

    base.draw(win)

    bird.draw(win)
    pygame.display.update()  # apdejtuje ekran


class Enviroment():

    def __init__(self):
        self.bird = Bird(230, 350)
        self.base = Base(730)
        self.pipes = [Pipe(600)]
        self.win = pygame.display.set_mode((WIN_WIDTH, WIN_HIGHT))  # przypisuje wartości okna
        self.clock = pygame.time.Clock()  # tworzy możliwość klatek na sekundę
        self.liveReward = 0.1
        self.negReward = -10.
        self.posReward = 2.
        self.score = 0
        self.reward = 0
        self.gameOver = False
        self.time_elapsed_since_last_action = 0
        self.global_time = 0

        # Przedział ptaka -> od 0 do 730
        # Przedział self.top -> od -590 do -190
        # Przedział self.bottom -> 250 do 650
        self.low = np.array([0])
        self.high = np.array([730])

        self.action_space = spaces.Discrete(2)
        # self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        # Action 0 nothing, Action 1 jump
        gameOver = False
        reward = self.liveReward

        #self.clock.tick(30)  # 30 klatek na sekundę
        for event in pygame.event.get():  # Śledzi czy użytkownik wykonał jakąś akcję
            if event.type == pygame.QUIT:
                run = False  # Jeżeli wciśniemy czerwony krzyżyk to gra się zamknie
                pygame.quit()
                quit()

        pipe_ind = 0
        if len(self.pipes) > 1 and self.bird.x > self.pipes[0].x + self.pipes[0].PIPE_TOP.get_width():
            pipe_ind = 1

        topPos = abs(self.bird.y - self.pipes[pipe_ind].height)
        botPos = abs(self.bird.y - self.pipes[pipe_ind].bottom)
        #self.state = (self.bird.y, abs(self.bird.y - self.pipes[pipe_ind].height), abs(self.bird.y - self.pipes[pipe_ind].bottom))



        self.bird.move()  # ptak się porusza w osi pionowej z każdą klatką
        add_pipe = False
        passed = False
        rem = []  # Lista usuniętych rur
        for pipe in self.pipes:
            if pipe.collide(self.bird):  # Jeżeli występuje kolizja z ptakiem to
                gameOver = True
                reward = self.negReward
            if pipe.x + pipe.PIPE_TOP.get_width() < 0:  # Sprawdza czy jakaś rura jest całkowice poza ekranem, bo przeszła przez całość
                rem.append(pipe)

            if not pipe.passed and pipe.x < self.bird.x:  # Jeżeli pozycja ptaka jest za pozycją rury, to wtedy passed = true
                pipe.passed = True
                add_pipe = True

            pipe.move()

        if add_pipe:  # Dodaje rurę i punkt
            self.score += 1
            passed = True
            reward = self.posReward
            self.pipes.append(Pipe(600))

        for r in rem:  # Usuwa rury które są w liście poza ekranem
            self.pipes.remove(r)

        if self.bird.y + self.bird.img.get_height() >= 730 or self.bird.y < 0:  # Jeżeli ptak uderzy w ziemię to koniec gry
            gameOver = True
            reward = self.negReward

        self.base.move()
        reward = self.liveReward
        #if gameOver == True:
        #    reward = self.negReward
        #elif gameOver == False and passed == True:
         #   reward = self.posReward
       # else:
        #    reward = self.liveReward

        if action == 1:
            self.bird.jump()
            if gameOver == True:
                reward = self.negReward
            elif gameOver == False and passed == True:
                reward = self.posReward
            self.state = (
            self.bird.y, abs(self.bird.y - self.pipes[pipe_ind].height), abs(self.bird.y - self.pipes[pipe_ind].bottom))

        elif action ==0:
            if gameOver == True:
                reward = self.negReward
            elif gameOver == False and passed == True:
                reward = self.posReward
            self.state = (
            self.bird.y, abs(self.bird.y - self.pipes[pipe_ind].height), abs(self.bird.y - self.pipes[pipe_ind].bottom))
            pass

        draw_window(self.win, self.bird, self.pipes, self.base, self.score)  # Tworzy okno
        return np.array(self.state), reward, gameOver, {}

    def reset(self):
        self.bird = Bird(230, 350)
        self.base = Base(730)
        self.pipes = [Pipe(600)]
        self.score = 0
        self.clock = pygame.time.Clock()  # tworzy możliwość klatek na sekundę
        self.liveReward = 0.1
        self.negReward = -1.
        self.posReward = 2.
        self.score = 0
        self.reward = 0
        self.gameOver = False
        self.time_elapsed_since_last_action = 0
        self.global_time = 0

        self.state, reward, gameOver, _ = self.step(0)

        return np.array(
            self.state)  # obs to qTable zawierające akcje oraz stany, czyli bird.y, top pipe oraz bottom pipe

    def render(self):
        draw_window(self.win, self.bird, self.pipes, self.base, self.score)  # Tworzy okno


'''
def main():
    bird = Bird(230,350)                                 #200,200 to pozycja startowa
    #birds = []         # Wiele ptaków
    base = Base(730)
    pipes= [Pipe(600)]
    win = pygame.display.set_mode((WIN_WIDTH,WIN_HIGHT))   #przypisuje wartości okna
    clock = pygame.time.Clock()                           #tworzy możliwość klatek na sekundę
    score=0

    run = True
    while run:
        clock.tick(30)                         # 30 klatek na sekundę
        for event in pygame.event.get():       #Śledzi czy użytkownik wykonał jakąś akcję
            if event.type == pygame.QUIT:
                run = False                    #Jeżeli wciśniemy czerwony krzyżyk to gra się zamknie

        pipe_ind=0
        if len(pipes)>1 and bird.x > pipe.x + pipe.PIPE_TOP.get_width():
            pipe_ind=1

        #bird.move()                            #ptak się porusza w osi pionowej z każdą klatką
        add_pipe=False
        rem=[]                                 #Lista usuniętych rur
        for pipe in pipes:
            if pipe.collide(bird):             #Jeżeli występuje kolizja z ptakiem to
                pass
            if pipe.x + pipe.PIPE_TOP.get_width() < 0:    #Sprawdza czy jakaś rura jest całkowice poza ekranem, bo przeszła przez całość
                rem.append(pipe)

            if not pipe.passed and pipe.x < bird.x:       #Jeżeli pozycja ptaka jest za pozycją rury, to wtedy passed = true
                pipe.passed = True
                add_pipe = True

            pipe.move()

        if add_pipe:                                     #Dodaje rurę i punkt
            score += 1
            pipes.append(Pipe(600))

        for r in rem:                                     #Usuwa rury które są w liście poza ekranem
            pipes.remove(r)

        if bird.y +bird.img.get_height() >=730 or bird.y < 0:            #Jeżeli ptak uderzy w ziemię to koniec gry
            pass

        base.move()



        print(bird.y, abs(bird.y-pipes[pipe_ind].height), abs(bird.y-pipes[pipe_ind].bottom))

        draw_window(win,bird,pipes,base,score)                  #Tworzy okno




    pygame.quit()

main()
'''
