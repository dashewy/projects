import pygame
from pygame.locals import *
import random

# intialize instance of pygame
pygame.init()

# adding for framerate
clock = pygame.time.Clock()
fps = 60

# setting dim for game window
screen_width = 864
screen_height = 936
# set for event handling to start game
flying = False
death = False
# dist btwn pipes
pipe_gap = 200
# how often pipes come
pipe_freq = 1500 # mls
last_pipe = pygame.time.get_ticks() - pipe_freq

score = 0
pass_pipe = False

screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('flapping sphere')

# load images
bg = pygame.image.load('/Users/alex/MyProject/flappy_stuff/flappy_background.png')
ground = pygame.image.load('/Users/alex/MyProject/flappy_stuff/flappy_ground.png')
restart_butt = pygame.image.load('/Users/alex/MyProject/flappy_stuff/flappy_restart.png')

ground_scroll = 0
# set pixel_speed
scroll_speed = 3

# font, size
font = pygame.font.SysFont('Bauhaus 93', 60)
white = (255, 255, 255)

# draws score image
def draw_text(text, font, text_col, x, y):
    img = font.render(text, True, text_col)
    screen.blit(img, (x, y))

def reset():
    # clears the pipe group
    pipe_group.empty()
    flappy.rect.x = 100
    flappy.rect.y = screen_height // 2
    score = 0
    return score 

# using pygames sprite class
class Bird(pygame.sprite.Sprite):
    def __init__(self, x, y):
        # inherant from pygames sprite class
        pygame.sprite.Sprite.__init__(self)
        # self.org_image = pygame.image.load('/Users/alex/MyProject/flappy_stuff/flappy_sphere.png')
        # self.image = self.org_image
        # use below to animate sprite
        self.images = []
        self.index = 0
        self.counter = 0
        for num in range(1, 4):
            img = pygame.image.load(f'/Users/alex/MyProject/flappy_stuff/flappy_sphere{num}.png')
            self.images.append(img)
        self.image = self.images[self.index]
        self.rect = self.image.get_rect()
        self.rect.center = [x, y]
        self.vel = 0
        self.press = False
        
    # handle animation
    def update(self):
        if flying:
            # gravity
            self.vel += 0.4
            if self.vel > 9.8:
                self.vel = 9.8
            # def crashing into bottom
            if self.rect.bottom < 761:
                self.rect.y += int(self.vel)
                
        if not death:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_SPACE] and self.press == False:
                self.press = True
                self.vel = -10
            # keeps you from flying away by holding keys
            if not keys[pygame.K_SPACE]:
                self.press = False
            
            self.counter += 1
            flap_cooldown = 5
        
            if self.counter > flap_cooldown:
                self.counter = 0
                self.index += 1
                if self.index >= len(self.images):
                    self.index = 0
            self.image = self.images[self.index]
        
            # rotate bird (rotate is counter clockwise + -> up, - -> down)
            self.image = pygame.transform.rotate(self.images[self.index], -self.vel * 2)
            # use self.images[self.index], self.vel
            # recenter the rect around the sprite
            self.rect = self.image.get_rect(center=self.rect.center)
        else:
            self.image = pygame.transform.rotate(self.images[self.index], -90)
            
class Pipe(pygame.sprite.Sprite):
    def __init__(self, x, y, position):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load('/Users/alex/MyProject/flappy_stuff/flappy_pipe.png')
        self.rect = self.image.get_rect()
        # postion 1: TOP -1: BOTTOM
        if position == 1:
            # image, x, y flipping rotation
            self.image = pygame.transform.flip(self.image, False, True)
            self.rect = self.image.get_rect()
            self.rect.bottomleft = [x, y - pipe_gap // 2]
        if position == - 1:
            self.rect.topleft = [x, y + pipe_gap // 2]
        
    def update(self):
        self.rect.x -= scroll_speed
        if self.rect.right < 0:
            # deletes pipe after it goes off screen
            self.kill()
        

class Button():
    def __init__(self, x, y, img):
        self.image = img
        self.rect = self.image.get_rect()
        self.rect.topleft = (x, y)
        
    def draw(self):
        action = False
        # mouse pos (returns list)
        pos = pygame.mouse.get_pos()
        # check if mouse over botton
        if self.rect.collidepoint(pos):
            # return list
            if pygame.mouse.get_pressed()[0] == 1:
                action = True

        # draw button
        screen.blit(self.image, (self.rect.x, self.rect.y))
        
        return action 
# can add for animating sprite with flapping wings etc. and using .draw func
bird_group = pygame.sprite.Group()
pipe_group = pygame.sprite.Group()
# can add below to bird group
flappy = Bird(100, screen_height // 2)
bird_group.add(flappy)
# can add more images as list to flip through to animate
# restart button
button = Button(screen_width // 2 - 180, screen_height // 2, restart_butt)


run = True
while run:
    clock.tick(fps)
    
    # blit puts images on screen, 0,0 are coordinates where (TOP LEFT CORENER)
    screen.blit(bg, (0,0))
    # PUT HERE TO LAYER UNDER GROUND
    pipe_group.draw(screen)
    
    # draw and scroll ground leave at gs ,0 which is how illastrator file is set up
    screen.blit(ground, (ground_scroll, 766))
    # looping for every event in pygame
    
    # check the score check if there are pipes
    if pipe_group:
        if bird_group.sprites()[0].rect.left > pipe_group.sprites()[0].rect.left \
            and bird_group.sprites()[0].rect.right < pipe_group.sprites()[0].rect.right\
                and not pass_pipe:
                    pass_pipe = True
                    
        if pass_pipe:
            if bird_group.sprites()[0].rect.left > pipe_group.sprites()[0].rect.right:
                score += 1
                pass_pipe = False
    # adds score on screen
    draw_text(str(score), font, white, screen_width // 2, 20)
    
    # check for game over and reset
    if death:
        if button.draw():
            death = False
            score = reset()
    for event in pygame.event.get():
        # QUIT terminates when hitting x button on screen
        if event.type == pygame.QUIT:
            run = False
        if event.type == pygame.KEYDOWN and not flying and not death:
            flying = True
    # puts bird on screen
    bird_group.draw(screen)
    # call update funcion to move flappy
    bird_group.update()
    # look for collision group, group, kill, kill
    if pygame.sprite.groupcollide(bird_group, pipe_group, False, False) or flappy.rect.top < 0:
        death = True
    
    # check if bird died
    if flappy.rect.bottom >= 761:
        death = True
        flying = False
        
    if not death and flying:
        # generate new pipes
        time_now = pygame.time.get_ticks()
        if time_now - last_pipe > pipe_freq:
            pipe_height = random.randint(-100, 100)
            btm_pipe = Pipe(screen_width, (screen_height // 2) + pipe_height, -1)
            top_pipe = Pipe(screen_width, (screen_height // 2) + pipe_height, 1)
            pipe_group.add(btm_pipe)
            pipe_group.add(top_pipe)   
            last_pipe = time_now 
        
        # decrement ground scroll by the speed to adjust position on screen
        ground_scroll -= scroll_speed
        # resetting ground lines (set to the amount of over hang)
        if abs(ground_scroll) > 35:
            ground_scroll = 0
        # move pipes
        pipe_group.update()  
    # need this to update display backgroud
    pygame.display.update()
            
pygame.quit()