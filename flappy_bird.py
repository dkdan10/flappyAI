import pygame
import time
import os
import random
import neat

pygame.font.init()

WIN_WIDTH = 500
WIN_HEIGHT = 600
GEN = -1

BIRD_IMGS = [
    pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird1.png"))),
    pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird2.png"))),
    pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird3.png")))
]

PIPE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "pipe.png")))
BASE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "base.png")))
BG_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bg.png")))

STAT_FONT = pygame.font.SysFont("comicsans", 50)

################################### BIRD CLASS ###################################
class Bird:
    IMGS = BIRD_IMGS
    MAX_ROTATION = 25 #How much bird will tilt
    ROT_VEL = 20 #How much we will rotate each frame
    ANIMATION_TIME = 3 #How fast or slow bird flaps its wings in frame

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0 #how many times we have moved since the last jump
        self.vel = 0
        self.height = self.y #height of last jump
        self.img_count = 0
        self.img = self.IMGS[0]

    def jump(self):
        self.vel = -10.5 #negative velocity points the bird upwards
        self.tick_count = 0
        self.height = self.y

    def move(self):
        self.tick_count += 1
        # Based on velocity tick count find the displacement of bird pos
        d = self.vel * self.tick_count + 1.5 * self.tick_count**2
        # Terminal velocity...
        if d >= 16: 
            d = 16
        
        # Fine tunes jump to make it smooth. 
        if d < 0:
            d-=2

        self.y = self.y + d
        
        # find tilt
        if d < 0 or self.y < self.height:
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        else:
            if self.tilt > -90:
                self.tilt -= self.ROT_VEL

    def draw(self, window):
        self.img_count += 1

        if self.img_count < self.ANIMATION_TIME:
            self.img = self.IMGS[0]
        elif self.img_count < self.ANIMATION_TIME*2:
            self.img = self.IMGS[1]
        elif self.img_count < self.ANIMATION_TIME*3:
            self.img = self.IMGS[2]
        elif self.img_count < self.ANIMATION_TIME*4:
            self.img = self.IMGS[1]
        elif self.img_count < self.ANIMATION_TIME*4 + 1:
            self.img = self.IMGS[0]
            self.img_count = 0

        if self.tilt <= -80:
            self.img = self.IMGS[1]
            self.img_count = self.ANIMATION_TIME*2

        #Rotate image around left top corner
        rotated_image = pygame.transform.rotate(self.img, self.tilt)
        #Fix rotation to be around center image
        new_rect = rotated_image.get_rect(center=self.img.get_rect(topleft=(self.x, self.y)).center)
        window.blit(rotated_image, new_rect.topleft)

    def get_mask(self):
        return pygame.mask.from_surface(self.img)

################################### PIPE CLASS ###################################
class Pipe:
    GAP = 200
    VEL = 5

    def __init__(self, x):
        self.x = x
        self.height = 0

        self.top = 0
        self.bottom = 0
        self.PIPE_TOP = pygame.transform.flip(PIPE_IMG, False, True)
        self.PIPE_BOTTOM = PIPE_IMG

        self.passed = False
        self.set_height()

    def set_height(self):
        self.height = random.randrange(50,350)
        # find position to draw top Pipe
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP

    def move(self):
        self.x -= self.VEL

    def draw(self, window):
        window.blit(self.PIPE_TOP, (self.x, self.top))
        window.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    def collide(self, bird):
        # Use masks to get pixel perfect collision detection. 
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)

        top_offset = (self.x - bird.x, int(self.top - round(bird.y)))
        bottom_offset = (self.x - bird.x, int(self.bottom - round(bird.y)))

        bottom_point = bird_mask.overlap(bottom_mask, bottom_offset)
        top_point = bird_mask.overlap(top_mask, top_offset)

        if top_point or bottom_point:
            return True
        return False

################################### BASE CLASS ###################################
class Base:
    VEL = 5
    WIDTH = BASE_IMG.get_width()
    IMG = BASE_IMG

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        #Moving two background images at once so it looks like continuous movement.
        self.x1 -= self.VEL
        self.x2 -= self.VEL

        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH

        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH
    
    def draw(self, window):
        window.blit(self.IMG, (self.x1, self.y))
        window.blit(self.IMG, (self.x2, self.y))

################################### MAIN LOOP ###################################
def draw_window(window, birds, pipes, base, score, gen):
    window.blit(BG_IMG, (0, 0))

    for pipe in pipes:
        pipe.draw(window)

    text = STAT_FONT.render("Score: " + str(score), 1, (255,255,255))
    window.blit(text, (WIN_WIDTH - 10 - text.get_width(), 10))

    text = STAT_FONT.render("Gen: " + str(gen), 1, (255,255,255))
    window.blit(text, (10 , 10))

    base.draw(window)
    for bird in birds:
        bird.draw(window)
    pygame.display.update()

# genomes, config are the required params for a fitness function
def main(genomes, config):
    global GEN
    GEN+=1
    nets = []
    ge = []
    birds = []

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g,config)
        nets.append(net)
        birds.append(Bird(150, 310))
        g.fitness = 0
        ge.append(g)

    base = Base(540)
    pipes = [Pipe(600)] 
    window = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    #Set clock to set frame rate
    clock = pygame.time.Clock()
    score = 0

    run = True
    while run:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
            # FOR USER TO PLAY
            # if event.type == pygame.KEYDOWN:
            #     if event.key == pygame.K_SPACE:
            #         bird.jump()
        
        pipe_idx = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipe_idx = 1
        else:
            run = False
            break

        for x, bird in enumerate(birds):
            bird.move()
            ge[x].fitness += 0.1

            # Activate network to see if bird will jump
            output = nets[x].activate((
                    bird.y, 
                    abs(bird.y - pipes[pipe_idx].height),
                    200
                    # abs(bird.y - pipes[pipe_idx].bottom)
                ))
            
            # returns a list of output nuerons, but bird only has one.
            if output[0] > 0.5:
                bird.jump()

        rem = []
        add_pipe = False
        for pipe in pipes:
            for x, bird in enumerate(birds): 
                if pipe.collide(bird):
                    # encurages bird to go in between the pipes.
                    ge[x].fitness-=1
                    birds.pop(x)
                    nets.pop(x)
                    ge.pop(x)

            if not pipe.passed and len(birds) > 0 and pipe.x + pipe.PIPE_TOP.get_width() < birds[0].x:
                pipe.passed = True
                add_pipe = True

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)

            pipe.move()

        # If pipe was passed
        if add_pipe:
            score += 1
            # bonus fitness for making it through pipe.
            for g in ge: 
                g.fitness+=5
            pipes.append(Pipe(495))
        
        #remove pipes
        for r in rem:
            pipes.remove(r)

        for x, bird in enumerate(birds):
                # check collision with ground
            if bird.y + bird.img.get_height() >= base.y or bird.y < 0:
                ge[x].fitness -= 100
                birds.pop(x)
                nets.pop(x)
                ge.pop(x)

        bird.move()
        base.move()
        draw_window(window, birds, pipes, base, score, GEN)

################################### NEAT SETUP ###################################

def run(config_path):
    # Tells neat the headers it needs to set from config file. NEAT is assumed. Order matters.
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter())
    pop.add_reporter(neat.StatisticsReporter())

    # call fitness function
    # returns best genome from population sim
    winner = pop.run(main, 1000)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "neat-config.txt")
    run(config_path)
