import pygame
from constants import HEIGHT, WIDTH, FONT, GREEN, WHITE, BLACK, FPS


class Player:
    def __init__(self, posx, posy, width, height, speed, color):
        self.rect = pygame.Rect(posx, posy, width, height)
        self.speed = speed
        self.color = color

    def display(self, surface):
        pygame.draw.rect(surface, self.color, self.rect)

    def update(self, yFac):
        self.rect.y += self.speed * yFac
        self.rect.y = max(0, min(self.rect.y, HEIGHT - self.rect.height))

    def display_score(self, surface, label, score, x, y, color):
        text = FONT.render(f"{label}{score}", True, color)
        text_rect = text.get_rect(center=(x, y))
        surface.blit(text, text_rect)

    def _get_pos(self):
        return pygame.Vector2(self.rect.x, self.rect.y)


class Ball:
    def __init__(self, posx, posy, radius, speed, color):
        self.pos = pygame.Vector2(posx, posy)
        self.radius = radius
        self.speed = speed
        self.color = color
        self.direction = pygame.Vector2(1, -1)
        self.first_time = True

    def display(self, surface):
        pygame.draw.circle(
            surface, self.color, (int(self.pos.x), int(self.pos.y)), self.radius
        )

    def update(self):
        self.pos += self.direction * self.speed

        if self.pos.y <= 0 or self.pos.y >= HEIGHT:
            self.direction.y *= -1

        if self.pos.x <= 0 and self.first_time:
            self.first_time = False
            return 1
        elif self.pos.x >= WIDTH and self.first_time:
            self.first_time = False
            return -1
        return 0

    def reset(self):
        self.pos = pygame.Vector2(WIDTH // 2, HEIGHT // 2)
        self.direction.x *= -1
        self.first_time = True

    def hit(self):
        self.direction.x *= -1

    def get_rect(self):
        return pygame.Rect(
            int(self.pos.x - self.radius),
            int(self.pos.y - self.radius),
            self.radius * 2,
            self.radius * 2,
        )

    def _get_pos(self):
        return self.pos


class Game:
    def __init__(self, player_1=None, player_2=None):
        pygame.display.set_caption("Pong")
        self.player_1 = player_1 or Player(
            posx=20, posy=0, width=10, height=100, speed=10, color=GREEN
        )
        self.player_2 = player_2 or Player(
            posx=WIDTH - 30, posy=0, width=10, height=100, speed=10, color=GREEN
        )

        self.ball = Ball(WIDTH // 2, HEIGHT // 2, 7, 7, WHITE)
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.scores = [0, 0]
        self.is_running = True
        self.y_factors = [0, 0]

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_running = False
                pygame.quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.y_factors[1] = -1
                elif event.key == pygame.K_DOWN:
                    self.y_factors[1] = 1
                elif event.key == pygame.K_w:
                    self.y_factors[0] = -1
                elif event.key == pygame.K_s:
                    self.y_factors[0] = 1

            elif event.type == pygame.KEYUP:
                if event.key in (pygame.K_UP, pygame.K_DOWN):
                    self.y_factors[1] = 0
                elif event.key in (pygame.K_w, pygame.K_s):
                    self.y_factors[0] = 0

    def _detect_collisions(self):
        if self.ball.get_rect().colliderect(
            self.player_1.rect
        ) or self.ball.get_rect().colliderect(self.player_2.rect):
            self.ball.hit()

    def start(self):
        while self.is_running:
            self.screen.fill(BLACK)

            self._handle_events()
            # Collision detection
            self._detect_collisions()

            # Update positions
            self.player_1.update(self.y_factors[0])
            self.player_2.update(self.y_factors[1])
            point = self.ball.update()

            if point == -1:
                self.scores[0] += 1
            elif point == 1:
                self.scores[1] += 1

            if point:
                self.ball.reset()

            # Draw everything
            self.player_1.display(self.screen)
            self.player_2.display(self.screen)

            self.ball.display(self.screen)

            self.player_1.display_score(
                self.screen, "Player 1 : ", self.scores[0], 100, 20, WHITE
            )
            self.player_2.display_score(
                self.screen, "Player 2 : ", self.scores[1], WIDTH - 100, 20, WHITE
            )

            pygame.display.update()
            self.clock.tick(FPS)


def main():
    game = Game()
    game.start()


if __name__ == "__main__":
    main()
