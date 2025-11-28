import pygame
import random
import sys
import copy
import heapq
import statistics
import time

#region Core
mode = None
fast_mode = False
test_mode = False
fps = 10

GRID_SIZE = 20
GAME_WIDTH = 20
GAME_HEIGHT = 20
SCORE_HEIGHT = 40

SCREEN_WIDTH = GAME_WIDTH * GRID_SIZE
SCREEN_HEIGHT = GAME_HEIGHT * GRID_SIZE + SCORE_HEIGHT

# Dimensões ampliadas para o menu
MENU_WIDTH = 800
MENU_HEIGHT = 600

# Cores
BLACK = (0, 0, 0)

GREEN = (0, 255, 0)
DARK_GREEN = (0, 200, 0)
TRANSPARENT_GREEN = (0, 255, 0, 128)

RED = (255, 0, 0)
TRANSPARENT_RED = (255, 0, 0, 128)

YELLOW = (255, 255, 0)
DARK_GREY = (40, 40, 40)
WALL_GREY = (100, 100, 100)
SCORE = (25, 25, 25)


pygame.init()

screen = pygame.display.set_mode((SCREEN_WIDTH + GRID_SIZE*2, SCREEN_HEIGHT + GRID_SIZE))
menu_screen = pygame.display.set_mode((MENU_WIDTH, MENU_HEIGHT))

pygame.display.set_caption("Jogo da Cobrinha")
font = pygame.font.SysFont("arial", 25)

transparent_surface = pygame.Surface((SCREEN_WIDTH + GRID_SIZE*2, SCREEN_HEIGHT + GRID_SIZE), pygame.SRCALPHA)

def draw_grid():
    # Black 
    pygame.draw.rect(screen, BLACK, (GRID_SIZE, SCORE_HEIGHT, SCREEN_WIDTH, SCREEN_HEIGHT-SCORE_HEIGHT)) 

    # Vertical Lines
    for x in range(GRID_SIZE, SCREEN_WIDTH + GRID_SIZE, GRID_SIZE):
        pygame.draw.line(screen, DARK_GREY, (x, SCORE_HEIGHT), (x, SCREEN_HEIGHT))
    
    # Horizontal Lines
    for y in range(SCORE_HEIGHT, SCREEN_HEIGHT, GRID_SIZE):
        pygame.draw.line(screen, DARK_GREY, (GRID_SIZE, y), (SCREEN_WIDTH + GRID_SIZE, y))

def get_apple_coords(snake_body):
    while True:
        x = random.randrange(GRID_SIZE, SCREEN_WIDTH + GRID_SIZE, GRID_SIZE)
        y = random.randrange(SCORE_HEIGHT, SCREEN_HEIGHT, GRID_SIZE)

        if (not is_body_collision(x, y, snake_body)):
            return (x, y)

def draw_snake(snake_body, isSim = False, color = TRANSPARENT_GREEN):
    if (isSim):
        for cell in snake_body:
            pygame.draw.rect(transparent_surface, color, (cell[0], cell[1], GRID_SIZE, GRID_SIZE))  
    else:
        for i, cell in enumerate(snake_body):
            if (i == len(snake_body) - 1):
                pygame.draw.rect(screen, DARK_GREEN, (cell[0], cell[1], GRID_SIZE, GRID_SIZE))
            else:    
                pygame.draw.rect(screen, GREEN, (cell[0], cell[1], GRID_SIZE, GRID_SIZE))

def draw_apple(apple, isSim = False):
    if (isSim):
        pygame.draw.rect(transparent_surface, TRANSPARENT_RED, (apple[0], apple[1], GRID_SIZE, GRID_SIZE))
    else:
        pygame.draw.rect(screen, RED, (apple[0], apple[1], GRID_SIZE, GRID_SIZE))
    
def draw_score(score):
    pygame.draw.rect(screen, SCORE, (0, 0, SCREEN_WIDTH + GRID_SIZE*2, SCORE_HEIGHT))
    score_text = font.render(f"Score: {score}", True, YELLOW)
    screen.blit(score_text, (10, 8))

def is_wall_collision(x, y):
    if (x < GRID_SIZE or x >= SCREEN_WIDTH + GRID_SIZE):
        return True
    
    if (y < SCORE_HEIGHT or y > SCREEN_HEIGHT - GRID_SIZE):
        return True
    
    return False

def is_body_collision(x, y, snake_body):
    if (x, y) in snake_body:
        return True
    return False

def is_safe(x, y, snake_body):
    return not (is_wall_collision(x, y) or is_body_collision(x, y, snake_body))

def is_apple_collision(x, y, apple):
    if (x == apple[0] and y == apple[1]):
        return True
    return False

#endregion

#region Monte Carlo

def MT_simulation(snake_body, last_move_x, last_move_y, sim_steps, apple):
    sim_snake_body = list(copy.deepcopy(snake_body))

    x = sim_snake_body[-1][0] + last_move_x * GRID_SIZE
    y = sim_snake_body[-1][1] + last_move_y * GRID_SIZE            
    
    apples_eaten = 0

    sim_snake_body.append((x, y))
    if (is_apple_collision(x, y, apple)):
        apples_eaten += 1
        apple = get_apple_coords(sim_snake_body)
    else:
        del sim_snake_body[0]

    
    history = [list(sim_snake_body)]

    for _ in range(sim_steps):
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        random.shuffle(moves)
        valid_moves = []

        for move in moves:
            if move[0] == -last_move_x and move[1] == -last_move_y:
                continue

            x += move[0] * GRID_SIZE
            y += move[1] * GRID_SIZE

            if (not is_safe(x, y, sim_snake_body)):
                continue

            valid_moves.append((move[0], move[1]))
        
        if (valid_moves):
            move_x, move_y = random.choice(valid_moves)
            x = sim_snake_body[-1][0] + move_x * GRID_SIZE
            y = sim_snake_body[-1][1] + move_y * GRID_SIZE
            
            sim_snake_body.append((x, y))
            if (is_apple_collision(x, y, apple)):
                apples_eaten += 1
                apple = get_apple_coords(sim_snake_body)
            else:
                del sim_snake_body[0]

            history.append(list(sim_snake_body))

            last_move_x = move_x
            last_move_y = move_y
        else:
            break
    
    return apples_eaten, history

def MT_choose_move(snake_body, last_move_x, last_move_y, apple, num_sim, sim_steps):
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    random.shuffle(moves)
    valide_moves = []
    all_sim = []

    w1 = 10
    w2 = 1

    for move in moves:
        if move[0] == -last_move_x and move[1] == -last_move_y:
            continue

        x = snake_body[-1][0] + move[0] * GRID_SIZE
        y = snake_body[-1][1] + move[1] * GRID_SIZE

        if (not is_safe(x, y, snake_body)):
            continue

        total = 0
        histories = []

        for _ in range(num_sim):
            apple_eaten, history = MT_simulation(snake_body, move[0], move[1], sim_steps, apple)
        
            color = (random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255), 100)
            histories.append((color, history))

            final_pos = history[-1][-1]
            dist_x = abs(final_pos[0] - apple[0])
            dist_y = abs(final_pos[1] - apple[1])
            dist = (dist_x + dist_y) / GRID_SIZE

            initial_head = snake_body[-1]
            initial_dist_x = abs(initial_head[0] - apple[0])
            initial_dist_y = abs(initial_head[1] - apple[1])
            initial_dist = (initial_dist_x + initial_dist_y) / GRID_SIZE

            progress = (max(0, initial_dist - dist) / max(1, initial_dist))
            total += w1 * apple_eaten + w2 * progress

        # random noise to avoid draws
        average = (total / num_sim) + random.uniform(-0.01, 0.01)
        valide_moves.append(((move[0], move[1]), average))
        all_sim.append((move, histories))

    if not valide_moves:
        return (0, 0), []
    
    valide_moves.sort(key=lambda X: X[1], reverse=True)
    best_move = valide_moves[0][0]

    return best_move, all_sim

#endregion

#region Astar
def calc_h(a, b):
    # Distância Manhattan em unidades de pixels (multiplo de TAM)
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star(start, obj, snake_body):
    start = (start[0], start[1])
    obj = (obj[0], obj[1])

    heap = []
    heapq.heappush(heap, (0, start))
    fromWay = {}

    cust = {start: 0}
    h_score = {start: calc_h(start, obj)}

    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    random.shuffle(moves)

    while heap:
        _, actual = heapq.heappop(heap)
        if actual == obj:
            # reconstruir caminho
            way = []

            cell = actual
            while cell in fromWay:
                way.append(cell)
                cell = fromWay[cell]

            way.reverse()
            return way

        for dx, dy in moves:
            move = (actual[0] + dx * GRID_SIZE, actual[1] + dy * GRID_SIZE)

            if not is_safe(move[0], move[1], snake_body) and move != obj:
                continue

            score = cust[actual] + 1
            if move not in cust or score < cust[move]:
                fromWay[move] = actual
                cust[move] = score
                h_score[move] = score + calc_h(move, obj)

                # heap is sorted by fscore
                # so smallers moves have priority
                heapq.heappush(heap, (h_score[move], move))
    return None
#endregion

#region Simplex
def count_free_neighbors(pos, snake_body):
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    free = 0
    for dx, dy in moves:
        x, y = pos[0] + dx * GRID_SIZE, pos[1] + dy * GRID_SIZE
        if is_safe(x, y, snake_body):
            free += 1
    return free

def simplex_tableau(c, A, b, eps=1e-9, max_iter=1000):
    m = len(A)
    n = len(c)
    tableau = [[0.0]*(n+m+1) for _ in range(m+1)]

    # Montagem das restrições
    for i in range(m):
        for j in range(n):
            tableau[i][j] = float(A[i][j])
        tableau[i][n+i] = 1.0
        tableau[i][-1] = float(b[i])

    # Linha da função objetivo
    for j in range(n):
        tableau[m][j] = -float(c[j])

    basis = [n+i for i in range(m)]

    def pivot(r, c):
        p = tableau[r][c]
        tableau[r] = [v/p for v in tableau[r]]
        for i in range(m+1):
            if i != r:
                f = tableau[i][c]
                tableau[i] = [tableau[i][j] - f*tableau[r][j] for j in range(n+m+1)]

    it = 0
    while it < max_iter:
        it += 1
        entering = None
        mv = -eps
        for j in range(n+m):
            if tableau[m][j] < mv:
                mv = tableau[m][j]
                entering = j

        if entering is None:
            x = [0.0]*n
            for i in range(m):
                if basis[i] < n:
                    x[basis[i]] = tableau[i][-1]
            return x, tableau[m][-1]

        leaving = None
        mr = float('inf')
        for i in range(m):
            a = tableau[i][entering]
            if a > eps:
                r = tableau[i][-1] / a
                if r < mr:
                    mr = r
                    leaving = i

        if leaving is None:
            return None, None

        basis[leaving] = entering
        pivot(leaving, entering)

    return None, None

def simplex(coefs):
    n = len(coefs)
    A = []
    b = []

    # Σ pesos = 1
    A.append([1.0]*n); b.append(1.0)
    # Σ pesos >= 1 (invertido para Simplex)
    A.append([-1.0]*n); b.append(-1.0)

    c = [float(x) for x in coefs]

    x, val = simplex_tableau(c, A, b)

    if x is None:
        return [1.0/n]*n

    w = [max(0.0, xi) for xi in x]
    s = sum(w)

    if s == 0:
        return [1.0/n]*n

    return [wi/s for wi in w]

def simplex_move(head, move_x, move_y, snake_body, apple):
    """
    Estima ganhos locais e resolve o LP via simplex.
    Em seguida escolhe o movimento com melhor score combinado.
    """
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    random.shuffle(moves)

    valid_moves = []
    for dx, dy in moves:
        if dx == -move_x and dy == -move_y:
            continue

        nx = head[0] + dx * GRID_SIZE
        ny = head[1] + dy * GRID_SIZE

        if is_safe(nx, ny, snake_body):
            valid_moves.append((nx, ny))

    if not valid_moves:
        return 0, 0

    scores = []
    for move in valid_moves:
        distance = abs(move[0] - apple[0]) + abs(move[1] - apple[1])
        safety_scores = count_free_neighbors(move, snake_body) / 4.0

        if distance != 0:
            dist_scores = 1 / distance ** 0.5
        else:
            dist_scores = 1

        

        coefs = [dist_scores, safety_scores]
        weights = simplex(coefs)

        # if apple is in any corner, the distance get priority
        # to avoid loops due security        
        if apple[0] in (GRID_SIZE, SCREEN_WIDTH) and apple[1] in (SCORE_HEIGHT, SCREEN_HEIGHT - GRID_SIZE):
            weights[0] = max(weights[0], 0.8)
            weights[1] = 1 - weights[0]



        if weights[0] < 0.6:
            weights[0] = 0.6
            weights[1] = 0.4
        
        score = evaluate_move(move, apple, snake_body, weights)
        scores.append((move, score))

    # avaliar candidatos e escolher o melhor
    best = None
    best_score = -1
    for move, score in scores:
        if score > best_score:
            best_score = score
            best = (move[0] - head[0], move[1] - head[1])

    return best if best else (0, 0)

def evaluate_move(move, apple, snake_body, weights):
    """Combina heurísticas de distância e segurança."""
    max_dist = SCREEN_WIDTH + SCREEN_HEIGHT
    
    distance = abs(move[0] - apple[0]) + abs(move[1] - apple[1])
   
    if distance != 0:
        h_dist = 1 / distance ** 0.5
    else:
        h_dist = 1

    #h_dist = 1 - (distance / max_dist)
    h_seg = count_free_neighbors(move, snake_body) / 4.0
    
    return weights[0] * h_dist + weights[1] * h_seg
#endregion

#region Game

def run_tests(n = 50, fast_mode = True):
    start_time = time.time()
    print(f"Executando {n} partidas automaticas no modo {mode}:")
    results = []

    for i in range(n):
        game_start = time.time() # Tempo de inicio da partida

        metrics = game(auto = True, visualize = False, fast_mode = fast_mode)
        
        game_end = time.time()  # Tempo de fim da partida
        duration = game_end - game_start

        metrics['duration'] = duration  # Armazena o tempo no resultado
        results.append(metrics)
        print(f"Jogo {i+1}/{n}: Score = {metrics['score']} Steps = {metrics['steps']} Tempo = {duration:.2f}")

    scores = [r['score'] for r in results]
    durations = [r['duration'] for r in results]
    

    avg_score = sum(scores) / len(scores)
    std_score = statistics.stdev(scores)

    avg_duration = sum(durations) / len(durations)
    avg_steps = sum(r['steps'] for r in results) / len(results)

    
    wall_death = sum(r['wall_hits'] for r in results)
    body_death = sum(r['body_hits'] for r in results)

    total_time = time.time() - start_time

    print("-=- Resultados Gerais -=-")
    print(f"Partidas: {n}")
    print(f"Média de Score: {avg_score:.2f}")
    print(f"Desvio Padrão do Score: {std_score:.2f}")
    print(f"Média de Passos: {avg_steps:.2f}")
    print(f"Média de Tempo por Jogo: {avg_duration:.2f}s")
    print(f"Colisoes com Parede: {wall_death}")
    print(f"Colisoes com Corpo: {body_death}")
    print(f"Tempo total de execução: {total_time:.2f} segundos")

    return results

def game(auto, visualize, fast_mode):
    # initial x and initial y, at the center of the screen
    x = SCREEN_WIDTH // 2
    y = SCREEN_HEIGHT // 2
    move_x = 0
    move_y = 0

    snake_body = [(x, y)]
    snake_size = 1

    apple = get_apple_coords(snake_body)

    phase = "decide"
    sim_data = None
    sim_index = 0

    steps = 0
    wall_hits = 0
    body_hits = 0

    if (fast_mode):
        fps = 200
    else:
        fps = 10

    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if (mode == "montecarlo"):
            if (snake_size < 10):
                num_sim = 30
                sim_steps = 15
            else:
                num_sim = 15
                sim_steps = 25

            if (phase == "decide"):
                best_move, all_sim = MT_choose_move(snake_body, move_x, move_y, apple, num_sim, sim_steps)

                move_x = best_move[0]
                move_y = best_move[1]

                sim_data = all_sim
                sim_index = 0

                phase = "show_sims"
                continue
            elif (phase == "show_sims"):
                active = False
                if not auto:
                    if visualize:
                        transparent_surface.fill((0, 0, 0, 0))

                        for move, histories in sim_data:
                            for color, history in histories:
                                if (sim_index < len(history)):
                                    draw_snake(history[sim_index], True, color)
                                    active = True

                    screen.fill(WALL_GREY)
                    draw_grid()
                    draw_snake(snake_body)
                    draw_apple(apple)
                    draw_score(snake_size)
                    screen.blit(transparent_surface, (0, 0))
                    pygame.display.update()
                    clock.tick(fps)

            if not active:
                x += move_x * GRID_SIZE
                y += move_y * GRID_SIZE
                steps += 1

                if is_wall_collision(x, y):
                    wall_hits += 1
                    break

                if is_body_collision(x, y, snake_body[:-1]):
                    body_hits += 1
                    break

                snake_body.append((x, y))
                if (is_apple_collision(x, y, apple)):
                    snake_size += 1
                    apple = get_apple_coords(snake_body)
                else:
                    del snake_body[0]
                
                phase = "decide"
            else:
                sim_index += 1
                continue

        elif (mode == "astar"):
            way = a_star((x, y), apple, snake_body[:-1])

            if (way and len(way) > 0):
                next_move = way[0]
                move_x = next_move[0] - x
                move_y = next_move[1] - y

            if not auto:
                screen.fill(WALL_GREY)
                draw_grid()
                draw_snake(snake_body)
                draw_apple(apple)
                draw_score(snake_size)
                screen.blit(transparent_surface, (0, 0))
                pygame.display.update()
                clock.tick(fps)

            x += move_x
            y += move_y
            steps += 1

            if is_wall_collision(x, y):
                wall_hits += 1
                break

            if is_body_collision(x, y, snake_body[:-1]):
                body_hits += 1
                break

            snake_body.append((x, y))
            if (is_apple_collision(x, y, apple)):
                snake_size += 1
                apple = get_apple_coords(snake_body)
            else:
                del snake_body[0]
        elif (mode == "simplex"):
            move_x, move_y = simplex_move((x, y), move_x, move_y, snake_body, apple)

            if not auto:
                screen.fill(WALL_GREY)
                draw_grid()
                draw_snake(snake_body)
                draw_apple(apple)
                draw_score(snake_size)
                screen.blit(transparent_surface, (0, 0))
                pygame.display.update()
                clock.tick(fps)

            x += move_x
            y += move_y
            steps += 1

            if is_wall_collision(x, y):
                wall_hits += 1
                break

            if is_body_collision(x, y, snake_body[:-1]):
                body_hits += 1
                break

            snake_body.append((x, y))
            if (is_apple_collision(x, y, apple)):
                snake_size += 1
                apple = get_apple_coords(snake_body)
            else:
                del snake_body[0]

    if not auto:
        screen.fill(BLACK)
        text = font.render(f"Game Over! Score: {snake_size - 1}", True, YELLOW)
        screen.blit(text, ((SCREEN_WIDTH + 2*GRID_SIZE) // 4, (SCREEN_HEIGHT + GRID_SIZE) // 2))
        pygame.display.update()
        pygame.time.wait(1500)

    return{
        "score" : snake_size-1,
        "steps" : steps,
        "wall_hits" : wall_hits,
        "body_hits" : body_hits
    }


def menu():
    """
    Menu com etapas:
    1) Seleciona tipo de execução (Jogar / Teste Automático)
    2) Seleciona modo de IA (Monte Carlo / A* / Simplex)
    3) (Opcional) Seleciona velocidade se for modo Jogar
    """
    global mode, fast_mode, test_mode

    run_options = [
        ("Tempo Real", False),
        ("Teste Automático", True)
    ]

    ia_options = [
        ("Monte Carlo", "montecarlo"),
        ("A*", "astar"),
        ("Simplex", "simplex")
    ]

    speed_options = [
        ("Normal", False),
        ("Rápido", True)
    ]

    selected_run = 0
    selected_mode = 0
    selected_speed = 0
    stage = 0  # 0 = tipo de execução, 1 = algoritmo, 2 = velocidade

    clock = pygame.time.Clock()
    running = True

    while running:
        menu_screen.fill(BLACK)
        title = font.render("Configuração do Jogo", True, YELLOW)
        title_x = (MENU_WIDTH - title.get_width()) // 2
        menu_screen.blit(title, (title_x, 40))

        # 0 - Tipo de execucao
        subtitle = font.render("1) Selecione o modo de execução:", True, YELLOW)
        menu_screen.blit(subtitle, (50, 90))
        for i, (label, _) in enumerate(run_options):
            color = (0, 200, 0) if (i == selected_run and stage == 0) else (255, 255, 255)
            txt = font.render(f"{i+1} - {label}", True, color)
            txt_x = (MENU_WIDTH - txt.get_width()) // 2
            menu_screen.blit(txt, (txt_x, 130 + i * 36))

        # 1 - Escolha Algoritmo
        subtitle2 = font.render("2) Selecione o algoritmo:", True, YELLOW)
        y_ia = 130 + len(run_options) * 40 + 20
        menu_screen.blit(subtitle2, (50, y_ia))
        for i, (label, _) in enumerate(ia_options):
            color = (0, 200, 0) if (i == selected_mode and stage == 1) else (255, 255, 255)
            alpha = 255 if stage >= 1 else 120
            txt = font.render(f"{i+1} - {label}", True, color)
            txt.set_alpha(alpha)
            txt_x = (MENU_WIDTH - txt.get_width()) // 2
            menu_screen.blit(txt, (txt_x, y_ia + 36 + i * 36))

        # 2 - Velocidade (somente se nao for auto) 
        y_speed = y_ia + 36 + len(ia_options) * 36 + 20
        subtitle3 = font.render("3) Selecione a velocidade:", True, YELLOW)
        menu_screen.blit(subtitle3, (50, y_speed))
        for i, (label, _) in enumerate(speed_options):
            color = (0, 200, 0) if (i == selected_speed and stage == 2) else (255, 255, 255)
            alpha = 255 if (stage >= 2 and not test_mode) else 120
            txt = font.render(f"{i+1} - {label}", True, color)
            txt.set_alpha(alpha)
            txt_x = (MENU_WIDTH - txt.get_width()) // 2
            menu_screen.blit(txt, (txt_x, y_speed + 36 + i * 36))

        # Instrucoes 
        instr = font.render("↑ ↓ para navegar, ENTER para confirmar, ESC para sair", True, (200, 200, 200))
        instr_x = (MENU_WIDTH - instr.get_width()) // 2
        menu_screen.blit(instr, (instr_x, MENU_HEIGHT - 60))

        pygame.display.update()
        clock.tick(30)

        # Eventos 
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

                if event.key == pygame.K_UP:
                    if stage == 0:
                        selected_run = (selected_run - 1) % len(run_options)
                    elif stage == 1:
                        selected_mode = (selected_mode - 1) % len(ia_options)
                    elif stage == 2 and not test_mode:
                        selected_speed = (selected_speed - 1) % len(speed_options)

                elif event.key == pygame.K_DOWN:
                    if stage == 0:
                        selected_run = (selected_run + 1) % len(run_options)
                    elif stage == 1:
                        selected_mode = (selected_mode + 1) % len(ia_options)
                    elif stage == 2 and not test_mode:
                        selected_speed = (selected_speed + 1) % len(speed_options)

                elif event.key == pygame.K_RETURN:
                    if stage == 0:
                        test_mode = run_options[selected_run][1]
                        stage = 1

                    elif stage == 1:
                        mode = ia_options[selected_mode][1]
                        # se for teste, não pede velocidade
                        if test_mode:
                            fast_mode = True
                            running = False
                        else:
                            stage = 2

                    elif stage == 2 and not test_mode:
                        fast_mode = speed_options[selected_speed][1]
                        running = False

    # returns to Game Screen display
    pygame.display.set_mode((SCREEN_WIDTH + GRID_SIZE*2, SCREEN_HEIGHT + GRID_SIZE))

if __name__ == "__main__":
    mode = None
    fast_mode = False
    test_mode = False

    menu()

    if test_mode:
        # Roda em modo automático com fast_mode sempre True
        run_tests(100, fast_mode=True)
    else:
        # Jogo manual
        game(auto=False, visualize=not fast_mode, fast_mode=fast_mode)

#endregion