'''
Atividade Prática: Implementação e Comparação de Agentes Racionais

Alunas: Amanda Ameida Carsoso, Amanda dos Santos Almeida, Paloma Santos Ferreira

'''

from mesa import Agent, Model
from mesa.space import SingleGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import random

OBSTACLE = -1  # obstáculo
EMPTY = 0  # limpo

EMOJIS = {
    "AGENTE": "🤖",
    "OBSTACULO": "🚫",  # obstáculo
    "POEIRA": "🧹",  # pó
    "LIQUIDO": "💧",  # gota
    "DETRITOS": "🗑",  # lixo
    "VAZIO": "🗒",  # espaço vazio
}
DIRT_POINTS = {
    "POEIRA": 1,
    "LIQUIDO": 2,
    "DETRITOS": 3,
}

ACTION_COST = {
    "MOVE": 1,
    "CLEAN": 2,
    "HALT": 0,
}
from dataclasses import dataclass


@dataclass(frozen=True)
class Coord:
    x: int
    y: int


# AGENTE SIMPLES DIVIDIDO EM FUNÇÕES
class VacuumSimpleAgent(Agent):
    def __init__(self, model):
        super().__init__(model.next_id(), model)
        self.pontos = 0
        self.battery = 30  # nível de bateria
        self.qt_steps = 0  # quantidade de passos dados
        self.cleaned_cells = 0  # contador de células limpas

    def step(self):
        pos = Coord(*self.pos)  # pega a posição atual
        self.clean(pos)
        self.move(pos)
        self.qt_steps += 1

    def clean(self, pos):
        # 1) limpar
        # pega o valor de pontuação da sujeira
        points = self.model.layer[pos.x][pos.y]
        if points > 0 and self.battery >= ACTION_COST["CLEAN"]:
            self.battery -= ACTION_COST["CLEAN"]
            print(
                f"Célula limpa: {pos}; Sujeira: {self.model.get_dirt_label(pos)}; Pontos: +{points} ; Bateria descontada: -{ACTION_COST['CLEAN']};")
            self.pontos += points
            self.cleaned_cells += 1
            self.model.layer[pos.x][pos.y] = EMPTY
        else:
            if (self.model.layer[pos.x][pos.y] == EMPTY):
                print(f"Agente {self.unique_id} em local vazio")
            else:
                print(f"Agente {self.unique_id} está sem bateria e não pode limpar.")

    def move(self, pos):
        # 2) mover
        if (self.battery < ACTION_COST["MOVE"]):
            print(f"Agente {self.unique_id} está sem bateria e não pode se mover.")
            return

        neighbors = self.perception(pos)
        self.random.shuffle(neighbors)

        # criando lista com coordenadas válidas (sujas e limpas, ou seja, exclui as que não são obstáculos)
        valid_cells = self.get_valid_cells(neighbors)

        # entre as válidas, pegamos as sujas
        dirty_cells = self.get_dirty_cells(valid_cells)

        # escolhe o alvo
        target = self.get_target_cell(dirty_cells, valid_cells)

        if target is not None:
            self.model.grid.move_agent(self, (target.x, target.y))
            self.battery -= ACTION_COST["MOVE"]
            print(f"Célula alvo: {target} ; Bateria descontada: -{ACTION_COST['MOVE']}")

    def perception(self, pos):
        # Função para perceber o ambiente ao redor
        neighbors = [
            Coord(nx, ny) for (nx, ny) in
            self.model.grid.get_neighborhood((pos.x, pos.y), moore=False, include_center=False)
        ]
        return neighbors

    def get_valid_cells(self, neighbors):
        # Função para obter células válidas (não obstáculos e vazias)
        v = [
            c for c in neighbors
            if self.model.layer[c.x][c.y] != OBSTACLE
               and self.model.grid.is_cell_empty((c.x, c.y))
        ]
        return v

    def get_dirty_cells(self, valid_cells):
        # Função para obter células sujas entre as válidas
        d = [c for c in valid_cells if self.model.layer[c.x][c.y] > 0]
        return d

    def get_target_cell(self, dirty_cells, valid_cells):
        # Função para determinar a célula alvo
        target = None
        if dirty_cells:
            target = self.random.choice(dirty_cells)  # aleatória entre sujas
        elif valid_cells:
            target = self.random.choice(valid_cells)  # aleatória entre as limpas
        return target


# AGENTE BASEADO EM MODELO
class VacuumModelBasedAgent(VacuumSimpleAgent):
    def __init__(self, model):
        super().__init__(model)

        # guarda a coordenada (x,y) e o valor da célula (-1, 0, > 0) e o status (visitado ou não)
        self.known_map: dict[tuple[int, int], tuple[int, int]] = {}

    def step(self):
        pos = Coord(*self.pos)  # pega a posição atual
        self.clean(pos)
        self.move(pos)
        self.print_known_world()
        self.qt_steps += 1

    def move(self, pos):
        # 2) mover
        if (self.battery < ACTION_COST["MOVE"]):
            print(f"Agente {self.unique_id} está sem bateria e não pode se mover.")
            return

        # guarda os vizinhos imediatos
        (self.add_cell_as_visited(pos))  # atualiza o conhecimento da célula atual após limpar
        neighbors = self.perception(pos)
        self.random.shuffle(neighbors)

        valid_cells = self.get_valid_cells(neighbors)
        dirty_cells = self.get_dirty_cells(valid_cells)
        target = self.get_target_cell(dirty_cells, valid_cells)

        if target is not None:
            self.model.grid.move_agent(self, (target.x, target.y))
            self.battery -= ACTION_COST["MOVE"]
            print(f"Célula alvo: {target} ; Bateria descontada: -{ACTION_COST['MOVE']}")

    def get_target_cell(self, dirty_cells, valid_cells):
        # Função para determinar a célula alvo
        target = None
        if dirty_cells:
            target = self.random.choice(dirty_cells)  # aleatória entre sujas
        elif valid_cells:
            # prioriza células nunca visitadas (0 = não visitado, 1 = visitado)
            unknown_cells = [c for c in valid_cells if
                             (c.x, c.y) in self.known_map and self.known_map[(c.x, c.y)][1] == 0]
            if unknown_cells:
                target = self.random.choice(unknown_cells)
            else:
                target = self.random.choice(valid_cells)  # aleatória entre as limpas
        return target

    def perception(self, pos):
        # Função para perceber o ambiente ao redor
        neighbors = [
            Coord(nx, ny) for (nx, ny) in
            self.model.grid.get_neighborhood((pos.x, pos.y), moore=False, include_center=False)
        ]
        # add vizinhos no mundo conhecido
        for n in neighbors:
            if (n.x, n.y) not in self.known_map:
                self.add_cell_to_known_map(n)
        return neighbors

    def add_cell_as_visited(self, pos):

        key = (pos.x, pos.y)
        if key in self.known_map:

            self.known_map[key] = (EMPTY, 1)  # marca como visitado

        else:  # primeira célula visitada que ainda não existe
            self.known_map[key] = (self.model.layer[key[0]][key[1]], 1)

    def add_cell_to_known_map(self, pos):

        key = (pos.x, pos.y)

        if key not in self.known_map:
            print(f"Agente {self.unique_id} descobriu célula {pos} com valor {self.model.layer[pos.x][pos.y]}")
            # registrar célula atual como conhecida
            self.known_map[key] = (self.model.layer[key[0]][key[1]], 0)  # 0 = não visitado

    def print_known_world(self):
        print("\n=== Mundo conhecido pelo agente a cada passo ===")
        for (x, y), (valor, visitado) in self.known_map.items():
            status = "visitado" if visitado == 1 else "não visitado"
            if valor == OBSTACLE:
                celula = "OBSTÁCULO"
            elif valor == EMPTY:
                celula = "VAZIA"
            else:
                celula = f"SUJEIRA ({valor} pontos)"
            print(f"({x}, {y}) -> {celula}, {status}")


# AGENTE BASEADO EM UTILIDADES
class VacuumUtilityBasedAgent(VacuumModelBasedAgent):
    def __init__(self, model):
        super().__init__(model)
        self.goal = "LIMPAR_TODAS"
        self.last_pos = None
        self.failed_targets = set()  # guarda obstáculos

    def step(self):
        pos = Coord(*self.pos)
        self.clean(pos)
        self.add_cell_as_visited(pos)
        self.perception(pos)
        self.move(pos)
        self.print_known_world()
        self.qt_steps += 1

    def move(self, pos):
        if self.battery < ACTION_COST["MOVE"]:
            print(f"Agente {self.unique_id} sem bateria para mover.")
            return

        if self.last_pos == pos:
            self.failed_targets.add(pos)

        # procura sujeiras conhecidas
        dirty_cells = [
            Coord(x, y)
            for (x, y), (valor, visitado) in self.known_map.items()
            if valor > 0 and (x, y) not in self.failed_targets
        ]

        # Se tiver sujeira, vai para a com o maior valor
        if dirty_cells:
            target = self.choose_best_dirty_cell(pos, dirty_cells)
        else:

            target = self.choose_exploration_target(
                pos)  # Se não tiver sujeira, escolhe célula desconhecida mais próxima

        if not target:
            print("Nenhum objetivo disponível. Agente decide parar.")
            self.battery = 0
            return

        next_step = self.step_toward(pos, target)

        if next_step and self.model.layer[next_step.x][next_step.y] != OBSTACLE:
            self.model.grid.move_agent(self, (next_step.x, next_step.y))
            self.battery -= ACTION_COST["MOVE"]
            print(f"Movendo em direção a {target} → próxima célula {next_step}")
            self.last_pos = next_step
        else:
            self.failed_targets.add((target.x, target.y))
            self.last_pos = pos

    def choose_best_dirty_cell(self, pos, dirty_cells):
        def score(c):
            valor = self.known_map[(c.x, c.y)][0]
            dist = abs(pos.x - c.x) + abs(pos.y - c.y)
            return (-valor, dist)

        dirty_cells.sort(key=score)
        return dirty_cells[0] if dirty_cells else None

    def choose_exploration_target(self, pos):
        unexplored = [
            Coord(x, y)
            for (x, y), (valor, visitado) in self.known_map.items()
            if visitado == 0 and valor != OBSTACLE and (x, y) not in self.failed_targets
        ]
        if unexplored:
            # célula desconhecida mais próxima
            unexplored.sort(key=lambda c: abs(pos.x - c.x) + abs(pos.y - c.y))
            return unexplored[0]
        return None

    def step_toward(self, start, goal):
        # tenta mover um passo válido na direção aproximada do alvo
        dx = goal.x - start.x
        dy = goal.y - start.y

        directions = []
        if dx != 0:
            directions.append((1 if dx > 0 else -1, 0))
        if dy != 0:
            directions.append((0, 1 if dy > 0 else -1))

        for d in directions:
            nx, ny = start.x + d[0], start.y + d[1]
            if (0 <= nx < self.model.grid.width and
                    0 <= ny < self.model.grid.height and
                    self.model.layer[nx][ny] != OBSTACLE):
                return Coord(nx, ny)
        return None


# AGENTE BASEADO EM OBJETIVOS (COM MEMÓRIA DE SUJEIRA GLOBAL E MOVIMENTO PASSO A PASSO)
class VacuumGoalBasedAgent(VacuumModelBasedAgent):
    def __init__(self, model):
        super().__init__(model)
        self.last_pos = None
        self.failed_targets = set()  # Células inacessíveis (obstáculos ou becos)
        self.found_dirty_cells = set()  # Memória de sujeiras descobertas

    def step(self):
        pos = Coord(*self.pos)
        self.clean(pos)
        self.add_cell_as_visited(pos)
        self.move(pos)
        self.print_known_world()
        self.qt_steps += 1

    def move(self, pos):
        if self.battery < ACTION_COST["MOVE"]:
            print(f"Agente {self.unique_id} sem bateria para mover.")
            return

        if self.last_pos == pos:  # evitar ficar preso, tira as sujeiras da mémoria global que ja foram limpas
            self.failed_targets.add((pos.x, pos.y))

        # percepção local
        neighbors = self.perception(pos)
        valid_cells = self.get_valid_cells(neighbors)
        dirty_neighbors = self.get_dirty_cells(valid_cells)

        # Registra a sujeira local na mémoria global de sujeira
        for c in dirty_neighbors:
            self.found_dirty_cells.add((c.x, c.y))

        # ESCOLHER CÉLULA QUE DEVE LIMPAR
        if dirty_neighbors:  # escolhe aleatoriamente vizinhos sujos
            target = self.random.choice(dirty_neighbors)
            print(f"Sujeira próxima detectada em {target}, indo limpar...")
        elif self.found_dirty_cells:
            # lista de células sujas lembradas e ainda acessíveis
            remembered_dirty = [
                Coord(x, y)
                for (x, y) in self.found_dirty_cells
                if (x, y) not in self.failed_targets
            ]
            if remembered_dirty:
                target = self.choose_closest_dirty_cell(pos, remembered_dirty)  # vai pra sujeira mais próxima
                print(f"Indo em direção à sujeira lembrada mais próxima: {target}")
            else:
                target = self.choose_exploration_target(pos)  # volta a andar pelo grid (vazio)
                print("Todas sujeiras conhecidas inacessíveis, explorando...")
        else:
            target = self.choose_exploration_target(pos)
            print("Nenhuma sujeira vista ainda, explorando...")

        # MOVE UM PASSO EM DIREÇÃO AO ALVO (HEURÍSTICA LOCAL)
        self.move_towards(target)

    # Move em uma direção
    def move_towards(self, target):
        self.DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # (baixo, cima, direita, esquerda)
        # Controle de movimento
        moved = False
        start = Coord(*self.pos)

        melhor_dir = None
        menor_dist = float("inf")

        for d in self.DIRECTIONS:
            nx, ny = start.x + d[0], start.y + d[1]  # tests cada direção possivel

            # Ignora fora do grid
            if not (0 <= nx < self.model.grid.width and 0 <= ny < self.model.grid.height):
                continue

            # Ignora obstáculos
            if self.model.layer[nx][ny] == OBSTACLE:
                continue

            # calcula qual o vizinho mais promissor
            dist = abs(target.x - nx) + abs(target.y - ny)
            if dist < menor_dist:
                menor_dist = dist
                melhor_dir = (nx, ny)

        # Nenhuma direção válida
        if melhor_dir is None:
            print(f"[DEBUG] Nenhum movimento possível a partir de {start}")
            return

        # Tenta mover
        if self.model.grid.is_cell_empty(melhor_dir):
            self.model.grid.move_agent(self, melhor_dir)
            self.battery -= ACTION_COST["MOVE"]
            self.last_pos = Coord(*melhor_dir)
            moved = True
            print(f"Movendo de {start} → {melhor_dir} em direção a {target}")
        else:
            print(f"[DEBUG] Célula {melhor_dir} ocupada — movimento ignorado.")

        # Checa se realmente moveu
        if not moved:
            self.failed_targets.add((target.x, target.y))
            print(f"[DEBUG] Não foi possível mover em direção a {target}, marcando como inacessível.")

    # soma das diferenças absolutas das coordenadas cartesianas
    def choose_closest_dirty_cell(self, pos, dirty_cells):
        return min(dirty_cells, key=lambda c: abs(c.x - pos.x) + abs(c.y - pos.y))

    # buscando células ja vazias e não-obstáculo
    def choose_exploration_target(self, pos):
        unexplored = [
            Coord(x, y)
            for (x, y), (valor, visitado) in self.known_map.items()
            if visitado == 0 and valor != OBSTACLE and (x, y) not in self.failed_targets
        ]
        if unexplored:
            unexplored.sort(key=lambda c: abs(pos.x - c.x) + abs(pos.y - c.y))
            return unexplored[0]
        return None


class VacuumModel(Model):
    def __init__(self, width, height, obstacle_prob=0.18, dirt_probs=None, seed=None, agent_type=None):
        super().__init__(seed=seed)
        self.grid = SingleGrid(width, height, torus=False)
        self.schedule = RandomActivation(self)
        self.agent_type = agent_type

        # probs de sujeira (restante fica vazio)
        if dirt_probs is None:
            dirt_probs = {"POEIRA": 0.30, "LIQUIDO": 0.20, "DETRITOS": 0.10}
        self.dirt_probs = dirt_probs
        self.obstacle_prob = obstacle_prob

        # --- ÚNICA LAYER ---
        self.layer = [[EMPTY for _ in range(height)] for _ in range(width)]
        self._randomize_layer()

        print(f"Grid sem agente: ")
        self.render_text()

        # posiciona um agente em célula livre
        free = [(x, y) for x in range(width) for y in range(height) if self.layer[x][y] != OBSTACLE]
        ax, ay = self.random.choice(free)
        if agent_type == 1:
            agent = VacuumSimpleAgent(self)
        elif agent_type == 2:
            agent = VacuumModelBasedAgent(self)
        elif agent_type == 3:
            agent = VacuumGoalBasedAgent(self)
        elif agent_type == 4:
            agent = VacuumUtilityBasedAgent(self)
        elif agent_type == 5:
            agent = VacuumModelBasedAgent(self)
        else:
            raise ValueError(f"Tipo de agente inválido: {agent_type}")

        self.grid.place_agent(agent, (ax, ay))
        self.schedule.add(agent)

        self.datacollector = DataCollector(model_reporters={"TotalPontos": self._total_points})

    def _randomize_layer(self):
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                if self.random.random() < self.obstacle_prob:
                    self.layer[x][y] = OBSTACLE
                else:
                    r = self.random.random()
                    acc = 0.0
                    val = EMPTY
                    for k, p in self.dirt_probs.items():
                        acc += p
                        if r < acc:
                            val = DIRT_POINTS[k]
                            break
                    self.layer[x][y] = val

    def get_dirt_label(self, pos):
        v = self.layer[pos.x][pos.y]
        for k, p in DIRT_POINTS.items():
            if v == p:
                return k
        return None

    def _total_points(self):
        s = 0
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                v = self.layer[x][y]
                if v > 0:
                    s += v
        return s

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()

    def render_text(self):
        for y in range(self.grid.height - 1, -1, -1):
            row = []
            for x in range(self.grid.width):
                if self.layer[x][y] == OBSTACLE:
                    ch = EMOJIS["OBSTACULO"]
                elif not self.grid.is_cell_empty((x, y)):
                    ch = EMOJIS["AGENTE"]
                else:
                    v = self.layer[x][y]
                    if v == EMPTY:
                        ch = EMOJIS["VAZIO"]
                    elif v == DIRT_POINTS["POEIRA"]:
                        ch = EMOJIS["POEIRA"]
                    elif v == DIRT_POINTS["LIQUIDO"]:
                        ch = EMOJIS["LIQUIDO"]
                    elif v == DIRT_POINTS["DETRITOS"]:
                        ch = EMOJIS["DETRITOS"]
                    else:
                        ch = EMOJIS["VAZIO"]  # fallback
                row.append(ch)
            print("".join(row))  # sem espaço, só emojis


def choose_agent():
    print("##### MENU DE AGENTE #####")
    print("0 - Sair \n"
          "1 - Agente reativo simples \n"
          "2 - Agente baseado em modelos \n"
          "3 - Agente baseado em objetivos \n"
          "4 - Agente baseado em utilidade \n"
          )
    opcao = int(input("Digite o número do agente que deseja selecionar: "))
    return opcao


def main():
    print("===== COMPARAÇÃO DE AGENTES RACIONAIS =====")

    while True:
        agent_type = choose_agent()

        if agent_type == 0:
            print("Encerrando o programa. Até logo!")
            break

        modelo = VacuumModel(5, 5, obstacle_prob=0.2, seed=25, agent_type=agent_type)
        print("Iniciando modelo de aspirador simples")
        print("Custos das sujeiras: ", DIRT_POINTS)
        print("Custo das ações: ", ACTION_COST)
        print("Emojis: ", EMOJIS)

        print("\n=== Estado inicial ===")
        agente = modelo.schedule.agents[0]
        print(
            f"Agente {agente.unique_id} iniciado na posição {agente.pos} com bateria {agente.battery} e pontos {agente.pontos}")

        modelo.render_text()

        while agente.battery > 0:
            step = modelo.schedule.time + 1
            print(f"\n=== Step {step} ===")
            modelo.render_text()
            modelo.step()
            for agent in modelo.schedule.agents:
                print(
                    f"Agente {agente.unique_id} -> pontos totais: {agente.pontos} ; bateria restante: {agente.battery}")

        print(f"\n⚡ Bateria esgotada! Fim da simulação do agente {agente.unique_id}")
        print(f"🏁 Total de pontos: {agente.pontos}")
        print(f"🕒 Total de passos: {agente.qt_steps}")
        print(f"🧹 Células limpas: {agente.cleaned_cells}")
        print("=" * 40)


main()