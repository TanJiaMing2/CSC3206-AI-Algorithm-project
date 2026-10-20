import math
import heapq
import itertools
import pygame

NODES = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
START, GOAL = 'A', 'G'

RAW = {
    'A': {'A': None,   'B': 0.0000, 'C': 0.0315, 'D': 0.0559, 'E': 0.1687, 'F': 0.2330, 'G': 0.8867},
    'B': {'A': 0.0000, 'B': None,   'C': 0.0389, 'D': 0.0670, 'E': 0.1513, 'F': 0.2112, 'G': 0.9561},
    'C': {'A': 0.0315, 'B': 0.0389, 'C': None,   'D': 0.0452, 'E': 0.1815, 'F': 0.2466, 'G': 0.9152},
    'D': {'A': 0.0944, 'B': 0.0859, 'C': 0.0748, 'D': None,   'E': 0.1685, 'F': 0.2385, 'G': 0.8397},
    'E': {'A': 0.1835, 'B': 0.1590, 'C': 0.1858, 'D': 0.1344, 'E': None,   'F': 0.2838, 'G': 0.9444},
    'F': {'A': 0.2065, 'B': 0.2130, 'C': 0.1981, 'D': 0.2083, 'E': 0.3170, 'F': None,   'G': 1.0000},
    'G': {'A': None,   'B': None,   'C': None,   'D': None,   'E': None,   'F': None,   'G': None},
}

RAW_BIKE = {
    'A': {'A': None,   'B': 0.0000, 'C': 0.0315, 'D': 0.0497, 'E': 0.1184, 'F': 0.1636, 'G': 0.6972},
    'B': {'A': 0.0000, 'B': None,   'C': 0.0389, 'D': 0.0633, 'E': 0.1242, 'F': 0.1570, 'G': 0.6678},
    'C': {'A': 0.0315, 'B': 0.0389, 'C': None,   'D': 0.0366, 'E': 0.1139, 'F': 0.1596, 'G': 0.6713},
    'D': {'A': 0.0455, 'B': 0.0497, 'C': 0.0239, 'D': None,   'E': 0.1012, 'F': 0.1551, 'G': 0.6554},
    'E': {'A': 0.1336, 'B': 0.1313, 'C': 0.1210, 'D': 0.0966, 'E': None,   'F': 0.2512, 'G': 0.7403},
    'F': {'A': 0.1660, 'B': 0.1831, 'C': 0.1584, 'D': 0.1715, 'E': 0.2142, 'F': None,   'G': 0.8143},
    'G': {'A': None,   'B': None,   'C': None,   'D': None,   'E': None,   'F': None,   'G': None},
}

DATASETS = [
    ("Route Set 1  (RAW)", RAW, (231, 76, 60)),
    ("Route Set 2  (RAW_BIKE)", RAW_BIKE, (52, 152, 219)), 
]

# A vs B, give lower value, this is use to turn ASTP into TSP, suitable for MST later
def mst_lowerbound(data, rFrom, rTo):
    a = data[rFrom][rTo]
    b = data[rTo][rFrom]

    if a is None:
        return b
    if b is None:
        return a

    return min(a, b)

#to ensure MST don create loop by storing and remembering groups
class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}
        self.rank = {x: 0 for x in items}

    # use to find the ultimate root parent of that group
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    #check if will create loop, if not, then true
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        #if both shares the same parent, connecting both will create loop, return false
        if ra == rb:
            return False
        #change order to make sure small rank always connect to big ranks
        #for computational concern, doesn't really matter in our small dataset
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True

#this whole thing serves as the heuristic for A* search
#MST is essentially:
#   - make sure all route is at least vitised once
#   - finds the cheapest way to do that
def mst_weight(raw, node_set):
    node_set = list(node_set)
    if len(node_set) <= 1:
        return 0.0
    edges = []
    #as ASTP scenario, this breaks, cause we don't calculate B->A, but because how we use mst_lowerbound to force this into a TSP scenario, it works
    #the for loop essentially goes thru all nodes in A->G, get all the smaller value for each edge(route), for later MST purpose
    for i in range(len(node_set)):
        for j in range(i + 1, len(node_set)):
            #capture the nodes
            u, v = node_set[i], node_set[j]
            #see which route has smaller value, then add that route into edges
            w = mst_lowerbound(raw, u, v)
            edges.append((w, u, v))
    #sort it so the smallest value comes first, and will be the first one to be expanded
    edges.sort(key=lambda e: e[0])
    uf = UnionFind(node_set)
    total, used = 0.0, 0
    for w, u, v in edges:
        #if it doesn't create loop, add the edge(route) into the total cost of MST
        if uf.union(u, v):
            total += w
            used += 1
            if used == len(node_set) - 1:
                break
    if used != len(node_set) - 1:
        return float('inf')  
    return total


def a_star_search(raw, nodes=NODES, start=START, goal=GOAL):
    #gets the cheapest way to connect all the nodes together
    #our MST lowerbound heuristic doesn't count the cost to goal, but rather the cheapest way to connect all nodes
    def h(current, unvisited):
        return mst_weight(raw, set(unvisited) | {current})

    #used for tuples order
    counter = itertools.count()
    start_visited = frozenset([start])
    #gets the h(n) for starting node
    h0 = h(start, set(nodes) - {start})
    #add the first node into frontier
    frontier = [(h0, 0.0, next(counter), start, (start,), start_visited)]
    best_g = {(start, start_visited): 0.0}

    steps_log = []
    expanded = 0
    solution = None

    while frontier:
        #in the first run, it will extract the first(smallest) node out and remove it from the frontier, so all f, g, current, path, visited becomes the value of the first node
        f, g, _, current, path, visited = heapq.heappop(frontier)
        #check if this route is the cheapest for the same route, e.g: A-B-C-D = 10 | A-C-B-D = 15, it will skip the 15
        #ensuring when we check cheapest cost to visit D while had visited A, B, C; it would return 10
        if g > best_g.get((current, visited), float('inf')):
            continue
        #exploration count
        expanded += 1
        #logs the exploration data
        step = {
            'expanded': expanded, 'current': current, 'path': path,
            'g': g, 'h': f - g, 'f': f, 'visited': visited,
            'frontier_size': len(frontier), 'goal': False,
        }
        steps_log.append(step)

        #check if reach goal node and all node is visited
        if current == goal and len(visited) == len(nodes):
            step['goal'] = True
            solution = (path, g)
            break

        #python default function to go to next loop
        #this for loop is basically expand children function
        for nxt in nodes:
            #if visited, skip this node
            if nxt in visited:
                continue
            cost = raw[current][nxt]
            #if don't have cost, skip this node
            if cost is None:
                continue
            new_visited = visited | {nxt}
            if nxt == goal and len(new_visited) != len(nodes):
                continue 
            new_g = g + cost
            key = (nxt, new_visited)
            if new_g < best_g.get(key, float('inf')):
                best_g[key] = new_g
                remaining = set(nodes) - new_visited
                hh = h(nxt, remaining)
                heapq.heappush(frontier, (new_g + hh, new_g, next(counter),
                                           nxt, path + (nxt,), new_visited))
    return solution, steps_log




EDGE_DRAW_TIME = 0.22   # seconds to draw one edge, at 1x speed
HOLD_TIME = 0.14        # pause between expansion steps, at 1x speed


class PanelAnimator:
    def __init__(self, steps_log, solution):
        self.steps_log = steps_log
        self.solution_path, self.solution_cost = solution
        self.step_idx = 0
        first_path = steps_log[0]['path']
        self.display_path = (first_path[0],)
        self.pending_edges = list(zip(first_path, first_path[1:]))
        self.edge_progress = 0.0
        self.phase = 'drawing' if self.pending_edges else 'holding'
        self.hold_timer = HOLD_TIME
        self.finished = False

    def _advance_to_next_step(self):
        self.step_idx += 1
        if self.step_idx >= len(self.steps_log):
            self.finished = True
            self.phase = 'done'
            return
        new_path = self.steps_log[self.step_idx]['path']
        i = 0
        while (i < len(self.display_path) and i < len(new_path)
               and self.display_path[i] == new_path[i]):
            i += 1
        self.display_path = new_path[:i]
        self.pending_edges = list(zip(new_path[i - 1:], new_path[i:]))
        self.edge_progress = 0.0
        self.phase = 'drawing' if self.pending_edges else 'holding'
        self.hold_timer = HOLD_TIME

    def update(self, dt):
        if self.phase == 'done':
            return
        if self.phase == 'drawing':
            self.edge_progress += dt / EDGE_DRAW_TIME
            if self.edge_progress >= 1.0:
                u, v = self.pending_edges.pop(0)
                self.display_path = self.display_path + (v,)
                self.edge_progress = 0.0
                if not self.pending_edges:
                    self.phase = 'holding'
                    self.hold_timer = HOLD_TIME
        elif self.phase == 'holding':
            self.hold_timer -= dt
            if self.hold_timer <= 0:
                self._advance_to_next_step()

    def get_render_info(self):
        static_edges = list(zip(self.display_path, self.display_path[1:]))
        animating_edge = None
        if self.phase == 'drawing' and self.pending_edges:
            u, v = self.pending_edges[0]
            animating_edge = (u, v, min(self.edge_progress, 1.0))
        step = self.steps_log[min(self.step_idx, len(self.steps_log) - 1)]
        return {
            'static_edges': static_edges,
            'animating_edge': animating_edge,
            'visited_nodes': set(self.display_path),
            'current_node': self.display_path[-1],
            'step_info': step,
            'fully_solved': self.phase == 'done',
        }


# (col, row) placeholder grid coordinates -- purely visual, not to scale
# with the actual matrix costs (the source data has no coordinates).
NODE_GRID = {
    'A': (0, 2),
    'B': (2, 0),
    'C': (2, 4),
    'D': (4, 2),
    'E': (5, 0),
    'F': (5, 4),
    'G': (7, 2),
}
GRID_COLS, GRID_ROWS = 7, 4  # max col index, max row index

NODE_RADIUS = 22
MAP_PADDING = 55

# ---- colors -----------------------------------------------------------
BG = (16, 17, 22)
GRID_LINE = (36, 38, 48)
PANEL_BG = (22, 24, 31)
PANEL_BORDER = (55, 58, 72)
TEXT = (228, 229, 235)
DIM_TEXT = (140, 142, 155)
UNVISITED_FILL = (52, 54, 66)
TIP_RING = (255, 209, 102)
SOLVED_COLOR = (46, 204, 113)
NODE_BORDER = (10, 10, 14)


def lerp(a, b, t):
    return a + (b - a) * t


def lerp_point(p1, p2, t):
    return (lerp(p1[0], p2[0], t), lerp(p1[1], p2[1], t))


def trimmed_endpoints(p1, p2, r):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return p1, p2
    ux, uy = dx / dist, dy / dist
    a = (p1[0] + ux * r, p1[1] + uy * r)
    b = (p2[0] - ux * r, p2[1] - uy * r)
    return a, b


def draw_arrow(surface, color, start, end, width=3, head_len=11, head_angle=24):
    dx, dy = end[0] - start[0], end[1] - start[1]
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return
    pygame.draw.line(surface, color, start, end, width)
    ux, uy = dx / dist, dy / dist
    ang = math.radians(head_angle)
    for sign in (1, -1):
        hx = end[0] - head_len * (ux * math.cos(ang) - sign * uy * math.sin(ang))
        hy = end[1] - head_len * (uy * math.cos(ang) + sign * ux * math.sin(ang))
        pygame.draw.line(surface, color, end, (hx, hy), width)


class Panel:
    """One dataset's map + info HUD, rendered inside a fixed screen rect."""

    def __init__(self, rect, title, raw, accent):
        self.rect = rect          # pygame.Rect for the whole panel
        self.title = title
        self.raw = raw
        self.accent = accent
        self.solution, self.steps_log = a_star_search(raw)
        self.solution_path, self.solution_cost = self.solution
        self.animator = PanelAnimator(self.steps_log, self.solution)

        map_h = int(rect.height * 0.62)
        self.map_rect = pygame.Rect(rect.x, rect.y + 34, rect.width, map_h)
        self.info_rect = pygame.Rect(rect.x, self.map_rect.bottom + 8,
                                      rect.width, rect.bottom - self.map_rect.bottom - 8)
        self.node_px = self._compute_node_pixels()

    def _compute_node_pixels(self):
        usable_w = self.map_rect.width - 2 * MAP_PADDING
        usable_h = self.map_rect.height - 2 * MAP_PADDING
        cell_w = usable_w / GRID_COLS
        cell_h = usable_h / GRID_ROWS
        px = {}
        for n, (c, r) in NODE_GRID.items():
            x = self.map_rect.x + MAP_PADDING + c * cell_w
            y = self.map_rect.y + MAP_PADDING + r * cell_h
            px[n] = (x, y)
        self._cell_w, self._cell_h = cell_w, cell_h
        return px

    def restart(self):
        self.animator = PanelAnimator(self.steps_log, self.solution)

    def update(self, dt):
        self.animator.update(dt)

    # ---- drawing --------------------------------------------------
    def draw_grid_lines(self, surface):
        for c in range(GRID_COLS + 1):
            x = self.map_rect.x + MAP_PADDING + c * self._cell_w
            pygame.draw.line(surface, GRID_LINE, (x, self.map_rect.y + MAP_PADDING),
                              (x, self.map_rect.bottom - MAP_PADDING), 1)
        for r in range(GRID_ROWS + 1):
            y = self.map_rect.y + MAP_PADDING + r * self._cell_h
            pygame.draw.line(surface, GRID_LINE, (self.map_rect.x + MAP_PADDING, y),
                              (self.map_rect.right - MAP_PADDING, y), 1)

    def draw(self, surface, fonts):
        pygame.draw.rect(surface, PANEL_BG, self.rect, border_radius=8)
        pygame.draw.rect(surface, PANEL_BORDER, self.rect, width=1, border_radius=8)

        title_surf = fonts['panel_title'].render(self.title, True, self.accent)
        surface.blit(title_surf, (self.rect.x + 14, self.rect.y + 8))

        self.draw_grid_lines(surface)

        info = self.animator.get_render_info()
        solved = info['fully_solved']
        line_color = SOLVED_COLOR if solved else self.accent

        # static (already-drawn) edges
        for (u, v) in info['static_edges']:
            p1, p2 = trimmed_endpoints(self.node_px[u], self.node_px[v], NODE_RADIUS)
            draw_arrow(surface, line_color, p1, p2, width=3)
            wtxt = fonts['edge'].render(f"{self.raw[u][v]:.3f}", True, DIM_TEXT)
            mid = lerp_point(p1, p2, 0.5)
            surface.blit(wtxt, (mid[0] - wtxt.get_width() / 2, mid[1] - wtxt.get_height() / 2))

        # currently animating edge
        if info['animating_edge'] is not None:
            u, v, t = info['animating_edge']
            p1, p2 = trimmed_endpoints(self.node_px[u], self.node_px[v], NODE_RADIUS)
            tip = lerp_point(p1, p2, t)
            draw_arrow(surface, self.accent, p1, tip, width=4)
            pygame.draw.circle(surface, TIP_RING, (int(tip[0]), int(tip[1])), 6)

        # nodes
        for n in NODES:
            x, y = self.node_px[n]
            visited = n in info['visited_nodes']
            if solved:
                fill = SOLVED_COLOR
            elif visited:
                fill = self.accent
            else:
                fill = UNVISITED_FILL
            pygame.draw.circle(surface, fill, (int(x), int(y)), NODE_RADIUS)
            pygame.draw.circle(surface, NODE_BORDER, (int(x), int(y)), NODE_RADIUS, 2)
            if n == info['current_node'] and not solved:
                pygame.draw.circle(surface, TIP_RING, (int(x), int(y)), NODE_RADIUS + 4, 2)
            label = fonts['node'].render(n, True, TEXT)
            surface.blit(label, (x - label.get_width() / 2, y - label.get_height() / 2))

        # info HUD
        pygame.draw.rect(surface, (0, 0, 0, 0), self.info_rect)  # placeholder, no fill
        step = info['step_info']
        lines = []
        if not solved:
            lines.append(f"Expanding state #{step['expanded']}   (frontier: {step['frontier_size']})")
            lines.append(f"Path: {' -> '.join(step['path'])}")
            lines.append(f"g={step['g']:.4f}   h={step['h']:.4f}   f={step['f']:.4f}")
        else:
            lines.append(f"SOLVED -- optimal path found")
            lines.append(f"Path: {' -> '.join(self.solution_path)}")
            lines.append(f"Total cost: {self.solution_cost:.4f}   (states expanded: {len(self.steps_log)})")
        for i, line in enumerate(lines):
            surf = fonts['info'].render(line, True, TEXT if i else TEXT)
            surface.blit(surf, (self.info_rect.x + 4, self.info_rect.y + 4 + i * 20))


# =========================================================================
# 6. Main loop
# =========================================================================
def build_fonts():
    return {
        'title': pygame.font.SysFont("couriernew,monospace", 26, bold=True),
        'sub': pygame.font.SysFont("couriernew,monospace", 14),
        'panel_title': pygame.font.SysFont("couriernew,monospace", 18, bold=True),
        'node': pygame.font.SysFont("couriernew,monospace", 17, bold=True),
        'info': pygame.font.SysFont("couriernew,monospace", 15),
        'edge': pygame.font.SysFont("couriernew,monospace", 12),
        'footer': pygame.font.SysFont("couriernew,monospace", 16, bold=True),
    }


def main(headless_max_frames=None):
    pygame.init()
    pygame.display.set_caption("A* Hamiltonian Path -- MST Heuristic (pygame)")
    W, H = 1500, 860
    screen = pygame.display.set_mode((W, H))
    clock = pygame.time.Clock()
    fonts = build_fonts()

    margin = 16
    panel_w = (W - 3 * margin) // 2
    panel_h = H - 130
    panels = [
        Panel(pygame.Rect(margin + i * (panel_w + margin), 96, panel_w, panel_h),
              title, raw, accent)
        for i, (title, raw, accent) in enumerate(DATASETS)
    ]

    speed = 1.0
    paused = False
    running = True
    frame_count = 0

    while running:
        dt = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    for p in panels:
                        p.restart()
                elif event.key == pygame.K_UP:
                    speed = min(speed * 1.5, 8.0)
                elif event.key == pygame.K_DOWN:
                    speed = max(speed / 1.5, 0.1)

        if not paused:
            for p in panels:
                p.update(dt * speed)

        screen.fill(BG)

        title_surf = fonts['title'].render(
            "A* Search -- Minimum-Cost Hamiltonian Path (MST Heuristic)", True, TEXT)
        screen.blit(title_surf, (margin, 12))
        sub_surf = fonts['sub'].render(
            f"SPACE pause/resume | R restart | UP/DOWN speed ({speed:.2f}x) | ESC quit",
            True, DIM_TEXT)
        screen.blit(sub_surf, (margin, 44))

        for p in panels:
            p.draw(screen, fonts)

        # comparison footer once both panels are solved
        if all(p.animator.get_render_info()['fully_solved'] for p in panels):
            c0 = panels[0].solution_cost
            c1 = panels[1].solution_cost
            if c0 < c1:
                cheaper, diff = panels[0].title, c1 - c0
            elif c1 < c0:
                cheaper, diff = panels[1].title, c0 - c1
            else:
                cheaper, diff = None, 0.0
            if cheaper:
                msg = (f"{panels[0].title}: {c0:.4f}   vs   {panels[1].title}: {c1:.4f}"
                       f"   ->  {cheaper} is cheaper by {diff:.4f}")
            else:
                msg = f"Both routes cost the same: {c0:.4f}"
            footer_surf = fonts['footer'].render(msg, True, SOLVED_COLOR)
            screen.blit(footer_surf, (margin, H - 26))

        pygame.display.flip()

        frame_count += 1
        if headless_max_frames is not None and frame_count >= headless_max_frames:
            running = False

    pygame.quit()


if __name__ == '__main__':
    main()