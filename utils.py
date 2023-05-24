import networkx as nx
from typing import Dict, List, Tuple


def json2graph(world_json: Dict) -> nx.graph:
    """
    Creates a NetworkX graph from JSON.
    """
    # extract grid size (n,m) and pits coordinates from JSON
    n, m = world_json['size']
    pits = world_json['pits']

    # construct full graph
    G = nx.grid_2d_graph(n=n, m=m)

    # remove the nodes corresponding to pits
    for pit in pits:
        G.remove_node(tuple(pit))

    # add edges between adjacent nodes that are not blocked by bits
    for node in G.nodes:
        adjacent_neigbours = [
            (node[0] - 1, node[1]), 
            (node[0] + 1, node[1]),
            (node[0], node[1] - 1), 
            (node[0], node[1] + 1)
        ]
        for neighbor in adjacent_neigbours:
            if neighbor in G.nodes and neighbor not in pits:
                G.add_edge(node, neighbor)

    return G
    

class Agent:
    """
    Represents path (sequence of coordinates) as sequence of hunter actions.
    """
    def __init__(self, 
                 hunter_location: Tuple[int], 
                 gold_location: Tuple[int], 
                 wumpus_location: Tuple[int], 
                 direction: str = 'N',
                 has_grabbed: bool = False,
                 has_shot: bool = False):
        
        self.x = hunter_location[0]
        self.y = hunter_location[1]
        self.gold_location = gold_location
        self.wumpus_location = wumpus_location
        self.direction = direction
        self.has_grabbed = has_grabbed  
        self.has_shot = has_shot
        self.actions = []
        self.turns = {
            'N': {'E': 'Right', 'W': 'Left', 'S': 'Rotate', 'N': None},
            'E': {'N': 'Left', 'S': 'Right', 'W': 'Rotate', 'E': None},
            'S': {'W': 'Right', 'E': 'Left', 'N': 'Rotate', 'S': None},
            'W': {'S': 'Left', 'N': 'Right', 'E': 'Rotate', 'W': None}
        }

    def navigate(self, path: List[Tuple[int]]) -> List[str]:
        """
        Navigates agent through path.
        """
        for step in path[1:]:
            self._move_to(step[0], step[1])
        self.actions.append('Climb')
        return self.actions

    def _move_to(self, x: int, y: int) -> List[str]:
        """
        Performs actions based on the next location (x,y).
        """
        if x == self.x + 1:
            self._turn_to('E')
            self._shoot_if_wumpus_ahead(x=x, y=y)
            self._move_forward_east()
            self._grab_if_gold(x=self.x, y=self.y)

        elif x == self.x - 1:
            self._turn_to('W')
            self._shoot_if_wumpus_ahead(x=x, y=y)
            self._move_forward_west()
            self._grab_if_gold(x=self.x, y=self.y)
        
        elif y == self.y + 1:
            self._turn_to('N')
            self._shoot_if_wumpus_ahead(x=x, y=y)
            self._move_forward_north()
            self._grab_if_gold(x=self.x, y=self.y)

        elif y == self.y - 1:
            self._turn_to('S')
            self._shoot_if_wumpus_ahead(x=x, y=y)
            self._move_forward_south()
            self._grab_if_gold(x=self.x, y=self.y)

        else:
            raise ValueError('Next coordinate must be different.')

        return self.actions

    def _turn_to(self, direction: str):
        turn = self.turns[self.direction][direction]
        if turn:
            if turn == 'Right':
                self._turn_right()
            elif turn == 'Left':
                self._turn_left()
            else:
                self._turn_around()
        self.direction = direction

    def _turn_left(self):
        self.actions.append('Left')

    def _turn_right(self):
        self.actions.append('Right')

    def _turn_around(self):
        self.actions.append('Right')
        self.actions.append('Right')

    def _grab_if_gold(self, x: int, y: int):
        if (x, y) == self.gold_location and not self.has_grabbed:
            self.actions.append('Grab')
            self.has_grabbed = True
    
    def _shoot_if_wumpus_ahead(self, x: int, y: int):
        if (x, y) == self.wumpus_location and not self.has_shot:
            self.actions.append('Shoot')
            self.has_shot = True

    def _move_forward_east(self):
        self.actions.append('Move')
        self.x += 1

    def _move_forward_west(self):
        self.actions.append('Move')
        self.x -= 1

    def _move_forward_north(self):
        self.actions.append('Move')
        self.y += 1

    def _move_forward_south(self):
        self.actions.append('Move')
        self.y -= 1


def cost_function(
        path: List[Tuple[int]], 
        has_wumpus: bool,
        hunter_loc: Tuple[int], 
        gold_loc: Tuple[int], 
        wumpus_loc: Tuple[int]
    ) -> float:
    """
    Helper function to calculate the cost of a path considering Wumpus.

    # Decide whether we are going through Wumpus or avoiding it (there might be 
    # a situation where avoiding Wumpus is more optimal, even if the way 
    # through Wumpus is shorter, e.g. wumpus_world_custom_02.json).

    Create an agent with the given path and simulate its actions to get 
    the sequence of moves. If the path passes through the Wumpus, add 9 
    to the cost (not 10 because 'Shoot' is calculated as 1).
    """
    stl = Agent(
        hunter_location=hunter_loc, 
        gold_location=gold_loc,
        wumpus_location=wumpus_loc
    )
    toyld = stl.navigate(path)

    if has_wumpus:
        cost = len(toyld) + 9
    else:
        cost = len(toyld)

    return cost
