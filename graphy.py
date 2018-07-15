import networkx
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import random
import math
from collections import deque
import sys
import heapq
import statistics
import timeit

draw = False

# depending on if the problem is defined as greedy or astar, produces diff results according to each heuristic
def best_first_search(problem, greedy=True, manhattan=False):
	if greedy:
		problem.greedy = True
	if manhattan:
		problem.manhattan = True
	#print(f'greedy: {greedy}')
	#	print(f'manhattan: {manhattan}')

	node = Node(problem.init)
	node.heuristic(problem)
	frontier = PriorityQueue(node)
	frontier.append(node)
	while frontier:
		#print(len(frontier))
		problem.space(len(problem.explored_cities) + len(frontier))
		node = frontier.pop()
		#	if not node: break
		if problem.goal_test(node.state):
			problem.solution_stats(node)
			return node
		problem.explored_cities.add(node.state)
		for action in problem.actions(node.state):
			problem.time()
			child = node.child_node(problem, action)
			child.heuristic(problem)
			if child.state not in problem.explored_cities and child not in frontier:
				frontier.append(child)
			elif child in frontier:
				incumbent = frontier[child]
				if child < incumbent:
					del frontier[incumbent]
					frontier.append(child)


def breadth_first_graph_search(problem):
	# BFS graph search - frontier is leaves of BFS tree.
	node = Node(problem.init)
	if problem.goal_test(node.state):
		problem.solution_stats(node)
		return node
	frontier = deque([node])
	while frontier:
		problem.space(len(problem.explored_cities) + len(frontier))
		node = frontier.popleft()
		problem.h(node)
		problem.explored_cities.add(node.state)
		children = node.expand(problem)
		for child in children:
			# increment number of nodes visited one by one until reaching goal state
			problem.time()
			if child.state not in problem.explored_cities and child not in frontier:
				if problem.goal_test(child.state):
					problem.solution_stats(child)
					return child
				frontier.append(child)

def depth_first_graph_search(problem):
	frontier = [(Node(problem.init))]  # Stack
	while frontier:
		problem.space(len(problem.explored_cities) + len(frontier))
		node = frontier.pop()
		if problem.goal_test(node.state):
			problem.solution_stats(node)
			return node
		problem.explored_cities.add(node.state)
		# add leaves to frontier if not yet explored, and not already in frontier
		for child in node.expand(problem):
			problem.time()
			if child.state not in problem.explored_cities and child not in frontier:
				frontier.append(child)

def depth_limited_search(problem, depth):
	frontier = [(Node(problem.init))]  # Stack
	max_depth_reached = 0
	while frontier:
		problem.space(len(problem.explored_cities) + len(frontier))
		node = frontier.pop()
		if problem.goal_test(node.state):
			problem.solution_stats(node)
			return False
		problem.explored_cities.add(node.state)
		if node.depth > max_depth_reached:
			max_depth_reached = node.depth
		# if node is at max depth for iteration we don't add its children to the frontier
		if node.depth != depth:
			# add leaves to frontier if not yet explored, and not already in frontier
			for child in node.expand(problem):
				problem.time()
				if child.state not in problem.explored_cities and child not in frontier:
					frontier.append(child)
	# if max depth reached is smaller than possible depth, and we didn't find a solution, then the goal must be disconnected and we should stop searching
	if max_depth_reached < depth:
		return False
	return True

def iterative_deepening(problem):
	for depth in range(sys.maxsize):
		problem.explored_cities = set()
		cutoff = depth_limited_search(problem, depth)
		#problem.draw(city_graph, cities)
		if not cutoff:
			return


class Node:
	"""A node in a search tree. Contains a pointer to the parent (the node
	that this is a successor of) and to the actual state for this node. Note
	that if a state is arrived at by two paths, then there are two nodes with
	the same state.  Also includes the action that got us to this state, and
	the total path_cost (also known as g) to reach the node.  Other functions
	may add an f and h value; see best_first_graph_search and astar_search for
	an explanation of how the f and h values are handled. You will not need to
	subclass this class."""

	def __init__(self, state, parent=None, action=None, path_cost=0):
		"""Create a search tree Node, derived from a parent by an action."""
		self.state = state
		self.parent = parent
		self.action = action
		self.path_cost = path_cost
		self.depth = 0
		self.f = 0
		if parent:
			self.depth = parent.depth + 1

	def __repr__(self):
		return "<Node {}>".format(self.state)

	# modify for greedy, astar searches - 3 uninformed don't need comparison
	def __lt__(self, node):
		return self.f < node.f

	# if greedy, set f = h
	# if astar, set f as path cost + h
	def heuristic(self, problem):
		if problem.greedy:
			self.f = problem.h(self)
		else:
			self.f = self.path_cost + problem.h(self)

	def expand(self, problem):
		"""List the nodes reachable in one step from this node."""
		# returns list of nodes created out of all the neighbours in the graph
		return [self.child_node(problem, action)
				for action in problem.actions(self.state)]

	def child_node(self, problem, action):
		"""[Figure 3.10]"""
		# parent is self, action is parent's action to get child (ie. action = child.state)
		state = problem.result(self.state, action)
		child = Node(state, self, action)
		child.path_cost = problem.path_cost(self.path_cost, self.state, action, child.state)
		return child

	def path(self):
		solution_path = []
		node = self
		while node:
			solution_path.insert(0, node.state)
			node = node.parent
		return solution_path

	# We want for a queue of nodes in breadth_first_graph_search or
	# astar_search to have no duplicated states, so we treat nodes
	# with the same state as equal. [Problem: this may not be what you
	# want in other contexts.]

	def __eq__(self, other):
		return isinstance(other, Node) and self.state == other.state

	def __hash__(self):
		return hash(self.state)

''''''
class PriorityQueue:
	def __init__(self, f):
		self.heap = []
		self.f = f

	def append(self, item):
		"""Insert item at its correct position."""
		heapq.heappush(self.heap, item)

	def pop(self):
		if self.heap:
			return heapq.heappop(self.heap)

		#else:
		#	raise Exception("queue empty bruv")

	def __contains__(self, item):
		"""Return True if item in PriorityQueue."""
		return (item) in self.heap

	def __getitem__(self, key):
		for item in self.heap:
			if item == key:
				return item

	def __len__(self):
		return len(self.heap)

	def __delitem__(self, key):
		"""Delete the first occurrence of key."""
		self.heap.remove(key)
		heapq.heapify(self.heap)


class GraphProblem():
	"""The problem of searching a graph from one node to another."""

	def __init__(self, initial, goal, graph, greedy=False, manhattan=False):
		self.init = initial
		self.goal = goal
		self.graph = graph
		self.explored_cities = set()
		self.greedy = greedy
		self.manhattan = manhattan
		self.space_complexity = 0
		self.time_complexity = 0
		self.soln_path_length = 0
		self.soln_path_depth = 0
		self.solved = False
		self.branches = 0
		self.soln_path = None

	def goal_test(self, state):
		return state == self.goal

	def actions(self, A):
		"""The actions at a graph node are just its neighbors."""
		return list(self.graph.graph_dict[A][1].keys())

	def result(self, state, action):
		"""The result of going to a neighbor is just that neighbor."""
		return action

	def path_cost(self, cost_so_far, A, action, B):
		#print(f'segment length from {A} to {B}: {self.graph.graph_dict[A][1][B]}')
		return cost_so_far + self.graph.graph_dict[A][1][B]

	def find_min_edge(self):
		"""Find minimum value of edges."""
		m = infinity
		for d in self.graph.graph_dict.values():
			local_min = min(d.values())
			m = min(m, local_min)

		return m

	def h(self, node):
		# euclidean heuristic if not manhattan

		nx = city_graph.graph_dict[node.state][0][0]
		ny = city_graph.graph_dict[node.state][0][1]
		gx = city_graph.graph_dict[goal][0][0]
		gy = city_graph.graph_dict[goal][0][1]
		if not self.manhattan:
			return math.sqrt((nx - gx) ** 2 + (ny - gy) ** 2)
		else:
			x = math.fabs(nx - gx) + math.fabs(ny - gy)
			return math.fabs(nx - gx) + math.fabs(ny - gy)


	def draw(self, city_graph, cities):
		# returns a networkx representation of the search result
		g = networkx.Graph()

		# add nodes
		for city in cities:
			x = city_graph.graph_dict[city][0][0]
			y = city_graph.graph_dict[city][0][1]
			g.add_node(city, pos=(x, y))

		# add edges
		for city1 in cities:
			for city2 in list(city_graph.graph_dict[city1][1].keys()):
				g.add_edge(city1, city2)

		# add colour for which nodes explored as well as goal node
		colour_map = []
		for node in g:
			if node in problem.explored_cities:
				if node == self.init:
					colour_map.append('yellow')
				else:
					colour_map.append('red')
			elif node == self.goal:
				colour_map.append('blue')
			else:
				colour_map.append('green')

		# add position to nodes on graph
		pos = networkx.get_node_attributes(g, 'pos')
		networkx.draw(g, pos, node_color=colour_map, with_labels=True)
		plt.show()

	# updates space complexity according to maximum size of frontier
	def space(self, nodes_in_memory):
		if nodes_in_memory > self.space_complexity:
			self.space_complexity = nodes_in_memory
			#print(f"size of frontier: {self.space_complexity}")
			return self.space_complexity

	# updates time complexity according to number of nodes visited
	# in BFS its number of nodes goal tested
	def time(self):
		self.time_complexity += 1
		return self.time_complexity

	def solution_stats(self, goal_node):
		#print(f"goal path: {goal_node.path()}")
		#print(f"goal state: {goal_node}")
		self.solved = True
		self.soln_path_length = goal_node.path_cost
		self.soln_path_depth = len(goal_node.path())
		self.soln_path = goal_node.path()

	def num_branches(self, cities):
		for city in cities:
			self.branches = self.branches + len((self.graph.graph_dict[city][1]))
		return self.branches

class Graph:

	def __init__(self, graph_dict=None):
		self.graph_dict = graph_dict or {}

	def connect(self, A, B, distance):
		"""Add a link from A and B of given distance, and also add the inverse link"""
		self.connect1(A, B, distance)
		self.connect1(B, A, distance)

	def connect1(self, A, B, distance):
		"""Add a link from A to B of given distance, in one direction only."""
		self.graph_dict[A][1][B] = distance

	def gen_nodes(self, cities):
		# generate the cities and their locations for each instance
		test_coords = []
		for city in cities:
			x = random.randint(0, 100)
			y = random.randint(0, 100)
			# make sure only positive distance cities
			while (x, y) in test_coords:
				x = random.randint(0, 100)
				y = random.randint(0, 100)
			test_coords.append((x, y))
			city_graph.graph_dict[city] = [(x, y), {}]

	def populate(self, cities):
		# populates the graph with edges: between 1 to 4 of the closest 5 neighbours selected

		# brute force nearest neighbours, take 5 closest neighbours
		for a in list(self.graph_dict.keys()):
			neighbours_distance = []
			for b in list(self.graph_dict.keys()):
				if a != b:
					ax = self.graph_dict[a][0][0]
					ay = self.graph_dict[a][0][1]
					bx = self.graph_dict[b][0][0]
					by = self.graph_dict[b][0][1]
					neighbours_distance.append((a, b, math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)))
			neighbours_distance.sort(key=lambda x: x[2])
			neighbours_distance = neighbours_distance[:5]

			# take 1 to 4 randomly and connect them
			temp = list(range(5))
			random.shuffle(temp)
			city1 = neighbours_distance[0][0]
			j = random.randint(1, 4)
			for i in range(j):
				index = temp[i]
				city2 = neighbours_distance[index][1]
				distance = neighbours_distance[index][2]
				adj = self.graph_dict[city1][1]
				if city2 not in adj.keys():
					self.connect(city1, city2, distance)


if __name__ == '__main__':
	draw = True

	# reading in city names from file
	with open('cities2.txt', mode='r') as my_file:
		lines = my_file.readlines()
		cities = [city_name.rstrip() for city_name in lines]

	space = []
	time = []
	path_length = []
	solution_depth =  []
	solved = []
	runtime = []
	branches = []
	function_dict = {'bfs': [breadth_first_graph_search], 'dfs': [depth_first_graph_search],
					 'iterative': [iterative_deepening], 'greedy euclidean': [best_first_search],
					 'greedy manhattan': [best_first_search],
					 'astar euclidean': [best_first_search],
					 'astar manhattan': [best_first_search]}
	parameter_dict = {'greedy manhattan': [True, True], 'astar euclidean': [False, False],
					  'astar manhattan': [False, True]}

	# generate 100 problem instances
	for i in range(0,5):
		# create random instance of graph
		city_graph = Graph()
		city_graph.gen_nodes(cities)
		city_graph.populate(cities)

		# generate problem
		init = random.choice(cities)
		goal = random.choice(cities)
		while init == goal:
			goal = random.choice(cities)

		# solve problem using different search strategies
		for function in function_dict:
			problem = GraphProblem(init, goal, city_graph)
			#print(function)
			start = timeit.default_timer()
			if function in parameter_dict:
				function_dict[function][0](problem, parameter_dict[function][0], parameter_dict[function][1])
			else:
				function_dict[function][0](problem)
				print(f'{function} solution path: {problem.soln_path}')
			stop = timeit.default_timer()
			if draw:
				problem.draw(city_graph, cities)

			#add the data to the relevant lists
			space.append(problem.space_complexity)
			time.append(problem.time_complexity)
			path_length.append(problem.soln_path_length)
			solved.append(problem.solved)
			runtime.append(stop - start)
			solution_depth.append(problem.soln_path_depth)
		branches.append(problem.num_branches(cities))

	# parse the lists to get the averages
	function_list = list(function_dict.keys())
	for function in function_list:
		function_space = space[function_list.index(function)::7]
		function_time = time[function_list.index(function)::7]
		function_path_length = path_length[function_list.index(function)::7]
		function_runtime = runtime[function_list.index(function)::7]
		function_dict[function].append(statistics.mean(function_space))
		function_dict[function].append(statistics.mean(function_time))
		function_dict[function].append(statistics.mean(function_path_length))
		function_dict[function].append(statistics.mean(function_runtime))
		print(f'{function} average solution depth: {statistics.mean(solution_depth[function_list.index(function)::7])}')
		print(f'{function}: {function_dict[function]}')

	'''
	print(f'space: {space}')
	print(f'time: {time}')
	print(f'path length: {path_length}')
	print(f'solved: {solved}')
	print(f'runtime: {runtime}')
	print(f'branches: {branches}')
	'''


	print(f'avg space complexity: {statistics.mean(space)}')
	print(f'avg time complexity: {statistics.mean(time)}')
	print(f'avg path length: {statistics.mean(path_length)}')
	print(f'number of problems solved: {len([x for x in solved if x == True])}')
	print(f'avg branch factor: {statistics.mean(branches)/26}')

	#print(f"average branch factor over {j} instances: {total_branches/(j*26)}")
