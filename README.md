# City-Search
searching for destination city in a randomized graph using different searching algorithms

# Description
This program creates a random graph on a 100X100 grid for any number of instances specified. The nodes are city locations (25 largest cities in Canada, according to Wikipedia), and the edges are roads. For each city, 1-4 edges are randomly generated between it and its 5 nearest neighbours. A start node, and goal node are randomly selected. Then uninformed (DFS, BFS, iterative deepening) and informed searching algorithms (best-first search, astar search) are performed. Aggregate results are computed and analyzed, to determine which searching algorithm performs best.

# Sample Output
Here is an example run of iterative deepening search. The algorithm calls depth-limited search, which runs depth-first search up to the depth specified as one of the parameters passed in. When the depth is reached, the search is terminated for that branch, and the search backs up and tries the next child if there is one available. 
The initial configuration of the randomly generated graph has Vaughan as the start city, and Calgary as the goal city, with 3 edges separating the two. 
![Screenshot](../master/iterative0.png)

After one iteration of the search, the explored nodes are just Vaughan's neighbours, since the depth of the search increments starting from 1.
![Screenshot](../master/iterative1.png)

On the second iteration, the algorithm explores up to 2 nodes distance from the root, before getting cut off. The graph shows the expanding search tree in red. 
![Screenshot](../master/iterative2.png)

Finally, on the third iteration, the agent's hard work pays off. Unfortunately, it finds Calgary as one of the last cities at depth 3 that it checks, so many cities are unnecessarily checked. This is a problem that is remedied by using an informed searching algorithm.
![Screenshot](../master/iterative3.png)

The path returned is Vaughan, London, Edmonton, Calgary. 

To do: Add more networkx graph examples, and data results

