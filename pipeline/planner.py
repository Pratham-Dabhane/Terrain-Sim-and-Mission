"""
Path Planning Module
A* pathfinding algorithm for terrain navigation.
"""

import numpy as np
import heapq
import logging
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def find_path(cost_grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Find optimal path through terrain using A* algorithm.
    
    Args:
        cost_grid (np.ndarray): Cost map where higher values are more expensive to traverse
        start (Tuple[int, int]): Starting position (row, col)
        goal (Tuple[int, int]): Goal position (row, col)
        
    Returns:
        List[Tuple[int, int]]: Path from start to goal as list of (row, col) coordinates
                               Empty list if no path found
    """
    h, w = cost_grid.shape
    
    # Validate start and goal positions
    if not (0 <= start[0] < h and 0 <= start[1] < w):
        logger.error(f"Start position {start} is out of bounds (grid shape: {cost_grid.shape})")
        return []
    
    if not (0 <= goal[0] < h and 0 <= goal[1] < w):
        logger.error(f"Goal position {goal} is out of bounds (grid shape: {cost_grid.shape})")
        return []
    
    # Priority queue: (f_score, position)
    open_set = [(0, start)]
    
    # Track path reconstruction
    came_from = {}
    
    # Cost from start to each position
    g_score = {start: 0}
    
    # 4-directional movement (up, down, left, right)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    logger.info(f"Starting A* pathfinding from {start} to {goal}")
    
    nodes_explored = 0
    
    while open_set:
        # Get position with lowest f_score
        _, current = heapq.heappop(open_set)
        nodes_explored += 1
        
        # Check if we reached the goal
        if current == goal:
            # Reconstruct path
            path = [goal]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            
            path.reverse()  # Reverse to get start-to-goal order
            
            # Calculate path statistics
            total_cost = sum(cost_grid[pos[0], pos[1]] for pos in path)
            
            logger.info(f"✓ Path found: {len(path)} waypoints, explored {nodes_explored} nodes, "
                       f"total cost: {total_cost:.2f}")
            return path
        
        # Explore neighbors
        for dx, dy in directions:
            nx, ny = current[0] + dx, current[1] + dy
            
            # Check bounds
            if 0 <= nx < h and 0 <= ny < w:
                # Calculate movement cost (base cost + terrain cost)
                # Higher cost_grid values mean more expensive terrain
                movement_cost = 1.0 + cost_grid[nx, ny]  # Base movement + terrain penalty
                new_cost = g_score[current] + movement_cost
                
                # If this path to neighbor is better than previous best
                if (nx, ny) not in g_score or new_cost < g_score[(nx, ny)]:
                    # Update cost and path
                    g_score[(nx, ny)] = new_cost
                    
                    # Heuristic: Manhattan distance to goal
                    heuristic = abs(goal[0] - nx) + abs(goal[1] - ny)
                    f_score = new_cost + heuristic
                    
                    # Add to open set
                    heapq.heappush(open_set, (f_score, (nx, ny)))
                    came_from[(nx, ny)] = current
    
    logger.warning(f"No path found from {start} to {goal} after exploring {nodes_explored} nodes")
    return []  # No path found


def calculate_path_statistics(path: List[Tuple[int, int]], cost_grid: np.ndarray, 
                              heightmap: np.ndarray) -> dict:
    """
    Calculate detailed statistics about a path.
    
    Args:
        path: List of (row, col) waypoints
        cost_grid: Cost map used for pathfinding
        heightmap: Terrain heightmap
        
    Returns:
        dict: Path statistics including cost, elevation changes, etc.
    """
    if not path or len(path) < 2:
        return {
            'valid': False,
            'waypoints': 0,
            'total_cost': 0,
            'elevation_gain': 0,
            'elevation_loss': 0
        }
    
    # Calculate costs along path
    costs = [cost_grid[pos[0], pos[1]] for pos in path]
    total_cost = sum(costs)
    mean_cost = np.mean(costs)
    max_cost = max(costs)
    
    # Calculate elevation changes
    elevations = [heightmap[pos[0], pos[1]] for pos in path]
    elevation_changes = [elevations[i+1] - elevations[i] for i in range(len(elevations)-1)]
    
    elevation_gain = sum(change for change in elevation_changes if change > 0)
    elevation_loss = abs(sum(change for change in elevation_changes if change < 0))
    
    # Calculate straight-line distance vs actual path length
    start, goal = path[0], path[-1]
    straight_line_dist = np.sqrt((goal[0] - start[0])**2 + (goal[1] - start[1])**2)
    path_efficiency = straight_line_dist / len(path) if len(path) > 0 else 0
    
    stats = {
        'valid': True,
        'waypoints': len(path),
        'total_cost': float(total_cost),
        'mean_cost': float(mean_cost),
        'max_cost': float(max_cost),
        'elevation_gain': float(elevation_gain),
        'elevation_loss': float(elevation_loss),
        'start_elevation': float(elevations[0]),
        'goal_elevation': float(elevations[-1]),
        'straight_line_distance': float(straight_line_dist),
        'path_efficiency': float(path_efficiency),
        'start': start,
        'goal': goal
    }
    
    logger.debug(f"Path stats: {len(path)} waypoints, cost={total_cost:.2f}, "
                f"gain={elevation_gain:.3f}, loss={elevation_loss:.3f}")
    
    return stats


def overlay_path_on_terrain(heightmap: np.ndarray, path: List[Tuple[int, int]], 
                            output_path: str, title: str = "Mission Path") -> None:
    """
    Create visualization of path overlaid on terrain heightmap.
    
    Args:
        heightmap (np.ndarray): Terrain heightmap
        path (List[Tuple[int, int]]): Path coordinates
        output_path (str): Output file path for visualization
        title (str): Title for the visualization
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Display heightmap with terrain colormap
    im = ax.imshow(heightmap, cmap='terrain', origin='upper')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Elevation')
    
    # Overlay path if it exists
    if path and len(path) > 1:
        # Extract coordinates (path is in (row, col) format)
        rows, cols = zip(*path)
        
        # Plot path
        ax.plot(cols, rows, color='red', linewidth=3, alpha=0.8, label='Optimal Path', zorder=10)
        
        # Mark start and goal points
        ax.plot(cols[0], rows[0], 'go', markersize=12, label='Start', 
               markeredgecolor='black', markeredgewidth=2, zorder=11)
        ax.plot(cols[-1], rows[-1], 'ro', markersize=12, label='Goal', 
               markeredgecolor='black', markeredgewidth=2, zorder=11)
        
        # Add legend
        ax.legend(loc='upper right')
        
        logger.info(f"Path overlay: {len(path)} waypoints from {path[0]} to {path[-1]}")
    else:
        ax.text(0.5, 0.5, "No Path Found", transform=ax.transAxes, 
               ha='center', va='center', fontsize=20, color='red',
               bbox=dict(boxstyle="round,pad=1", facecolor="white", alpha=0.8))
        logger.warning("No path to visualize")
    
    # Set title and labels
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    
    # Remove ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Save figure 
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"✓ Path visualization saved to: {output_path}")


def overlay_path_on_cost_map(cost_map: np.ndarray, path: List[Tuple[int, int]], 
                             output_path: str) -> None:
    """
    Create visualization of path overlaid on cost map.
    
    Args:
        cost_map (np.ndarray): Cost map
        path (List[Tuple[int, int]]): Path coordinates
        output_path (str): Output file path for visualization
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Display cost map with hot colormap (red = expensive, blue = cheap)
    im = ax.imshow(cost_map, cmap='hot_r', origin='upper', vmin=0, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Traversal Cost')
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(['Low', 'Moderate', 'High', 'Very High', 'Extreme'])
    
    # Overlay path if it exists
    if path and len(path) > 1:
        # Extract coordinates
        rows, cols = zip(*path)
        
        # Plot path with bright color for visibility
        ax.plot(cols, rows, color='cyan', linewidth=3, alpha=0.9, label='Optimal Path', zorder=10)
        
        # Mark start and goal
        ax.plot(cols[0], rows[0], 'go', markersize=12, label='Start',
               markeredgecolor='white', markeredgewidth=2, zorder=11)
        ax.plot(cols[-1], rows[-1], 'mo', markersize=12, label='Goal',
               markeredgecolor='white', markeredgewidth=2, zorder=11)
        
        ax.legend(loc='upper right')
    
    # Set title and labels
    ax.set_title("Cost Map with Optimal Path", fontsize=14, fontweight='bold')
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Save figure 
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"✓ Cost map path visualization saved to: {output_path}")


def select_random_points(heightmap: np.ndarray, min_distance: int = 50) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Automatically select start and goal points with minimum separation.
    
    Args:
        heightmap (np.ndarray): Terrain heightmap
        min_distance (int): Minimum pixel distance between start and goal
        
    Returns:
        Tuple[Tuple[int, int], Tuple[int, int]]: (start, goal) positions
    """
    h, w = heightmap.shape
    
    # Select start point randomly
    start = (np.random.randint(0, h), np.random.randint(0, w))
    
    # Select goal point with minimum distance from start
    attempts = 0
    max_attempts = 100
    
    while attempts < max_attempts:
        goal = (np.random.randint(0, h), np.random.randint(0, w))
        
        # Check distance
        distance = np.sqrt((goal[0] - start[0])**2 + (goal[1] - start[1])**2)
        
        if distance >= min_distance:
            logger.info(f"Selected random points: start={start}, goal={goal}, distance={distance:.1f}")
            return start, goal
        
        attempts += 1
    
    # Fallback: use opposite corners
    goal = (h - start[0] - 1, w - start[1] - 1)
    logger.warning(f"Could not find well-separated points, using corners: start={start}, goal={goal}")
    return start, goal
