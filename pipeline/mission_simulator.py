"""
Mission Simulator with A* Pathfinding
Provides intelligent pathfinding for terrain navigation with cost-based routing.
"""

import numpy as np
import heapq
import matplotlib.pyplot as plt
import os
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


def compute_costmap(heightmap: np.ndarray) -> np.ndarray:
    """
    Compute cost map for pathfinding based on elevation and slope.
    
    Higher elevations and steeper slopes have higher costs.
    
    Args:
        heightmap (np.ndarray): 2D heightmap array (normalized 0-1)
        
    Returns:
        np.ndarray: Normalized cost map (0-1) where higher values are more expensive
    """
    # Ensure heightmap is properly normalized
    if heightmap.max() > 1.0:
        heightmap = heightmap / heightmap.max()
    
    # Compute gradients for slope calculation
    gx, gy = np.gradient(heightmap)
    slope = np.sqrt(gx**2 + gy**2)
    
    # Combine elevation penalty (0.5 weight) and slope penalty (2.0 weight)
    # This makes steep slopes much more expensive than high elevation
    cost = heightmap * 0.5 + slope * 2.0
    
    # Normalize to 0-1 range
    if cost.max() > 0:
        cost = cost / cost.max()
    
    return cost


def a_star_search(costmap: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    A* pathfinding algorithm to find optimal path through terrain.
    
    Args:
        costmap (np.ndarray): Cost map where higher values are more expensive
        start (Tuple[int, int]): Starting position (row, col)
        goal (Tuple[int, int]): Goal position (row, col)
        
    Returns:
        List[Tuple[int, int]]: Path from start to goal as list of (row, col) coordinates
    """
    h, w = costmap.shape
    
    # Validate start and goal positions
    if not (0 <= start[0] < h and 0 <= start[1] < w):
        logger.error(f"Start position {start} is out of bounds")
        return []
    
    if not (0 <= goal[0] < h and 0 <= goal[1] < w):
        logger.error(f"Goal position {goal} is out of bounds")
        return []
    
    # Priority queue: (f_score, position)
    open_set = [(0, start)]
    
    # Track path reconstruction
    came_from = {}
    
    # Cost from start to each position
    g_score = {start: 0}
    
    # 4-directional movement (up, down, left, right)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    logger.info(f"Starting A* search from {start} to {goal}")
    
    while open_set:
        # Get position with lowest f_score
        _, current = heapq.heappop(open_set)
        
        # Check if we reached the goal
        if current == goal:
            # Reconstruct path
            path = [goal]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            
            logger.info(f"✓ Path found with {len(path)} waypoints")
            return path[::-1]  # Reverse to get start-to-goal order
        
        # Explore neighbors
        for dx, dy in directions:
            nx, ny = current[0] + dx, current[1] + dy
            
            # Check bounds
            if 0 <= nx < h and 0 <= ny < w:
                # Calculate movement cost (base cost + terrain cost)
                movement_cost = 1.0 + costmap[nx, ny]  # Base movement + terrain penalty
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
    
    logger.warning("No path found to goal")
    return []  # No path found


def overlay_path_on_heightmap(heightmap: np.ndarray, path: List[Tuple[int, int]], output_path: str) -> None:
    """
    Create visualization of path overlaid on terrain heightmap.
    
    Args:
        heightmap (np.ndarray): Terrain heightmap
        path (List[Tuple[int, int]]): Path coordinates
        output_path (str): Output file path for visualization
    """
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Display heightmap with terrain colormap
    im = ax.imshow(heightmap, cmap='terrain', origin='upper')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Elevation')
    
    # Overlay path if it exists
    if path and len(path) > 1:
        # Extract coordinates (remember path is in (row, col) format)
        rows, cols = zip(*path)
        
        # Plot path
        ax.plot(cols, rows, color='red', linewidth=3, alpha=0.8, label='Optimal Path')
        
        # Mark start and end points
        ax.plot(cols[0], rows[0], 'go', markersize=10, label='Start', markeredgecolor='black')
        ax.plot(cols[-1], rows[-1], 'ro', markersize=10, label='Goal', markeredgecolor='black')
        
        # Add legend
        ax.legend()
        
        logger.info(f"Path visualization: {len(path)} waypoints from {path[0]} to {path[-1]}")
    else:
        ax.set_title("No Path Found")
        logger.warning("No path to visualize")
    
    # Set title and labels
    ax.set_title("Mission Simulation: A* Pathfinding", fontsize=14, fontweight='bold')
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


def select_points_and_simulate(heightmap: np.ndarray, output_dir: str = "Output") -> Optional[str]:
    """
    Interactive point selection and pathfinding simulation.
    
    Args:
        heightmap (np.ndarray): Terrain heightmap
        output_dir (str): Directory to save output files
        
    Returns:
        Optional[str]: Path to generated visualization file, or None if failed
    """
    logger.info("Starting interactive mission simulation...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Create interactive figure for point selection
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(heightmap, cmap='terrain', origin='upper')
        ax.set_title("Mission Planning: Click Start Point, then Goal Point", fontsize=14, fontweight='bold')
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        
        # Add instructions
        ax.text(0.02, 0.98, "Instructions:\n1. Click START point\n2. Click GOAL point", 
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        logger.info("Waiting for user to select start and goal points...")
        print("🎯 Mission Planning Interface Opened!")
        print("📍 Click on the terrain to select:")
        print("   1. START point (green)")
        print("   2. GOAL point (red)")
        
        # Get two points from user
        points = plt.ginput(2, timeout=0)  # No timeout - wait for user
        plt.close(fig)
        
        # Validate point selection
        if len(points) < 2:
            logger.warning("Insufficient points selected for mission planning")
            print("❌ Mission planning cancelled - need both start and goal points")
            return None
        
        # Convert points to heightmap coordinates (row, col)
        # plt.ginput returns (x, y) which corresponds to (col, row)
        start = (int(points[0][1]), int(points[0][0]))  # (row, col)
        goal = (int(points[1][1]), int(points[1][0]))   # (row, col)
        
        logger.info(f"Mission points selected - Start: {start}, Goal: {goal}")
        print(f"🚀 Mission Planning:")
        print(f"   📍 Start: {start}")
        print(f"   🎯 Goal: {goal}")
        
        # Compute cost map for pathfinding
        print("🧮 Computing terrain cost map...")
        costmap = compute_costmap(heightmap)
        
        # Find optimal path using A* algorithm
        print("🔍 Searching for optimal path...")
        path = a_star_search(costmap, start, goal)
        
        # Generate output visualization
        output_path = os.path.join(output_dir, "mission_path_overlay.png")
        overlay_path_on_heightmap(heightmap, path, output_path)
        
        # Report results
        if path:
            path_length = len(path)
            total_cost = sum(costmap[pos[0], pos[1]] for pos in path)
            
            print(f"✅ Mission planning complete!")
            print(f"   📏 Path length: {path_length} waypoints")
            print(f"   💰 Total cost: {total_cost:.2f}")
            print(f"   🖼️  Visualization: {output_path}")
            
            logger.info(f"Mission simulation successful: {path_length} waypoints, cost: {total_cost:.2f}")
        else:
            print(f"❌ No viable path found between selected points")
            print(f"🖼️  Visualization saved anyway: {output_path}")
            
            logger.warning("No path found in mission simulation")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Mission simulation failed: {e}")
        print(f"❌ Mission simulation error: {e}")
        return None


def analyze_terrain_difficulty(heightmap: np.ndarray) -> dict:
    """
    Analyze terrain characteristics for mission planning.
    
    Args:
        heightmap (np.ndarray): Terrain heightmap
        
    Returns:
        dict: Terrain analysis statistics
    """
    costmap = compute_costmap(heightmap)
    
    # Compute gradients for slope analysis
    gx, gy = np.gradient(heightmap)
    slope = np.sqrt(gx**2 + gy**2)
    
    stats = {
        'elevation_range': (float(heightmap.min()), float(heightmap.max())),
        'mean_elevation': float(heightmap.mean()),
        'max_slope': float(slope.max()),
        'mean_slope': float(slope.mean()),
        'terrain_difficulty': float(costmap.mean()),  # Overall difficulty
        'navigable_area': float(np.sum(costmap < 0.5) / costmap.size * 100),  # % low-cost terrain
        'shape': heightmap.shape
    }
    
    return stats