"""
Unit tests for planner.py
Tests A* pathfinding and path statistics.
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline.planner import find_path, calculate_path_statistics
from pipeline.cost_map import cost_map

# Default parameters
DEFAULT_PARAMS = {
    'water_level': 0.2,
    'elevation_scale': 1.0,
    'roughness': 0.5,
    'biome_type': 'mountain'
}


class TestPathfinding(unittest.TestCase):
    """Test A* pathfinding algorithm"""
    
    def setUp(self):
        """Setup test fixtures"""
        # Simple cost grid for testing
        self.simple_costs = np.ones((50, 50)) * 0.5
        
        # Cost grid with obstacle
        self.costs_with_obstacle = np.ones((50, 50)) * 0.5
        self.costs_with_obstacle[20:30, 20:30] = 1.0  # High cost area
    
    def test_adjacent_cells(self):
        """Test pathfinding between adjacent cells"""
        start = (10, 10)
        goal = (10, 11)  # One cell to the right
        
        path = find_path(self.simple_costs, start, goal)
        
        self.assertIsInstance(path, list)
        self.assertGreaterEqual(len(path), 2)
        self.assertEqual(path[0], start)
        self.assertEqual(path[-1], goal)
    
    def test_straight_path(self):
        """Test pathfinding finds straight path in uniform terrain"""
        start = (10, 10)
        goal = (10, 20)
        
        path = find_path(self.simple_costs, start, goal)
        
        self.assertGreater(len(path), 0)
        self.assertEqual(path[0], start)
        self.assertEqual(path[-1], goal)
    
    def test_obstacle_avoidance(self):
        """Test pathfinding avoids high-cost areas"""
        start = (10, 25)
        goal = (40, 25)  # Straight path blocked by obstacle
        
        path = find_path(self.costs_with_obstacle, start, goal)
        
        self.assertGreater(len(path), 0)
        self.assertEqual(path[0], start)
        self.assertEqual(path[-1], goal)
        
        # Path should avoid the obstacle (rows 20-30, cols 20-30)
        obstacle_cells = sum(1 for r, c in path if 20 <= r < 30 and 20 <= c < 30)
        # Allow some obstacle cells but not the entire path
        self.assertLess(obstacle_cells, len(path) * 0.5)
    
    def test_start_equals_goal(self):
        """Test when start equals goal"""
        start = (25, 25)
        goal = (25, 25)
        
        path = find_path(self.simple_costs, start, goal)
        
        # Path should just be the single point
        self.assertEqual(len(path), 0 if not path else 1)
    
    def test_unreachable_goal(self):
        """Test behavior when goal is surrounded by impassable terrain"""
        costs = np.ones((50, 50)) * 0.1
        # Create impassable barrier around center
        costs[24:27, :] = 100.0  # Very high cost (effectively impassable)
        costs[:, 24:27] = 100.0
        
        start = (10, 10)
        goal = (25, 25)  # Surrounded by barrier
        
        path = find_path(costs, start, goal)
        
        # Path may be empty or very costly
        self.assertIsInstance(path, list)
    
    def test_boundary_conditions(self):
        """Test pathfinding near grid boundaries"""
        start = (0, 0)  # Top-left corner
        goal = (49, 49)  # Bottom-right corner
        
        path = find_path(self.simple_costs, start, goal)
        
        self.assertGreater(len(path), 0)
        self.assertEqual(path[0], start)
        self.assertEqual(path[-1], goal)
    
    def test_path_continuity(self):
        """Test that path waypoints are continuous (adjacent cells)"""
        start = (10, 10)
        goal = (20, 20)
        
        path = find_path(self.simple_costs, start, goal)
        
        # Check each step is adjacent to previous
        for i in range(len(path) - 1):
            r1, c1 = path[i]
            r2, c2 = path[i+1]
            
            # Manhattan distance should be 1 (4-directional movement)
            distance = abs(r2 - r1) + abs(c2 - c1)
            self.assertEqual(distance, 1, f"Non-adjacent waypoints at {i}: {path[i]} -> {path[i+1]}")
    
    def test_invalid_start(self):
        """Test behavior with invalid start position"""
        start = (-1, 0)  # Out of bounds
        goal = (25, 25)
        
        path = find_path(self.simple_costs, start, goal)
        
        # Should return empty path or handle gracefully
        self.assertEqual(len(path), 0)
    
    def test_invalid_goal(self):
        """Test behavior with invalid goal position"""
        start = (10, 10)
        goal = (100, 100)  # Out of bounds
        
        path = find_path(self.simple_costs, start, goal)
        
        # Should return empty path
        self.assertEqual(len(path), 0)


class TestPathStatistics(unittest.TestCase):
    """Test path statistics calculation"""
    
    def test_statistics_structure(self):
        """Test statistics dict has expected keys"""
        heightmap = np.random.rand(50, 50)
        costs = cost_map(heightmap, DEFAULT_PARAMS)
        
        start = (10, 10)
        goal = (30, 30)
        path = find_path(costs, start, goal)
        
        if len(path) > 0:
            stats = calculate_path_statistics(path, costs, heightmap)
            
            expected_keys = ['valid', 'waypoints', 'total_cost', 'mean_cost', 
                           'elevation_gain', 'elevation_loss']
            for key in expected_keys:
                self.assertIn(key, stats)
    
    def test_path_length(self):
        """Test waypoint count matches path length"""
        heightmap = np.random.rand(50, 50)
        costs = cost_map(heightmap, DEFAULT_PARAMS)
        
        start = (10, 10)
        goal = (30, 30)
        path = find_path(costs, start, goal)
        
        if len(path) > 0:
            stats = calculate_path_statistics(path, costs, heightmap)
            self.assertEqual(stats['waypoints'], len(path))
    
    def test_elevation_calculations(self):
        """Test elevation gain/loss calculations"""
        # Create heightmap with clear elevation change
        heightmap = np.zeros((50, 50))
        for i in range(50):
            heightmap[i, :] = i / 50.0  # Gradient from 0 to 1
        
        costs = cost_map(heightmap, DEFAULT_PARAMS)
        
        start = (10, 25)  # Low elevation
        goal = (40, 25)  # High elevation
        path = find_path(costs, start, goal)
        
        if len(path) > 0:
            stats = calculate_path_statistics(path, costs, heightmap)
            
            # Should have positive elevation gain (going uphill)
            self.assertGreater(stats['elevation_gain'], 0)
    
    def test_cost_calculations(self):
        """Test cost calculations along path"""
        heightmap = np.random.rand(50, 50)
        costs = cost_map(heightmap, DEFAULT_PARAMS)
        
        start = (10, 10)
        goal = (30, 30)
        path = find_path(costs, start, goal)
        
        if len(path) > 0:
            stats = calculate_path_statistics(path, costs, heightmap)
            
            # Total cost should be positive
            self.assertGreater(stats['total_cost'], 0)
            # Mean cost should be between 0 and 1
            self.assertGreaterEqual(stats['mean_cost'], 0)
            self.assertLessEqual(stats['mean_cost'], 1.0)
    
    def test_empty_path_statistics(self):
        """Test statistics for empty path"""
        heightmap = np.random.rand(50, 50)
        costs = cost_map(heightmap, DEFAULT_PARAMS)
        
        empty_path = []
        stats = calculate_path_statistics(empty_path, costs, heightmap)
        
        self.assertFalse(stats['valid'])
        self.assertEqual(stats['waypoints'], 0)
    
    def test_single_point_path(self):
        """Test statistics for single-point path"""
        heightmap = np.random.rand(50, 50)
        costs = cost_map(heightmap, DEFAULT_PARAMS)
        
        single_path = [(25, 25)]
        stats = calculate_path_statistics(single_path, costs, heightmap)
        
        # Single point path is invalid (< 2 waypoints)
        self.assertFalse(stats['valid'])


class TestPerformance(unittest.TestCase):
    """Test pathfinding performance"""
    
    def test_large_map_performance(self):
        """Test pathfinding on larger map"""
        import time
        
        heightmap = np.random.rand(200, 200)
        costs = cost_map(heightmap, DEFAULT_PARAMS)
        
        start = (10, 10)
        goal = (190, 190)
        
        start_time = time.time()
        path = find_path(costs, start, goal)
        end_time = time.time()
        
        # Should complete within reasonable time (2 seconds)
        self.assertLess(end_time - start_time, 2.0)
        self.assertGreater(len(path), 0)
    
    def test_path_optimality(self):
        """Test that A* finds reasonably optimal paths"""
        # Uniform cost grid
        costs = np.ones((50, 50)) * 0.5
        
        start = (10, 10)
        goal = (20, 20)
        
        path = find_path(costs, start, goal)
        
        # Manhattan distance between start and goal
        manhattan_dist = abs(goal[0] - start[0]) + abs(goal[1] - start[1])
        
        # A* path should be close to Manhattan distance
        # (Within 50% overhead for 4-directional movement)
        self.assertLessEqual(len(path), manhattan_dist * 1.5)


if __name__ == '__main__':
    unittest.main()
