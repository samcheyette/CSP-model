"""
Union-Find (Disjoint Set) implementation for efficient connectivity tracking.


"""

class UnionFind:
    """Union-Find data structure with path compression and union by rank."""
    
    def __init__(self, elements=None):
        """
        Initialize a Union-Find data structure.
        
        Args:
            elements: Optional iterable of elements to initialize with
        """
        # Initialize with each element in its own set
        self.parent = {}
        self.rank = {}
        
        # Add initial elements if provided
        if elements:
            for element in elements:
                self.add(element)
    
    def add(self, element):
        """
        Add a new element as a singleton set.
        
        Args:
            element: The element to add
        """
        if element not in self.parent:
            self.parent[element] = element
            self.rank[element] = 0
    
    def find(self, element):
        """
        Find the representative (root) of the set containing element.
        Uses path compression for efficiency.
        
        Args:
            element: The element to find
            
        Returns:
            The representative of the set
        """
        # Ensure element exists
        if element not in self.parent:
            self.add(element)
            return element
            
        # Path compression: point all nodes along the path to the root
        if self.parent[element] != element:
            self.parent[element] = self.find(self.parent[element])
        return self.parent[element]
    
    def union(self, element1, element2):
        """
        Merge the sets containing element1 and element2.
        Uses union by rank for efficiency.
        
        Args:
            element1: First element
            element2: Second element
        """
        # Ensure both elements exist
        if element1 not in self.parent:
            self.add(element1)
        if element2 not in self.parent:
            self.add(element2)
            
        # Find the roots
        root1 = self.find(element1)
        root2 = self.find(element2)
        
        # Already in the same set
        if root1 == root2:
            return
        
        # Union by rank: attach smaller tree under root of larger tree
        if self.rank[root1] < self.rank[root2]:
            self.parent[root1] = root2
        elif self.rank[root1] > self.rank[root2]:
            self.parent[root2] = root1
        else:
            # Same rank, so increment rank of one of them
            self.parent[root2] = root1
            self.rank[root1] += 1
    
    def same_set(self, element1, element2):
        """
        Check if element1 and element2 are in the same set.
        
        Args:
            element1: First element
            element2: Second element
            
        Returns:
            True if in same set, False otherwise
        """
        return self.find(element1) == self.find(element2)
    
    def get_sets(self):
        """
        Return all sets as a dictionary mapping representatives to sets.
        
        Returns:
            Dict mapping each representative to the set it represents
        """
        sets = {}
        for element in self.parent:
            representative = self.find(element)
            if representative not in sets:
                sets[representative] = set()
            sets[representative].add(element)
        return sets
    
    def count_sets(self):
        """
        Count the number of disjoint sets.
        
        Returns:
            The number of disjoint sets
        """
        representatives = set()
        for element in self.parent:
            representatives.add(self.find(element))
        return len(representatives) 