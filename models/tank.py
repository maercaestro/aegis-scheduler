class Tank:
    def __init__(self, id, crude_type, capacity):
        self.id = id
        self.crude_type = crude_type
        self.capacity = capacity
        self.level = 0.0
        
    def add_crude(self, amount):
        """Add crude to tank, respecting capacity limits"""
        can_add = min(amount, self.capacity - self.level)
        self.level += can_add
        return can_add
        
    def remove_crude(self, amount):
        """Remove crude from tank, cannot remove more than available"""
        can_remove = min(amount, self.level)
        self.level -= can_remove
        return can_remove
    
    def get_space_available(self):
        """Return available space in tank"""
        return self.capacity - self.level
        
    @property
    def is_empty(self):
        return self.level <= 0.001  # Using small epsilon for floating point comparison