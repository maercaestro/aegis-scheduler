class CrudeInventory:
    def __init__(self):
        self.inventory = {}
        self.tanks = []
        
    def add_tank(self, tank):
        """Register a tank with the inventory system"""
        self.tanks.append(tank)
        if tank.crude_type not in self.inventory:
            self.inventory[tank.crude_type] = 0.0
    
    def sync_with_tanks(self):
        """Update inventory levels based on tank contents"""
        # Reset inventory
        self.inventory = {crude_type: 0.0 for crude_type in self.inventory}
        
        # Sum up tank contents
        for tank in self.tanks:
            if tank.crude_type not in self.inventory:
                self.inventory[tank.crude_type] = 0.0
            self.inventory[tank.crude_type] += tank.level
            
    def add_crude(self, crude_type, amount):
        """Add crude to appropriate tanks"""
        remaining = amount
        
        # Find tanks of matching crude type with available space
        for tank in sorted(self.tanks, key=lambda t: t.get_space_available(), reverse=True):
            if tank.crude_type == crude_type:
                added = tank.add_crude(remaining)
                remaining -= added
                if remaining <= 0:
                    break
                    
        # Update inventory to match tank levels
        self.sync_with_tanks()
        return amount - remaining  # Return amount successfully added
        
    def use_crude(self, crude_type, amount):
        """Use crude from tanks"""
        remaining = amount
        
        # Remove from tanks with matching crude type
        for tank in sorted(self.tanks, key=lambda t: t.level, reverse=True):
            if tank.crude_type == crude_type:
                removed = tank.remove_crude(remaining)
                remaining -= removed
                if remaining <= 0:
                    break
                    
        # Update inventory to match tank levels
        self.sync_with_tanks()
        return amount - remaining  # Return amount successfully used
        
    def get_inventory_level(self, crude_type):
        """Get current inventory level for crude type"""
        # First ensure inventory matches tank levels
        self.sync_with_tanks()
        return self.inventory.get(crude_type, 0.0)