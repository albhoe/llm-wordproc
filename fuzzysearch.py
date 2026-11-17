class FuzzySearchIterator:
    def __init__(self,block_ids:list[str]):
        """"Stores the contents of `block_ids` for iteration"""
        self.block_ids = block_ids
        self.size = len(block_ids)
        self.position = -1

    def next(self):
        """Iterates to the next position in the list, returning the object stored there"""
        self.position = (self.position+1) % (self.size)
        return self.block_ids[self.position]

    def prev(self):
        """Iterates to the previous position in the list, returning the object stored there"""
        self.position = (self.position-1) % (self.size)
        return self.block_ids[self.position]