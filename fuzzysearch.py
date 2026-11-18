class FuzzySearchIterator:

    def __init__(self,block_ids:list[str]):
        """"Stores the contents of `block_ids` for iteration"""
        self.block_ids = block_ids
        self.size = len(block_ids)
        self.position = int(-1)
        if self.size == 0:
            raise ValueError("Can not initialize an empty Fuzzy Search Iterator.")
    
    def progress(self) -> str:
        """"Returns a string showing current 1-indexed [position]/[number of items stored in the iterator]"""
        position = (self.position % self.size) + 1
        return f"{position}/{self.size}"

    def next(self) -> str:
        """Iterates to the next position in the list, returning the object stored there"""
        self.position = (self.position+1) % (self.size)
        return self.block_ids[self.position]

    def prev(self) -> str:
        """Iterates to the previous position in the list, returning the object stored there"""
        self.position = (self.position-1) % (self.size)
        return self.block_ids[self.position]