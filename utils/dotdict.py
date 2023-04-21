class DotNotationDict(dict):
    """Enables dot notation access to dictionary attributes."""

    def getattr(self, key):
        return self.get(key)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __delattr__(self, key):
        self.__delitem__(key)
