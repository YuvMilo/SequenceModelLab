

def dict_to_immutable(d):
    return frozenset(d.items())


class ImmutableDict:
    def __init__(self, d):
        self.d = d
        self.immutable_d = dict_to_immutable(self.d)

    def __getitem__(self, key):
        return self.d[key]

    def __hash__(self):
        return hash(self.immutable_d)

    def __str__(self):
        return str(self.d)

    def __eq__(self, other):
        return self.immutable_d == other.immutable_d

    def get_dict(self):
        return self.d
