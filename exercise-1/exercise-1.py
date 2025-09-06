class PetriNet():

    def __init__(self):
        self.p = []
        self.t = []
        self.f = []
        self.M = []

    def add_place(self, name):
        self.p.append(name)
        self.M.append(0)

    def add_transition(self, name, id):
        self.t.append(id)

    def add_edge(self, source, target):
        self.f.append([source, target])
        return self

    def get_tokens(self, place):
        return self.M[place-1]

    def is_enabled(self, transition):
        acc = -1
        for edge in self.f:
            if (edge[1] == transition):
                acc = edge[0]

        return (self.M[acc-1] == 1)
        


    def add_marking(self, place):
        self.M[place-1] += 1

    def fire_transition(self, transition):
        p_s = -1
        p_t = -1

        for edge in self.f:
            if (edge[0] == transition):
                p_t = edge[1]
            if (edge[1] == transition):
                p_s = edge[0]
        
        self.M[p_s-1] -= 1
        self.M[p_t-1] += 1

print(p.is_enabled(-1), p.is_enabled(-2), p.is_enabled(-3), p.is_enabled(-4))

p.add_marking(1)  # add one token to place id 1
print(p.is_enabled(-1), p.is_enabled(-2), p.is_enabled(-3), p.is_enabled(-4))

p.fire_transition(-1)  # fire transition A
print(p.is_enabled(-1), p.is_enabled(-2), p.is_enabled(-3), p.is_enabled(-4))

p.fire_transition(-3)  # fire transition C
print(p.is_enabled(-1), p.is_enabled(-2), p.is_enabled(-3), p.is_enabled(-4))

p.fire_transition(-4)  # fire transition D
print(p.is_enabled(-1), p.is_enabled(-2), p.is_enabled(-3), p.is_enabled(-4))

p.add_marking(2)  # add one token to place id 2
print(p.is_enabled(-1), p.is_enabled(-2), p.is_enabled(-3), p.is_enabled(-4))

p.fire_transition(-2)  # fire transition B
print(p.is_enabled(-1), p.is_enabled(-2), p.is_enabled(-3), p.is_enabled(-4))

p.fire_transition(-4)  # fire transition D
print(p.is_enabled(-1), p.is_enabled(-2), p.is_enabled(-3), p.is_enabled(-4))

# by the end of the execution there should be 2 tokens on the final place
print(p.get_tokens(4))