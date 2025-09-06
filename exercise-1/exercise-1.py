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
        m_idx = []
        for edge in self.f:
            if edge[1] == transition:
                m_idx.append(edge[0] - 1)

        if not m_idx:
            return False

        return all(self.M[idx] >= 1 for idx in m_idx)

        


    def add_marking(self, place):
        self.M[place-1] += 1

    def fire_transition(self, transition):
        # Gather all input places (place -> transition) and output places (transition -> place).
        in_places = [edge[0]-1 for edge in self.f if edge[1] == transition]
        out_places = [edge[1]-1 for edge in self.f if edge[0] == transition]

        # No arcs for this transition -> nothing to do.
        if not in_places and not out_places:
            return

        # Safety: indices must exist.
        n = len(self.M)
        if any(i < 0 or i >= n for i in in_places + out_places):
            raise IndexError("Edge references a place index that doesn't exist.")

        # Classic Petri net semantics require ALL input places to have >= 1 token.
        # (If you truly want your new 'any input has a token' rule, replace 'all' with 'any'.)
        if not all(self.M[i] >= 1 for i in in_places):
            return  # not enabled under AND semantics

        # Consume one token from each input place.
        for i in in_places:
            self.M[i] -= 1

        # Produce one token in each output place.
        for j in out_places:
            self.M[j] += 1


    # def fire_transition(self, transition):
    #     p_s = -1
    #     p_t = -1

    #     for edge in self.f:
    #         if (edge[0] == transition):
    #             p_t = edge[1]
    #         if (edge[1] == transition):
    #             p_s = edge[0]

    #     if (self.M[p_s-1]<=0):
    #         return

    #     self.M[p_s-1] -= 1
    #     self.M[p_t-1] += 1

# p = PetriNet()

# p.add_place(1)  # add place with id 1
# p.add_place(2)
# p.add_place(3)
# p.add_place(4)
# p.add_transition("A", -1)  # add transition "A" with id -1
# p.add_transition("B", -2)
# p.add_transition("C", -3)
# p.add_transition("D", -4)

# p.add_edge(1, -1)
# p.add_edge(-1, 2)
# p.add_edge(2, -2).add_edge(-2, 3)
# p.add_edge(2, -3).add_edge(-3, 3)
# p.add_edge(3, -4)
# p.add_edge(-4, 4)

# print(p.is_enabled(-1), p.is_enabled(-2), p.is_enabled(-3), p.is_enabled(-4))

# p.add_marking(1)  # add one token to place id 1
# print(p.is_enabled(-1), p.is_enabled(-2), p.is_enabled(-3), p.is_enabled(-4))

# p.fire_transition(-1)  # fire transition A
# print(p.is_enabled(-1), p.is_enabled(-2), p.is_enabled(-3), p.is_enabled(-4))

# p.fire_transition(-3)  # fire transition C
# print(p.is_enabled(-1), p.is_enabled(-2), p.is_enabled(-3), p.is_enabled(-4))

# p.fire_transition(-4)  # fire transition D
# print(p.is_enabled(-1), p.is_enabled(-2), p.is_enabled(-3), p.is_enabled(-4))

# p.add_marking(2)  # add one token to place id 2
# print(p.is_enabled(-1), p.is_enabled(-2), p.is_enabled(-3), p.is_enabled(-4))

# p.fire_transition(-2)  # fire transition B
# print(p.is_enabled(-1), p.is_enabled(-2), p.is_enabled(-3), p.is_enabled(-4))

# p.fire_transition(-4)  # fire transition D
# print(p.is_enabled(-1), p.is_enabled(-2), p.is_enabled(-3), p.is_enabled(-4))

# # by the end of the execution there should be 2 tokens on the final place
# print(p.get_tokens(4))