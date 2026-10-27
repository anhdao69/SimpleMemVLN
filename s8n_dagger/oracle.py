"""Instruction-route follower with explicit final radius and error propagation."""


class ReferenceOracle:
    def __init__(self, path, follower_factory):
        if len(path) < 2:
            raise ValueError('Reference route requires at least two positions')
        self.path = path
        self.index = 1
        self.factory = follower_factory
        self.follower = self.factory(.25 if self.index == len(path) - 1 else 1.8)

    def action(self):
        advanced = False
        while self.index < len(self.path):
            action = self.follower.get_next_action(self.path[self.index])
            if action is None or int(action) not in range(4):
                raise ValueError('Oracle produced no valid action')
            if int(action) != 0:
                return int(action), advanced
            self.index += 1
            advanced = True
            if self.index == len(self.path) - 1:
                self.follower = self.factory(.25)
        return 0, advanced
