
class Model:
    def __init__(self,grid, diffusion = None, advection = None) -> None:
        self.grid = grid
        self.diffusion = diffusion
        self.advection = advection