from common import *
from controller.iLQGameCarController import iLQGameCarController
from controller.LQGame import my_solve_lq_game
from scipy.linalg import block_diag

# a blocking controller
class AggressiveCarController(iLQGameCarController):
    def __init__(self, car,config):
        super().__init__(car,config)

        ConfigObject.__init__(self,config)

    def getCostMatrices(self,xx_i,uu_i,xx_j,uu_j):
        return self.getAggressiveCostMatrices(xx_i,uu_i,xx_j,uu_j)

