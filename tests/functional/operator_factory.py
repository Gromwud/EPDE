from epde.operators.common.coeff_calculation import LinRegBasedCoeffsEquation
from epde.operators.common.sparsity import LASSOSparsity
from epde.operators.utils.operator_mappers import map_operator_between_levels
from epde.operators.utils.template import CompoundOperator
import epde.operators.common.fitness as fitness

class FitnessOperatorFactory:
    @staticmethod
    def create(name: str, params: dict) -> CompoundOperator:
        cls_map = {
            "PIC": fitness.PIC,
            "DeepXDEBasedFitness": fitness.DeepXDEBasedFitness,
            "L2LRFitness": fitness.L2LRFitness,
        }
        if name not in cls_map:
            raise ValueError(f"Unknown operator: {name}")

        operator = cls_map[name](list(params.keys()))
        operator.set_suboperators({
            "sparsity": LASSOSparsity(),
            "coeff_calc": LinRegBasedCoeffsEquation(),
        })
        operator.params = params

        fitness_cond = lambda x: not getattr(x, "fitness_calculated", False)
        operator = map_operator_between_levels(
            operator,
            'gene level',
            "chromosome level",
            objective_condition=fitness_cond,
        )
        return operator
    #"chromosome level"
    #"gene level"