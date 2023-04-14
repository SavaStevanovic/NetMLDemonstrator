import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import numpy as np
from citylearn.agents.rbc import HourRBC
# env = CityLearn()

# # Define Pyomo model
# model = pyo.ConcreteModel()

# # Define optimization variables
# model.x = pyo.Var(range(24), within=pyo.NonNegativeReals)

# # Define objective function
# model.obj = pyo.Objective(expr=sum(model.x[i] for i in range(24)), sense=pyo.minimize)

# # Define constraints
# model.constraints = pyo.ConstraintList()
# for t in range(24):
#     model.constraints.add(env.get_building_obs()[0]['elec_load'][t] <= model.x[t])

# # Solve optimization problem
# opt = SolverFactory('glpk')
# results = opt.solve(model)

# # Print optimal solution
# print('Optimal solution:')
# print([model.x[t]() for t in range(24)])