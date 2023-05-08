import time
import numpy as np
from casadi import MX, Function
import biorbd_casadi as biorbd

from somersault import Models

model = biorbd.Model(Models.ACROBAT.value)

q_mx = MX.sym("q", model.nbQ(), 1)
qdot_mx = MX.sym("qdot", model.nbQ(), 1)
qddot_mx = MX.sym("qddot", model.nbQ(), 1)

def fun(q, qdot, qddot):
    """
    This function is a simple function.
    """
    model.ForwardDynamics(q, qdot, qddot).to_mx()


cse = False
casadi_issue = Function(
    "casadi_issue", [q_mx, qdot_mx, qddot_mx], [fun(q_mx, qdot_mx, qddot_mx)], {"cse": cse}).expand()

t0 = time.time()

m = int(1)

for i in range(m):
    q = np.random.rand(model.nbQ(), 1)
    qdot = np.random.rand(model.nbQ(), 1)
    qddot = np.random.rand(model.nbQ(), 1)
    casadi_issue(q, qdot, qddot)

print(cse, time.time() - t0)
