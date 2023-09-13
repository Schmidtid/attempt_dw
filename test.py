import hybrid
import numpy as np
import dimod as di 

q = np.load('c.npy')

import neal
import dimod
import hybrid

from hybrid.reference.pt import FixedTemperatureSampler
from hybrid.reference.pt import SwapReplicasDownsweep

bqm = di.BQM(q,di.Vartype.BINARY)

print("BQM: {} nodes, {} edges, {:.2f} density".format(
    len(bqm), len(bqm.quadratic), hybrid.bqm_density(bqm)))


# PT workflow: temperature/beta is a property of a branch

n_sweeps = 10000
n_replicas = 30
n_iterations = 90

# replicas are initialized with random samples
state = hybrid.State.from_problem(bqm)
replicas = hybrid.States(*[state.updated() for _ in range(n_replicas)])

# get a reasonable beta range
beta_hot, beta_cold = neal.default_beta_range(bqm)

# generate betas for all branches/replicas
betas = np.geomspace(beta_hot, beta_cold, n_replicas)

# QPU branch: limits the PT workflow to QPU-sized problems
qpu = hybrid.IdentityDecomposer() | hybrid.QPUSubproblemAutoEmbeddingSampler() | hybrid.IdentityComposer()

# use QPU as the hottest temperature sampler and `n_replicas-1` fixed-temperature-samplers
update = hybrid.Branches(
    qpu,
    *[FixedTemperatureSampler(beta=beta, num_sweeps=n_sweeps) for beta in betas[1:]])

# swap step is `n_replicas-1` pairwise potential swaps
swap = SwapReplicasDownsweep(betas=betas)

# we'll run update/swap sequence for `n_iterations`
workflow = hybrid.Loop(update | swap, max_iter=n_iterations) \
         | hybrid.MergeSamples(aggregate=True)

# execute the workflow
solution = workflow.run(replicas).result()

# show execution profile
hybrid.profiling.print_counters(workflow)

_ = solution.result().samples.to_pandas_dataframe()
x = _[_.energy==_.energy.min()].iloc[0].values[:-2]
np.save('vector',x)