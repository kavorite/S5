import haiku as hk
import numpy as np

from s5 import S5Encoder


@hk.transform
def model(x):
    return S5Encoder(16, 16, factor_rank=8)(x)


rng = hk.PRNGSequence(42)
x = np.zeros([256, 16])
params = model.init(next(rng), x)
print(model.apply(params, next(rng), x).shape)
