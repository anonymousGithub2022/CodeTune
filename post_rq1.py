import torch
import numpy as np

final = []
for model_id in range(1, 7):
    results = torch.load(str(model_id) + '_constant.tar')
    results = results[10:]
    assert len(results) == 30
    res = np.array([d[3] for d in results]).reshape([3, 10])
    final.append(res)
final = np.concatenate(final)
np.savetxt('rq1.csv', final, delimiter=',')