import numpy as np


def pass_json(run, luminosityBlock, lumi_mask):
    if str(run) not in lumi_mask:
        return False
    for lrange in lumi_mask[str(run)]:
        if int(lrange[0]) <= luminosityBlock < int(lrange[1]):
            return True
    return False


def pass_json_array(runs, luminosityBlocks, lumi_mask):
    out = []
    for run, block in zip(runs, luminosityBlocks):
        out.append(pass_json(run, block, lumi_mask))
    return np.array(out)
