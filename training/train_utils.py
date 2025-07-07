import os
from config import BASE_MODEL

def get_last_checkpoint(output_dir):
    if not os.path.exists(output_dir):
        return BASE_MODEL
    checkpoints = [os.path.join(output_dir, d)
                   for d in os.listdir(output_dir)
                   if d.startswith("checkpoint-")]
    if not checkpoints:
        return BASE_MODEL, None
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))
    print(checkpoints)
    return checkpoints[-1], checkpoints[-1]