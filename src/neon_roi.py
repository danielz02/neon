import os
from glob import glob
from dataset import NeonDataset
from task_pool import MPITaskPool
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("shapefile", type=str)
args = parser.parse_args()


def crop_once(arg):
    path, comm = arg
    ds = NeonDataset(path=path)
    ds.crop_roi(shapefile=args.shapefile, dest_dir=os.path.dirname(path))


exe = MPITaskPool()
if exe.is_parent():
    jobs = [(x, exe.comm) for x in sorted(glob("/scratch/sciteam/ILL_bbdv/NEON/DP3.30006.001/**/*.h5", recursive=True))]
else:
    jobs = []
exe.run(jobs, crop_once, log_freq=1)
