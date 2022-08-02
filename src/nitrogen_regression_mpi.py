import os
import shutil
import rasterio
import numpy as np
from glob import glob
from tqdm import tqdm
from mpi4py import MPI
from rasterio import windows
from itertools import product
from scipy.stats import linregress
from argparse import ArgumentParser

comm = MPI.COMM_WORLD  # get MPI communicator object
size = comm.size  # total number of processes
rank = comm.rank  # rank of this process
status = MPI.Status()  # get MPI status object

NAMES = ["slope", "intercept", "r", "p", "se"]


def enum(*sequential, **named):
    """Handy way to fake an enumerated type in Python
    http://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
    """
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)


# Define MPI message tags
tags = enum('READY', 'DONE', 'EXIT', 'START')


def batched_lstsq(x, y):
    mask = np.isnan(x)
    x = x[~mask]
    y = y[~mask]
    if rank == 1:
        print(x, y)
    return np.array(linregress(x, y))  # slope, intercept, r, p, se


def get_tiles(ds, width=1024, height=1024):
    nols, nrows = ds.meta['width'], ds.meta['height']
    offsets = product(range(0, nols, width), range(0, nrows, height))
    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
    for col_off, row_off in offsets:
        window = windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        transform = windows.transform(window, ds.transform)
        yield window, transform


def main(folder):
    dest_path = os.path.join(folder, f"ts_trend_{TRAIT}.tif")

    fs = []
    years = []
    for f in sorted(glob(os.path.join(folder, "*", f"{TRAIT}_stack0.tif"))):
        try:
            yyyy = int(f.split("/")[-2])
            shutil.copy(f, f"/tmp/{yyyy}.tif")
            f = os.path.join("/tmp", f"{yyyy}.tif")
            fs.append(rasterio.open(f))
            years.append(yyyy)
        except rasterio.errors.RasterioIOError as e:
            print(e)
            continue

    meta = fs[0].meta.copy()
    meta["count"] = 5
    meta["nodata"] = -9999
    meta["dtype"] = np.float32
    with rasterio.open(dest_path, "w", **meta) as dest:
        tasks = [x[0] for x in get_tiles(fs[0])]
        task_index = 0
        num_workers = size - 1
        closed_workers = 0

        while closed_workers < num_workers:
            data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            source = status.Get_source()
            tag = status.Get_tag()
            if tag == tags.READY:
                # Worker is ready, so send it a task
                if task_index < len(tasks):
                    trait_stack = np.stack([x.read(1, window=tasks[task_index]) for x in fs])
                    comm.send((trait_stack, tasks[task_index]), dest=source, tag=tags.START)
                    print("Sending task %d/%d to worker %d" % (task_index, len(tasks), source))
                    task_index += 1
                else:
                    comm.send(None, dest=source, tag=tags.EXIT)
            elif tag == tags.DONE:
                results, window = data
                print("Got data from worker %d" % source, results.shape, window)
                dest.write(results, window=window)
            elif tag == tags.EXIT:
                print("Worker %d exited." % source)
                closed_workers += 1

        [x.close() for x in fs]
    print(dest_path)

    meta["count"] = 1
    with rasterio.open(dest_path) as src:
        for i, name in enumerate(NAMES):
            dest_path = os.path.join(folder, f"ts_trend_{TRAIT}_{name}.tif")
            with rasterio.open(dest_path) as dest:
                dest.write(src.read(i + 1), 1)


def worker():
    name = MPI.Get_processor_name()
    print("I am a worker with rank %d on %s." % (rank, name))
    while True:
        comm.send(None, dest=0, tag=tags.READY)
        data = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()

        if tag == tags.START:
            # Do the work here
            trait_stack, window = data
            y, h, w = trait_stack.shape
            xs = np.tile(np.arange(y), h * w).reshape(h * w, y) * 1.
            trait_stack = trait_stack.reshape(y, -1).T  # (h * w, y)
            trait_stack[trait_stack == -9999] = np.nan
            # xs[trait_stack == -9999] = np.nan
            results = np.ones((5, h * w)) * -9999.
            for i in range(h * w):
                if np.sum(np.isnan(trait_stack[i])) > 20:
                    continue
                results[:, i] = batched_lstsq(xs[i], trait_stack[i])
            # results[:, np.sum(np.isnan(trait_stack), axis=1) > 17] = -9999
            results = results.reshape(5, h, w)
            results[np.isnan(results)] = -9999
            if MASK is not None:
                with rasterio.open(MASK) as src:
                    mask = src.read(1, window=window)
                    results[:, mask == 0] = -9999
            # Work ends here... Sending results back
            comm.send((results, window), dest=0, tag=tags.DONE)
        elif tag == tags.EXIT:
            break

    comm.send(None, dest=0, tag=tags.EXIT)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--mask", type=str)
    parser.add_argument("--dir", required=True, type=str)
    parser.add_argument("--trait", required=True, type=str)
    args = parser.parse_args()

    TRAIT = args.trait
    MASK = args.mask

    if rank == 0:
        print(f"Running on {comm.size} cores")
        main(args.dir)
    else:
        worker()
