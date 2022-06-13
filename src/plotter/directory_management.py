import os
import shutil

from src.plotter.setup import SubdirectoryNames


def cd_plotting_directory(dist_scale: float, contrast_range: float):
    dir_name = f"../plots/{_get_directory_name(dist_scale, contrast_range)}"
    # remove directory if exists
    if os.path.isdir(dir_name):
        shutil.rmtree(dir_name)
    # create directory
    os.mkdir(dir_name)
    # enter directory
    os.chdir(dir_name)

    for subdir in SubdirectoryNames:
        os.mkdir(subdir.value)



def _get_directory_name(dist_scale: float, contrast_range: float):
    return "dist-{:.3f}___contrast-{:.3f}".format(dist_scale, contrast_range)