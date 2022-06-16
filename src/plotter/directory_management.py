import os
import shutil

from src.plotter.setup import SubdirectoryNames

start_path = "plots/SIM_PLOTS/"

def return_to_start_path():
    os.chdir("../../../")


def cd_plotting_directory(dist_scale: float, contrast_range: float):
    dir_name = f"{start_path}{get_directory_name(dist_scale, contrast_range)}"
    os.chdir(dir_name)


def cd_or_create_plotting_directory(dist_scale: float, contrast_range: float):
    dir_name = f"{start_path}{get_directory_name(dist_scale, contrast_range)}"
    # remove directory if existsd
    if os.path.isdir(dir_name):
        shutil.rmtree(dir_name)
    # create directory
    os.mkdir(dir_name)
    # enter directory
    os.chdir(dir_name)

    for subdir in SubdirectoryNames:
        os.mkdir(subdir.value)



def get_directory_name(dist_scale: float, contrast_range: float):
    return "dist-{:.4f}___contrast-{:.4f}".format(dist_scale, contrast_range)