import os
import shutil

from src.plotter.setup import ParticSubdirectoryNames, GeneralSubdirectoryNames

start_path = "plots/SIM_PLOTS/"


def clear_simulation_directory():
    if os.listdir(start_path):
        shutil.rmtree(start_path)
        os.mkdir(start_path)


def return_to_start_path_from_general():
    os.chdir("../../../")


def cd_general_plotting_directory():
    os.chdir(f"{start_path}{GeneralSubdirectoryNames.OVERVIEWS.value}")


def cd_or_create_general_plotting_directory():
    dir_name = f"{start_path}{GeneralSubdirectoryNames.OVERVIEWS.value}"
    # remove directory if exists
    if os.path.isdir(dir_name):
        shutil.rmtree(dir_name)
    # create directory
    os.mkdir(dir_name)
    # enter directory
    os.chdir(dir_name)


def return_to_start_path_from_partic():
    os.chdir("../../../")


def cd_partic_plotting_directory(dist_scale: float, contrast_range: float):
    dir_name = f"{start_path}{get_directory_name(dist_scale, contrast_range)}"
    os.chdir(dir_name)


def cd_or_create_partic_plotting_directory(dist_scale: float, contrast_range: float):
    dir_name = f"{start_path}{get_directory_name(dist_scale, contrast_range)}"
    # remove directory if exists
    if os.path.isdir(dir_name):
        shutil.rmtree(dir_name)
    # create directory
    os.mkdir(dir_name)
    # enter directory
    os.chdir(dir_name)

    for subdir in ParticSubdirectoryNames:
        os.mkdir(subdir.value)


def get_directory_name(dist_scale: float, contrast_range: float):
    return "dist-{:.4f}___contrast-{:.4f}".format(dist_scale, contrast_range)
