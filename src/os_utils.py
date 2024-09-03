import os
import glob

def delete_contents(path_dir: str):
    path_files = path_dir + '/*'
    files = glob.glob(path_files)
    for f in files:
        os.remove(f)
    return None

def delete_file(path_file: str):
    os.remove(path_file)
    return None

def convert_gif(path_dir_imgs:str, path_dir_output:str, file_name:str):
    cwd = os.getcwd()
    # os.system('chmod +x gifenc.sh')  # give permission to run the bash script
    os.chdir(path_dir_imgs)
    path_file = os.path.join(path_dir_output, file_name)
    command = 'ffmpeg -hide_banner -loglevel error -i %04d.png -filter_complex "[0:v] split [a][b];[a] palettegen [p];[b][p] paletteuse" ' + path_file + '.gif -y'
    os.system(command)
    os.chdir(cwd)
    return None

def prep_animation():
    path_dir_imgs = os.path.abspath('imgs/')
    path_dir_gif = os.path.abspath('results/')
    delete_contents(path_dir_imgs)  # clear imgs directory
    return path_dir_imgs, path_dir_gif
