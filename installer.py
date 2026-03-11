import sys
import subprocess
import os
import requests

packages_to_install = ['pygame']
user_path = os.path.expanduser("~")
folder_name = 'flappy_stuff'


def package_installer(package):
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"'{package}' has been installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while installing '{package}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        
        
def folder_maker(folder_name):
    
    try:
        folder_path = os.path.join(user_path, folder_name)
        
        os.makedirs(folder_path, exist_ok=True)
        
        print(f"Folder '{folder_name}' has been created at {folder_path}.")
        
    except Exception as e:
        print(f"An error occurred while creating the folder '{folder_name}': {e}")


if __name__ == "__main__":
    # install_packages = [package_installer(package) for package in packages_to_install]
    folder_maker(folder_name)