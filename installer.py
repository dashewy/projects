import subprocess
import os
import sys
import re
import importlib.util



packages_to_install = ['pygame', 'requests']
user_path = os.path.expanduser("~")
folder_name = 'flappy_stuff'
file_pattern = re.compile('flappy.*\.p')
owner = 'dashewy'
repo = 'projects'
repo_url = f'https://api.github.com/repos/{owner}/{repo}/contents/'

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
        
    return folder_path


install_packages = [package_installer(package) for package in packages_to_install]
new_path = folder_maker(folder_name)
# clear cache to ensure new packages are recognized
importlib.invalidate_caches()

try:
    import requests   
    response = requests.get(repo_url)

    if response.status_code == 200:

        contents = response.json()
        
        for item in contents:
            if item['type'] == 'file' and file_pattern.match(item['name']):
                file_name = item['name']
                download_url = item['download_url']

                local_path = os.path.join(new_path, file_name)
                
                file_data = requests.get(download_url).content
                with open(local_path, 'wb') as f:
                    f.write(file_data)
                
    else:
        print(f"Failed to fetch repository contents. Status code: {response.status_code}")
        
except ImportError:
    print("Failed to import 'requests' after installation.")


def alias_machine():
    
    alias = f'alias flappy="python3 {new_path}/flappy.py"' 
    zshrc_path = os.path.expanduser('~/.zshrc')
    
    if sys.platform == 'darwin':
        with open(zshrc_path, 'a+') as f:
            f.seek(0)
            content = f.read()
            if alias not in content:
                f.write(f'\n{alias}\n')
                print("Alias has been added to .zshrc.")
            
    else:
        print("Alias setup is currently only supported on macOS with zsh.")
        return

alias_machine()