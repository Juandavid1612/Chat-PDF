import os 
comando = "streamlit run c:/Users/juans/APP/App.py"
os.system(comando)

import subprocess

# Ruta al script que deseas iniciar
ruta_script = "App.py"

# Ejecuta el script usando subprocess
subprocess.call(["python", ruta_script])


