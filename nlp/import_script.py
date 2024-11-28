import sys
import os

# Adiciona o caminho do diretório onde está script_a.py
caminho_diretorio_a = os.path.abspath(os.path.join(os.path.dirname(__file__), '../diretorio_a'))
sys.path.append(caminho_diretorio_a)

# Agora você pode importar script_a
import script_a

# Chamar uma função de script_a, se houver
# script_a.sua_funcao()