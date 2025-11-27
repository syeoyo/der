import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os

notebooks = [
    "0905_convex_hull_1.ipynb",
    "0905_convex_hull_2.ipynb",
    "0905_convex_hull_3.ipynb",
    "0905_convex_hull_4.ipynb",
    "0905_convex_hull_5.ipynb",
    "0905_convex_hull_6.ipynb",
    "0905_convex_hull_7.ipynb",
    "0905_convex_hull_8.ipynb",
    "0905_convex_hull_9.ipynb",
    "0905_convex_hull_10.ipynb"
]

ep = ExecutePreprocessor(kernel_name='python3')

for nb_file in notebooks:
    print(f"\n{'='*20}\n▶ Running {nb_file}\n{'='*20}")
    with open(nb_file, encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    try:
        ep.preprocess(nb, {'metadata': {'path': os.getcwd()}})
        with open(nb_file, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print(f"✅ {nb_file} executed successfully.")
    except Exception as e:
        print(f"❌ Error executing {nb_file}: {e}")
