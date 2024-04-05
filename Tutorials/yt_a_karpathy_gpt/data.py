from pathlib import Path

import requests as r

PATH = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
DATA_PATH = Path('.data')

txt = r.get(PATH).text

with open(DATA_PATH / 'tinyshakespear.txt', 'w') as f:
    f.write(txt)
