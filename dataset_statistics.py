import json
from matplotlib import pyplot as plt
import numpy as np


if __name__ == '__main__':
    len_of_text, len_of_summary = list(), list()
    for file in ['../datasets/lcsts/train.jsonl', '../datasets/lcsts/valid.jsonl']:
        with open(file, 'r') as fp:
            for line in fp.readlines():
                line_json = json.loads(line)
                len_of_text.append(len(line_json['text']))
                len_of_summary.append(len(line_json['summary']))
    len_of_text, len_of_summary = np.array(len_of_text), np.array(len_of_summary)
    print(len_of_text.mean(), len_of_text.max(), len_of_summary.mean(), len_of_summary.max())

    plt.cla()
    plt.hist(len_of_text, bins=50, range=(0, 160), rwidth=0.8)
    plt.title('len_of_text')
    plt.savefig('len_of_text.png')
    plt.cla()
    plt.hist(len_of_summary, bins=50, range=(0, 40), rwidth=0.8)
    plt.title('len_of_summary')
    plt.savefig('len_of_summary.png')
