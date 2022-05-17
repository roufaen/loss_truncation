import torch
import json
import bmtrain as bmt
from config import Config


class LcstsDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, split, tokenizer) -> None:
        self.datas = list()
        file = data_path + split + '.jsonl'
        bmt.print_rank(f"Start loading dataset {file}")
        with open(file, 'r') as fp:
            for i, line in enumerate(fp):
                if (i + 1) % 10000 == 0:
                    bmt.print_rank(f'Loading dataset number {i + 1}.')
                line_json = json.loads(line)
                input_ids = ([1] + tokenizer.encode(line_json['text']) + [tokenizer.eod_id] + [tokenizer.pad_id] * Config.max_source_len)[:Config.max_source_len]
                labels = ([1] + tokenizer.encode(line_json['summary']) + [tokenizer.eod_id] + [tokenizer.pad_id] * Config.max_target_len)[:Config.max_target_len]
                self.datas.append({'input_ids': input_ids, 'labels': labels})

                if (split == 'train' and i + 1 >= Config.train_size) or (split == 'valid' and i + 1 >= Config.valid_size):
                    break

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        data = self.datas[index]
        input_ids = torch.LongTensor(data['input_ids'])
        labels = torch.LongTensor(data['labels'])
        return input_ids, labels
