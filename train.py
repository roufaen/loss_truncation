import torch, jieba, os, re
import bmtrain as bmt
import numpy as np

from lcsts_dataset import LcstsDataset
from loss_truncation import LossTruncator
from model_center.dataset import DistributedDataLoader
from model_center.tokenizer import CPM1Tokenizer
from model_center.model import CPM1

from config import Config
from rouge import Rouge


class FineTuneCPM:

    def __init__(self):
        bmt.init_distributed(loss_scale_factor=2, loss_scale_steps=1024)
        if Config.save_model_dir != None:
            os.makedirs(Config.save_model_dir, exist_ok=True)
        if Config.output_dir != None:
            os.makedirs(Config.output_dir, exist_ok=True)

        self.tokenizer = CPM1Tokenizer.from_pretrained(Config.model_path)
        self.model = CPM1.from_pretrained(Config.model_path)
        bmt.synchronize()
        self.optimizer = bmt.optim.AdamOffloadOptimizer(self.model.parameters())
        self.lr_scheduler = bmt.lr_scheduler.Noam(self.optimizer, start_lr=5e-4, warmup_iter=100, end_iter=None)
        bmt.synchronize()

    def prepare_dataset(self):
        splits = ['train', 'valid']
        self.datasets = {}
        for split in splits:
            self.datasets[split] = LcstsDataset(Config.data_path, split, self.tokenizer)

    def load_model(self, name):
        bmt.load(self.model, Config.save_model_dir + name + '.pt')
        self.model.load_state_dict(torch.load(Config.save_model_dir + name + '.pt'), strict=True)

    def train(self):
        loss_func = bmt.loss.FusedCrossEntropy(reduction='none')
        truncator = LossTruncator()
        best_metrics = -float('inf')
        dataloader = {
            "train": DistributedDataLoader(self.datasets['train'], batch_size=Config.batch_size, shuffle=True),
            "valid": DistributedDataLoader(self.datasets['valid'], batch_size=1, shuffle=False),
        }

        for epoch_num in range(5):
            self.model.train()
            truncator.reset()
            bmt.print_rank('Train epoch ' + str(epoch_num) + '.')
            for iter_num, [input_ids, labels] in enumerate(dataloader['train']):
                self.optimizer.zero_grad()
                input_ids, labels = input_ids.cuda(), labels.cuda()

                input_tokens = torch.cat((input_ids, labels), dim=1)
                input_length = torch.Tensor([input_tokens.shape[1]] * input_tokens.shape[0]).cuda()
                input_context = torch.cat((torch.ones_like(input_ids), torch.zeros_like(labels)), dim=1)
                input_span = torch.zeros_like(input_context)
                logits = self.model(input_tokens, input_length, input_context, input_span)
                target = torch.cat([labels[:, 1:], torch.full((labels.shape[0], 1), fill_value=self.tokenizer.pad_id).cuda()], dim=1)

                loss = loss_func(logits[:, Config.max_source_len:, :].contiguous().view(-1, logits.shape[-1]), target.view(-1))
                if Config.loss_truncation == True:
                    loss = truncator.truncate_loss(loss)
                loss = loss.mean()
                loss.backward()
                bmt.optim.clip_grad_norm(self.optimizer.param_groups, Config.max_grad_norm, scale=self.optimizer.scale)
                bmt.optim_step(self.optimizer, self.lr_scheduler)

                if (iter_num + 1) % Config.save_steps == 0:
                    bmt.save(self.model, Config.save_model_dir + 'training_' + str(epoch_num) + '_' + str(iter_num) + '.pt')
                if (iter_num + 1) % Config.train_log_steps == 0:
                    bmt.print_rank('Traing iter ' + str(iter_num + 1) + '.')

            self.model.eval()
            with torch.no_grad():
                inputs, outputs, refs = list(), list(), list()
                bmt.print_rank('Validate epoch ' + str(epoch_num) + '.')
                for iter_num, [input_ids, labels] in enumerate(dataloader['valid']):
                    input_ids, labels = input_ids.cuda(), labels.cuda()
                    preds = torch.IntTensor([1]).cuda()
                    for _ in range(Config.max_target_len - 1):
                        preds_unsqueeze = preds.unsqueeze(0)
                        input = torch.cat((input_ids, preds_unsqueeze), dim=1)
                        length = torch.Tensor([input.shape[1]]).cuda()
                        context = torch.cat((torch.ones_like(input_ids), torch.zeros_like(preds_unsqueeze)), dim=1)
                        span = torch.zeros_like(context)
                        logit = self.model(input, length, context, span)
                        logit = logit.squeeze(0)[-1, :].float()  # (vocab_size)
                        pred = torch.argmax(logit)
                        preds = torch.cat((preds, pred.unsqueeze(0)), dim=0)

                    inputs.append(self.tokenizer.decode(input_ids.squeeze(0).cpu().tolist()) + '\n')
                    outputs.append(self.tokenizer.decode(preds.cpu().tolist()) + '\n')
                    refs.append(self.tokenizer.decode(labels.squeeze(0).cpu().tolist()) + '\n')
                    if (iter_num + 1) % Config.validation_log_steps == 0:
                        bmt.print_rank('Validating iter ' + str(iter_num + 1) + '.')

                with open(Config.output_dir + str(bmt.rank()) + '_' + str(epoch_num) + '_inputs.txt', 'w') as fp:
                    fp.writelines(inputs)
                with open(Config.output_dir + str(bmt.rank()) + '_' + str(epoch_num) + '_outputs.txt', 'w') as fp:
                    fp.writelines(outputs)
                with open(Config.output_dir + str(bmt.rank()) + '_' + str(epoch_num) + '_refs.txt', 'w') as fp:
                    fp.writelines(refs)

                global_metrics = bmt.sum_loss(torch.Tensor([self.rouge_score(outputs, refs)]).cuda(), method='sum').item()
                if global_metrics > best_metrics:
                    bmt.print_rank('Update metrics:' + str(best_metrics) + ' -> ' + str(global_metrics) + '.')
                    best_metrics = global_metrics
                    bmt.save(self.model, Config.save_model_dir + 'best_' + str(epoch_num) + '.pt')

    def rouge_score(self, preds, refs):
        res = list()
        for pred, ref in zip(preds, refs):
            pred = re.sub('<\w+>', '', pred)
            ref = re.sub('<\w+>', '', ref)
            pred = " ".join([w for w in jieba.cut("".join(pred.strip()))])
            ref = " ".join([w for w in jieba.cut("".join(ref.strip()))])
            score = Rouge().get_scores(refs=ref, hyps=pred)[0]
            res.extend([score['rouge-1']['f'], score['rouge-2']['f'], score['rouge-l']['f']])

        return np.mean(res) * 3


if __name__ == "__main__":
    fine_tune_cpm = FineTuneCPM()
    fine_tune_cpm.prepare_dataset()
    fine_tune_cpm.train()
