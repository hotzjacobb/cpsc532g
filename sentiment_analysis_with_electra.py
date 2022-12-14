# This sentiment classification benchmark is adapted from https://github.com/fajri91/discourse_probing/blob/main/nsp_choice/probe.py

import json, glob, os, random
import argparse
import logging
import numpy as np
import pandas as pd
import torch, os
from torch.autograd import Variable
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from datasets import load_dataset

logger = logging.getLogger(__name__)

args_parser = argparse.ArgumentParser()
args_parser.add_argument('--max_token_prem', default=450, help='maximum token for premise')
args_parser.add_argument('--max_token_next', default=50, help='maximum token for next sentence')
args_parser.add_argument('--batch_size', type=int, default=40, help='batch size')
args_parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
args_parser.add_argument('--weight_decay', type=int, default=0, help='weight decay')
args_parser.add_argument('--adam_epsilon', type=float, default=1e-8, help='adam epsilon')
args_parser.add_argument('--max_grad_norm', type=float, default=1.0)
args_parser.add_argument('--num_train_epochs', type=int, default=20, help='total epoch')
args_parser.add_argument('--logging_steps', type=int, default=200, help='report stats every certain steps')
args_parser.add_argument('--seed', type=int, default=2020)
args_parser.add_argument('--local_rank', type=int, default=-1)
args_parser.add_argument('--patience', type=int, default=5, help='patience for early stopping')
args_parser.add_argument('--no_cuda', default=False)
args_parser.add_argument('--model_type', type=str, default='bert', \
        choices=['bert', 'roberta', 'albert', 'electra', 'gpt2', 'bart', 't5', 't5-base', 'bert-large', 'bert-zh', 'bert-es', 'bert-de'], help='select one of language')
args_parser.add_argument('--num_layers', type=int, default=-1, help='start from number of layers')
args_parser.add_argument('--output_folder', type=str, default='output', help='output_folder')

model2hugmodel = {
    'bert': 'bert-base-uncased',
    'roberta': 'roberta-base',
    'electra': 'google/electra-base-discriminator',
}

model2layer = {
    'bert': 12,
    'roberta': 12,
    'electra': 12
}

# Map to huggingface model
args = args_parser.parse_args()
model_name = model2hugmodel[args.model_type]
args.model_name = model2hugmodel[args.model_type]
if args.num_layers == -1:
    args.num_layers = model2layer[args.model_type]

def AvgPooling(data, denominator, dim=1):
    assert len(data.size()) == 3
    assert len(denominator.size()) == 1
    sum_data = torch.sum(data, dim)
    avg_data = torch.div(sum_data.transpose(1, 0), denominator).transpose(1, 0)
    return avg_data.contiguous()


class ModelData():
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name, do_lower_case=True)
        self.cls_token = self.tokenizer.cls_token
        self.sep_token = self.tokenizer.sep_token
        self.pad_token = self.tokenizer.pad_token
        self.cls_vid = self.tokenizer.cls_token_id
        self.sep_vid = self.tokenizer.sep_token_id
        self.pad_vid = self.tokenizer.pad_token_id
        self.MAX_TOKEN_PREM = args.max_token_prem
        self.MAX_TOKEN_NEXT = args.max_token_next

    def preprocess_one(self, review, label):
        review_subtokens = self.tokenizer.tokenize(review)
        review_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(review_subtokens)
        if len(review_subtoken_idxs) > self.MAX_TOKEN_PREM:
            review_subtoken_idxs = review_subtoken_idxs[len(review_subtoken_idxs)-self.MAX_TOKEN_PREM:]
        
        return review_subtoken_idxs, label
    
    def preprocess(self, reviews, labels):
        assert len(reviews) == len(labels)
        output = []
        for idx in range(len(reviews)):
            output.append(self.preprocess_one(reviews[idx], labels[idx]))
        return output

class Batch():
    def _pad(self, data, pad_id, width=-1):
        if (width == -1):
            width = max(len(d) for d in data)
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        return rtn_data
    
    # do padding here
    def __init__(self, tokenizer, data, idx, batch_size, device):
        PAD_ID = tokenizer.pad_token_id
        cur_batch = data[idx:idx+batch_size]
        review = torch.tensor(self._pad([x[0] for x in cur_batch], PAD_ID))
        label = torch.tensor([x[1] for x in cur_batch])
        mask_review = 0 + (review != PAD_ID)
        
        self.review = review.to(device)
        self.label = label.to(device)
        self.mask_review = mask_review.to(device)

    def get(self):
        return (self.review, self.label, self.mask_review)

class Model(nn.Module):
    def __init__(self, args, device):
        super(Model, self).__init__()
        self.args = args
        self.device = device
        self.model = Model(model_name, args.num_layers)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(self.model.config.hidden_size, 1)
        self.loss = torch.nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, review, mask_review):
        batch_size = review.shape[0]
        with torch.no_grad():
            if self.args.model_type in ['electra']:
                top_vec = self.model(input_ids=review, attention_mask=mask_review)[0]
            else:
                top_vec, _ = self.model(input_ids=review, attention_mask=mask_review)
        _, _, hidden = top_vec.shape
        # bucket = Variable(torch.zeros(batch_size, 1, hidden)).type(torch.FloatTensor)
        # bucket = bucket.to(self.device)
        # top_vec = torch.cat((top_vec, bucket), 1) # batch_size, action_num + 1, hidden_size
        # top_vec = top_vec.view(batch_size * (num_token + 1), hidden)
        
        # stack_state = torch.index_select(top_vec, 0, stack_index)
        # stack_state = stack_state.view(batch_size * TOTAL_SENTENCE, -1, hidden)
        # stack_state = AvgPooling(stack_state, stack_denominator)
        # stack_state = stack_state.view(batch_size, TOTAL_SENTENCE, hidden)
        
        final_rep = self.dropout(model)
        prediction = self.linear(final_rep).squeeze()
        return prediction
    
    def get_loss(self, batch_data):
        review, label, mask_review = batch_data
        output = self.forward(review, mask_review)
        return self.loss(output, label.float())

    def predict(self, review, mask_review, label):
        prediction = self.forward(review, mask_review)
        answer = label
        return answer, prediction

def prediction(tokenizer, dataset, model, args):
    preds = []
    golds = []
    model.eval()
    for j in range(0, len(dataset), args.batch_size):
        review, label, mask_review = Batch(tokenizer, dataset, j, args.batch_size, args.device).get()
        answer, prediction = model.predict(review, mask_review, label)
        golds += answer
        preds += prediction
    return accuracy_score(golds, preds), preds


# Function for reading data
def read_data(fname):
    reviews = []
    labels = []
    data=load_dataset("rotten_tomatoes", split="train")
    for datum in data:
        context = [a.strip() for a in datum['text']]
        context = ' '.join(context)
        reviews.append(context)
        labels.append(datum['label'])
    return reviews, labels


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


# Function for train
def train(args, train_dataset, dev_dataset, test_dataset, model):
    """ Train the model """
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    t_total = len(train_dataset) // args.batch_size * args.num_train_epochs
    warmup_steps = int(0.1 * t_total)
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  LAYERS = %d", args.num_layers)
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Warming up = %d", warmup_steps)
    logger.info("  Patience  = %d", args.patience)

    # Added seed here for reproductibility
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, do_lower_case=True)
    set_seed(args)
    tr_loss = 0.0
    global_step = 1
    best_acc_dev = 0
    best_acc_test = 0
    cur_patience = 0
    for i in tqdm(range(int(args.num_train_epochs))):
        random.shuffle(train_dataset)
        epoch_loss = 0.0
        for j in range(0, len(train_dataset), args.batch_size):
            batch_data = Batch(tokenizer, train_dataset, j, args.batch_size, args.device).get()
            model.train()
            loss = model.get_loss(batch_data)
            loss = loss.sum()/args.batch_size
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            loss.backward()

            tr_loss += loss.item()
            epoch_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1
        
        logger.info("Finish epoch = %s, loss_epoch = %s", i+1, epoch_loss/global_step)
        dev_acc, dev_pred = prediction(tokenizer, dev_dataset, model, args)
        if dev_acc > best_acc_dev:
            best_acc_dev = dev_acc
            test_acc, test_pred = prediction(tokenizer, test_dataset, model, args)
            best_acc_test = test_acc
            cur_patience = 0
            logger.info("Better, BEST Acc in DEV = %s & BEST Acc in test = %s.", best_acc_dev, best_acc_test)
        else:
            cur_patience += 1
            if cur_patience == args.patience:
                logger.info("Early Stopping Not Better, BEST Acc in DEV = %s & BEST Acc in test = %s.", best_acc_dev, best_acc_test)
                break
            else:
                logger.info("Not Better, BEST Acc in DEV = %s & BEST Acc in test = %s.", best_acc_dev, best_acc_test)

    return global_step, tr_loss / global_step, best_acc_dev, best_acc_test, dev_pred, test_pred


# Setup CUDA, GPU & distributed training
if args.local_rank == -1 or args.no_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
else: # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    args.n_gpu = 1
args.device = device

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
)

# Load pretrained model and tokenizer
if args.local_rank not in [-1, 0]:
    # Make sure only the first process in distributed training will download model & vocab
    torch.distributed.barrier()

if args.local_rank == 0:
    # Make sure only the first process in distributed training will download model & vocab
    torch.distributed.barrier()

modeldata = ModelData(args)

train_data = load_dataset("rotten_tomatoes", split="train")
dev_data = load_dataset("rotten_tomatoes", split="validation")
test_data = load_dataset("rotten_tomatoes", split="test")

trainset = read_data(train_data)
devset = read_data(dev_data)
testset = read_data(test_data)
train_dataset = modeldata.preprocess(trainset[0], trainset[1])
dev_dataset = modeldata.preprocess(devset[0], devset[1])
test_dataset = modeldata.preprocess(testset[0], testset[1])

os.makedirs(args.output_folder, exist_ok = True)
output_file = args.output_folder+'/'+args.model_type+'.csv'
f = open(output_file, 'w')
f.write('layers, dev, test, dev_pred, test_pred\n')
f.close()
for idx in range(1,args.num_layers+1):
    args.num_layers = idx
    model = Model(args, device)
    # b/c of my limited compute I will just test the encoder of t5-base
    if idx > 6:
        continue
    model.to(args.device)
    global_step, tr_loss, best_acc_dev, best_acc_test, dev_pred, test_pred = train(args, train_dataset, dev_dataset, test_dataset, model)
    print('Dev set accuracy', best_acc_dev)
    print('Test set accuracy', best_acc_test)
    f = open(output_file, 'a+')
    str_dev_pred  = '|'.join([str(a) for a in dev_pred])
    str_test_pred = '|'.join([str(a) for a in test_pred])
    f.write(f"{idx}, {best_acc_dev}, {best_acc_test}, {str_dev_pred}, {str_test_pred}\n")
    f.close()

