import os
import torch
import fitlog
import logging
import numpy as np
import bert_score
from nltk.translate.bleu_score import sentence_bleu


from tqdm import tqdm, trange
from sklearn.metrics import f1_score, accuracy_score

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    AutoTokenizer,
)

logger = logging.getLogger(__name__)

tokenizer = AutoTokenizer.from_pretrained(
	"facebook/bart-base",
	cache_dir=None,
	use_fast=True)


def train(train_dataloader, eval_dataloader, test_dataloader, model, training_args, other_args):
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    best_score = 0

    steps_per_epoch = len(train_dataloader)

    # total number of training steps
    num_train_steps = int(steps_per_epoch * training_args.num_train_epochs)
    t_total = num_train_steps

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": training_args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate, eps=training_args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=training_args.warmup_ratio * t_total, num_training_steps=t_total)

    # multi-gpu training
    if training_args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    global_step = 0

    for epoch in trange(int(training_args.num_train_epochs), desc="Epoch"):

        training_steps = 0
        model.zero_grad()

        for data in tqdm(train_dataloader, desc="Iteration", smoothing=0.05):
            model.train()
            outputs = model(**data)
            loss, ce_loss, cl_loss, gen_loss = outputs.loss, outputs.ce_loss, outputs.cl_loss, outputs.gen_loss

            if training_args.n_gpu > 1:
                loss = loss.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_args.max_grad_norm)

            optimizer.step()
            scheduler.step()

            optimizer.zero_grad()

            training_steps += 1
            global_step += 1

            if training_args.logging_steps > 0 and global_step % training_args.logging_steps == 0:

                fitlog.add_loss(loss, name="Loss", step=global_step)
                fitlog.add_loss(ce_loss, name="CE_Loss", step=global_step)
                fitlog.add_loss(cl_loss, name="CL_Loss", step=global_step)
                fitlog.add_loss(gen_loss, name="Gen_Loss", step=global_step)

                results = evaluate(training_args, other_args, eval_dataloader, model, "evaluate")
                torch.cuda.empty_cache()
                res_for_display = {}
                for k, v in results.items():
                    res_for_display[k.replace("-", "_")] = v
                fitlog.add_metric({"dev": res_for_display}, step=global_step)
                if other_args.task_name in ['MELD', 'IEMOCAP', 'EmoryNLP']:
                    eval_metrics = 'macro_f1'
                else:
                    eval_metrics = 'micro_f1'
                if results[eval_metrics] > best_score:
                    best_score = results[eval_metrics]
                    fitlog.add_best_metric({"dev": {eval_metrics: best_score}})

                    # save the best model
                    output_dir = os.path.join(training_args.output_dir, "best_model_%d" % training_args.seed)
                    model_to_save = (model.module if hasattr(model, "module") else model)  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    logger.info("Saving model checkpoint to %s", output_dir)

                    results = evaluate(training_args, other_args, test_dataloader, model, "predict")
                    fitlog.add_metric({"test": {'macro_f1': results['macro_f1']}}, step=global_step)
                    fitlog.add_metric({"test": {'micro_f1': results['micro_f1']}}, step=global_step)

                    fitlog.add_best_metric({"test": {'macro_f1': results['macro_f1']}})
                    fitlog.add_best_metric({"test": {'micro_f1': results['micro_f1']}})

        torch.cuda.empty_cache()


def evaluate(training_args, other_args, eval_loader, model, eval_or_test):
    def compute_acc_for_categories(preds, labels):
        categories_count = {"label_%s" % i: 0 for i in range(other_args.num_labels)}
        categories_right = {"label_%s" % i: 0 for i in range(other_args.num_labels)}
        categories_acc = {}
        for pred, label in zip(preds, labels):
            categories_count["label_%s" % label] += 1
            if pred == label:
                categories_right["label_%s" % label] += 1
        for index, (key, value) in enumerate(categories_count.items()):
            categories_acc["label_%s" % index] = format(categories_right["label_%s" % index] / value, '.4f')
        print(categories_acc)
        return categories_acc

    def compute_metrics(preds_id, labels_id):
        results = {}

        # -------------- eval classification --------------
        accuracy = round(accuracy_score(labels_id, preds_id) * 100, 4)
        if other_args.task_name in ['MELD', 'EmoryNLP', 'IEMOCAP']:
            macro_f1 = f1_score(labels_id, preds_id, labels=[i for i in range(other_args.num_labels)], average='weighted')
            micro_f1 = f1_score(labels_id, preds_id, labels=[i for i in range(other_args.num_labels)], average='micro')
        else:
            macro_f1 = f1_score(labels_id, preds_id, labels=[i for i in range(1, other_args.num_labels)], average='weighted')
            micro_f1 = f1_score(labels_id, preds_id, labels=[i for i in range(1, other_args.num_labels)], average='micro')
        results['acc'] = accuracy
        results['macro_f1'] = round(macro_f1 * 100, 4)
        results['micro_f1'] = round(micro_f1 * 100, 4)

        return results

    results = {}

    if not os.path.exists(training_args.output_dir) and training_args.local_rank in [-1, 0]:
        os.makedirs(training_args.output_dir)

    # training_args.eval_batch_size = training_args.per_device_eval_batch_size * max(1, training_args.n_gpu)
    # Note that DistributedSampler samples randomly

    # multi-gpu eval
    if training_args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running %s *****" % eval_or_test)
    logger.info("  Num examples = %d", len(eval_loader.dataset))
    logger.info("  Batch size = %d", training_args.eval_batch_size)
    # eval_loss = 0.0

    all_preds, all_labels = [], []
    bleu_sum = 0
    bleu_count = 0
    bert_count = 0
    a = torch.zeros(1)
    bert_sum = [a,a,a]
    for batch in tqdm(eval_loader, desc=eval_or_test):
        model.eval()
        batch = tuple(v.to(training_args.device) for _, v in batch.items())

        with torch.no_grad():
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'speakers': batch[3], 'next_input_ids': batch[4], 'next_attention_mask': batch[5]} #####next 붙은것들 수정
            labels = batch[2]
###################
            lines = batch[4]
            lines = lines[lines.ne(-100)].cpu().numpy()
            decoded_lines = tokenizer.decode(lines)
            #print('decoded_lines ', decoded_lines)
            remove_pad = decoded_lines.replace('<pad>', '')
            splitted_lines = remove_pad.split("</s>")
            del splitted_lines[-1]
            for i in range(len(splitted_lines)):
             processing_lines = splitted_lines[i]
             processing_lines = processing_lines[processing_lines.find(':')+2:]
             splitted_lines[i] = processing_lines
           
            nlines = batch[0]
            nlines = nlines[nlines.ne(-100)].cpu().numpy()
            ndecoded_lines = tokenizer.decode(nlines)
            #print('ndecoded_lines ', ndecoded_lines)
            nremove_pad = ndecoded_lines.replace('<pad>', '')
            nsplitted_lines = nremove_pad.split("</s>")
            del nsplitted_lines[-1]
            for i in range(len(nsplitted_lines)):
             nprocessing_lines = nsplitted_lines[i]
             nprocessing_lines = nprocessing_lines[nprocessing_lines.find(':')+2:]
             nsplitted_lines[i] = nprocessing_lines
##################

            labels = labels[labels.ne(-100)].cpu().numpy()

            outputs = model(**inputs)
            #preds = outputs.cls_logits
            #preds = torch.argmax(preds, dim=-1)
            gen_logits = outputs.logits
            gen_logits = torch.argmax(gen_logits, dim = -1)
            j = 0
            for i in gen_logits:
             savingstring = tokenizer.decode(i)
             #print('savingstring', savingstring)
             savingstring = savingstring.replace('<pad>', '').replace('</s>','').replace('<s>','')
             savingstring = savingstring[savingstring.find(':')+2:]
             savingstring = savingstring.replace(': ','')
             #print('before splitted lines: ', splitted_lines[j])
             #print('before saving string: ', savingstring)
             #print(splitted_lines[j])
             #print(savingstring)
             context_splitted = nsplitted_lines[j].split()
             splitted_splitted = splitted_lines[j].split()
             splitted_saving = savingstring.split()
             #print('splitted_lines(Original) ', splitted_splitted)
             #print('savingstring(Predicted) ', splitted_saving)
             reference = [splitted_splitted]
             #print('reference', reference)
             print('Past Context', [nsplitted_lines[j]])
             print('Original', [splitted_lines[j]])
             print('Generated', [savingstring])
             BERT_score = bert_score.score([splitted_lines[j]], [savingstring], model_type = 'distilbert-base-uncased')
             BERT_score = list(BERT_score)
             print('BERT_score', BERT_score)
             bert_sum[0] = bert_sum[0] + BERT_score[0]
             bert_sum[1] = bert_sum[1] + BERT_score[1]
             bert_sum[2] = bert_sum[2] + BERT_score[2]
             print(bert_sum)
             bert_count += 1
             j += 1
             if len(splitted_splitted) <= 4 or len(splitted_saving) <= 4:
              continue
             BLEU_score = sentence_bleu([splitted_splitted], splitted_saving)

             print('BLEU score = ', BLEU_score)
             bleu_sum += BLEU_score
             bleu_count += 1





            #preds = preds.cpu().numpy()
            #all_labels.append(labels)
            #all_preds.append(preds)

    #all_preds = np.concatenate(all_preds, axis=0)
    #all_labels = np.concatenate(all_labels, axis=0)

    #correct_num = np.sum(all_preds == all_labels)

    # eval_loss = eval_loss / nb_eval_steps
    #result = compute_metrics(all_preds, all_labels)
    #results.update(result)
    logger.info("***** %s results *****" % eval_or_test)
    #for key in sorted(result.keys()):
    #    logger.info("  %s = %s", key, str(result[key]))
    #    print("  %s = %s" % (key, str(result[key])))
    #logger.info("Correct / Total num = ({}/{})".format(correct_num, len(all_labels)))
    final_bleu = bleu_sum/bleu_count
    bert_sum[0] = bert_sum[0]/bert_count
    bert_sum[1] = bert_sum[1]/bert_count
    bert_sum[2] = bert_sum[2]/bert_count
    print('final_bert: ', bert_sum)
    print('final_bleu: ', final_bleu)
    return results
