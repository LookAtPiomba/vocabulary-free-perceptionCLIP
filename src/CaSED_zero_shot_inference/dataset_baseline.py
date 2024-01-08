import sys
import torch
import argparse
import csv
import os, time
from itertools import chain
import src.datasets as datasets
import src.templates as templates
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.zero_shot_inference.utils import *
from src.models import utils
from src.models.modeling import CLIPEncoder
from custom_cased.modeling import CustomModel
from transformers import AutoModel, CLIPProcessor
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from src.datasets.imagenet_classnames import get_classnames
from torchvision.utils import save_image


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_location",
        type=str,
        default="./datasets/data/",
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--dataset",
        default='ImageNet',
        type=str,
        help=
        "Which datasets to use for evaluation",
    )
    parser.add_argument(
        "--template",
        type=str,
        default="simple_template",
        help=
        "Which prompt template is used.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-B/32",
        help="The type of model (e.g. RN50, ViT-B/32).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
    )
    parser.add_argument("--workers",
                        type=int,
                        default=2,
                        help="Number of dataloader workers per GPU.")
    parser.add_argument(
        "--eval_augmentation_param",
        type=int,
        default=1,
        help="The parameter of data augmentation used for evaluation",
    )
    parser.add_argument(
        "--eval_trainset",
        type=bool,
        default=False,
        help="Evaluate on training set or test set",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory for caching features and encoder",
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default='tmp',
        help="Name of the csv",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default='./results/CaSED_zero_shot_inference/eval_baseline',
        help="Name of the csv",
    )
    parser.add_argument(
        "--finetuned_checkpoint",
        type=str,
        default=None,
        help="model_path",
    )
    parser.add_argument(
        "--checkpoint_mode",
        type=int,
        default=0,
        help="mode of checkpoint",
    )
    parser.add_argument(
        "--num_attrs",
        type=int,
        default=2,
        help="number of attribute values",
    )
    parser.add_argument(
        "--num_labels",
        type=int,
        default=2,
        help="number of label values",
    )
    parser.add_argument(
        "--use_simple_template",
        type=int,
        default=1,
        help="whether to use simple template: 0 indicates no prompt, 1 indicates base prompt",
    )
    parsed_args = parser.parse_args()

    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return parsed_args


def main(args):
    # load model
    model = CLIPEncoder(args, keep_lang=True)
    print(f"Model arch: {args.model}")
    if args.finetuned_checkpoint is not None:
        if args.checkpoint_mode == 0:
            finetuned_checkpoint = torch.load(args.finetuned_checkpoint)
            model.load_state_dict(finetuned_checkpoint['model_state_dict'])
        elif args.checkpoint_mode == 1:
            model.load(args.finetuned_checkpoint)
        print('finetuned model loaded.')
    model = model.cuda()

    # load data
    dataset_class = getattr(datasets, args.dataset)
    dataset = dataset_class(model.val_preprocess,
                            location=args.data_location,
                            batch_size=args.batch_size,
                            num_workers=args.workers)
    print(f"Eval dataset: {args.dataset}")

    # load template
    template = getattr(templates, args.template)

    # get classifier
    cased = AutoModel.from_pretrained("altndrr/cased", trust_remote_code=True)
    cased = cased.cuda()
    cased._old_processor = cased.processor

    if args.use_simple_template == 1:
        print('Using simple template')
        def text_processor(self, text, template=template, **kwargs):
            formatted_text = [template(t) for t in text]

            return self._old_processor(text=formatted_text, **kwargs)

        cased.processor = text_processor.__get__(cased)
    elif args.use_simple_template == 0:
        print('Using no prompt')
    
    else:
        raise ValueError('Invalide value for argument use_simple_template, should be either 0 or 1.')

    # get sentence-BERT model
    sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    sbert_model.cuda()

    # evaluate
    acc = evaluate_accuracy(cased, sbert_model, dataset, args)

    print(f"Avg accuracy: {acc}")

    # save result to csv
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    file_path = os.path.join(args.save_path, args.save_name + '.csv')
    with open(file_path, mode='a') as file:
        writer = csv.writer(file)
        if args.use_simple_template:
            writer.writerow([args.dataset + " (base prompt)"] + [acc])
        else:
            writer.writerow([args.dataset + " (no prompt)"] + [acc])

    print('results saved!')

    return

def evaluate_accuracy(model, text_sim_model, dataset, args):

    model.eval()
    dataloader = get_dataloader(dataset,
                                is_train=args.eval_trainset,
                                args=args,
                                image_encoder=None)
    batched_data = enumerate(dataloader)
    device = args.device

    classnames = get_classnames('openai')

    with torch.no_grad():
        acc, sim, n = 0., 0., 0.
        
        for i, data in batched_data:
            
            data = maybe_dictionarize(data)
            x = data['images'].to(device)
            y = data['labels'].to(device)         
            n += len(x)
            ground_truth = [classnames[yi] for yi in y]
            pred = classify_batch(model, x)
            sim += sum([sentence_similarity(text_sim_model, pred[i], ground_truth[i]) for i in range(len(pred))])
            print(f"Accuracy: {(sim / n)*100}%, batch: {i+1}/{len(dataloader)}")
            
        acc = sim / n
    return acc

def sentence_similarity(text_sim_model, predicted, ground_truth):

    predicted_embeddings = text_sim_model.encode(predicted, convert_to_tensor=True)
    ground_truth_embeddings = text_sim_model.encode(ground_truth, convert_to_tensor=True)

    similarity_score = util.pytorch_cos_sim(predicted_embeddings, ground_truth_embeddings)

    return similarity_score.item()

def classify_batch(model, images):
    x = [torch.clamp(xi, 0, 1) for xi in images]
    x = [T.ToPILImage()(xi) for xi in x]
    processed_images = model._old_processor(images=x, return_tensors="pt", padding=True)
    outputs = model(processed_images, alpha=0.7)
    labels = list(chain(*outputs["vocabularies"]))
    pred = []
    for i in range(len(images)):
        scores = outputs["scores"][i]
        max_score_index = scores.argmax().item()
        pred.append(labels[max_score_index])
    
    return pred

if __name__ == '__main__':
    args = parse_arguments()
    print(args)
    main(args)
