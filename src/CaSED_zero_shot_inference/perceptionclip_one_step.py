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
from transformers import AutoModel, CLIPProcessor
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from src.datasets.imagenet_classnames import get_classnames
from torchvision.utils import save_image
from src.metrics.semantic_iou import SentenceIOU
from src.metrics.clustering import SemanticClusterAccuracy


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
        "--template0",
        type=str,
        default="simple_template",
        help=
        "Which prompt template is used.",
    )
    parser.add_argument(
        "--template1",
        type=str,
        default="simple_template",
        help=
        "Which prompt template is used.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-L/14",
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
        "--eval_augmentation",
        type=str,
        default="None",
        help="The type of data augmentation used for evaluation",
    )
    parser.add_argument(
        "--eval_augmentation_2",
        type=str,
        default="None",
        help="The type of data augmentation used for evaluation",
    )
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
        default='./results/CaSED_zero_shot_inference/eval_acc_ours',
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
        "--infer_mode",
        type=int,
        choices=[0, 1],
        default=0,
        help="method of inferring latent factor. 0: w/ y. 1: w/o y",
    )
    parser.add_argument(
        "--convert_text",
        type=str,
        default='object',
        help="convert template from a list of function to a list of pure text. only use it for infer_mode=1",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.,
        help="the temperature used for intervene p(z|x)",
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
        "--eval_group",
        type=bool,
        default=False,
        help="Evaluate group robustness",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="semantic_similarity",
        help="metric for evaluating the performance",
    )
    parsed_args = parser.parse_args()

    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return parsed_args

def main(args):        
    # create image transformer
    preprocess = T.transforms.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])

    # load data
    dataset_class = getattr(datasets, args.dataset)
    dataset = dataset_class(preprocess,
                            location=args.data_location,
                            batch_size=args.batch_size,
                            num_workers=args.workers)
    print(f"Eval dataset: {args.dataset}")

    #model = None

    # load template
    if args.template != "no_template":
        template_list = getattr(templates, args.template)
        template = template_list[0]
        print(f"template: {template('test')}")
    else:
        template = None
        print("Standard CaSED")

    # get classifier
    cased = AutoModel.from_pretrained("altndrr/cased", trust_remote_code=True)
    cased = cased.cuda()
    cased._old_processor = cased.processor

    if args.template != "no_template":
        def text_processor(self, text, template=template, **kwargs):
            formatted_text = [template(t) for t in text]

            return self._old_processor(text=formatted_text, **kwargs)

        cased.processor = text_processor.__get__(cased)

    sbert_model = None
    if args.metric == "semantic_similarity":
        # get sentence-BERT model
        sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        sbert_model.cuda()

    if args.eval_group:
        #acc, worst, _ = classify(model, classification_head, dataset, args, factor_head)
        pass
    else:
        #acc = classify(model, classification_head, dataset, args, factor_head)
        acc = evaluate_accuracy(cased, sbert_model, dataset, args)
    
    print(f"Avg accuracy: {acc}")

    # save result to csv
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    file_path = os.path.join(args.save_path, args.save_name + "_" + args.metric + '.csv')
    with open(file_path, mode='a') as file:
        writer = csv.writer(file)
        if args.eval_augmentation_2 != "None":
            writer.writerow(
                [args.eval_augmentation] + [args.eval_augmentation_2] + [acc])
        elif args.eval_augmentation != "None":
            writer.writerow([args.eval_augmentation] + [acc])
        else:
            if args.eval_group:
                #writer.writerow([args.template] + [acc] + [worst])
                pass
            else:
                writer.writerow([args.template] + [acc])
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

    classnames = get_classnames('openai') if 'ImageNet' in args.dataset else dataset.classnames

    if args.metric == "semantic_similarity":
        pass
    elif args.metric == "semantic_iou":
        semantic_iou = SentenceIOU()
    elif args.metric == "clustering_accuracy":
        clustering_accuracy = SemanticClusterAccuracy()

    with torch.no_grad():
        acc, sim, n = 0., 0., 0.
        start = time.time()
        for i, data in batched_data:
            
            data = maybe_dictionarize(data)
            x = data['images'].to(device)
            y = data['labels'].to(device)
            
            n += len(x)
            
            # classify
            pred = classify_batch(model, x)

            # evaluate accuracy
            y = [classnames[yi] for yi in y]          
            
            if args.metric == "semantic_similarity":
                sim += sum([sentence_similarity(text_sim_model, pred[i], y[i]) for i in range(len(pred))])
                print(f"Accuracy: {(sim / n)*100}%, batch: {i+1}/{len(dataloader)}")

            elif args.metric == "semantic_iou":
                #sim += sum([semantic_iou(pred[i], y[i]) for i in range(len(pred))])
                semantic_iou.update(pred, y)
                sim += semantic_iou.compute().item()
                print(f"Accuracy: {(sim/(i+1))*100}%, batch: {i+1}/{len(dataloader)}")
            elif args.metric == "clustering_accuracy":
                clustering_accuracy.update(pred, y)
                sim += clustering_accuracy.compute().item()
                print(f"Accuracy: {(sim/(i+1))*100}%, batch: {i+1}/{len(dataloader)}")
                
            else:
                raise ValueError(f"Unknown metric: {args.metric}")
            
        end = time.time()
        print(f"Time: {(end-start)/60} minutes")
        if args.metric == "semantic_similarity":
            acc = sim / n
        else:
            acc = sim / len(dataloader)
    return acc

def sentence_similarity(text_sim_model, predicted, ground_truth):

    predicted_embeddings = text_sim_model.encode(predicted, convert_to_tensor=True)
    ground_truth_embeddings = text_sim_model.encode(ground_truth, convert_to_tensor=True)

    similarity_score = util.pytorch_cos_sim(predicted_embeddings, ground_truth_embeddings)

    return similarity_score.item()

def semantic_iou(predicted, ground_truth):

    predicted = "".join([c for c in predicted if c.isalnum() or c == " "]).lower()
    ground_truth = "".join([c for c in ground_truth if c.isalnum() or c == " "]).lower()

    predicted = predicted.split()
    ground_truth = ground_truth.split()

    intersection = len(list(set(predicted) & set(ground_truth)))
    union = len(list(set(predicted) | set(ground_truth)))

    return intersection / union

def classify_batch(model, images):
    '''images = [torch.clamp(xi, 0, 1) for xi in images]'''
    images = [T.ToPILImage()(xi) for xi in images]
    images = model._old_processor(images=images, return_tensors="pt", padding=True)
    outputs = model(images, alpha=0.7)
    labels = list(chain(*outputs["vocabularies"]))
    pred = [labels[scores.argmax().item()] for scores in outputs["scores"]]

    return pred


if __name__ == '__main__':
    args = parse_arguments()
    print(args)
    main(args)
