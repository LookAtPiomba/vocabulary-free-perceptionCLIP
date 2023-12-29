import sys
import torch
import argparse
import csv
import os
import time
import src.datasets as datasets
import src.templates as templates
from src.datasets.common import get_dataloader, maybe_dictionarize
import src.models.transformer as transformer
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
        default='./results/zero_shot_inference/eval_acc',
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
                            num_workers=args.workers)  # class_descriptions=args.class_descriptions)
    print(f"Eval dataset: {args.dataset}")

    # load template
    template_0 = getattr(templates, args.template0)
    template_1 = getattr(templates, args.template1)
    template_list = [template_0[1], template_1[1]]
    classe = "Person"
    print(f"template_list: {[template(classe) for template in template_list]}, len: {len(template_list)}")

    print(f"infer_mode: {args.infer_mode}")

    # get classifier
    cased = AutoModel.from_pretrained("altndrr/cased", trust_remote_code=True)
    cased = cased.cuda()
    cased._old_processor = cased.processor

    if args.infer_mode == 0:
        # w/ y
        def text_processor(self, text, templ_list=template_list, **kwargs):
            formatted_text = []
            for template in templ_list:
                for t in text:
                    formatted_text.append(template(t))
            return self._old_processor(text=formatted_text, **kwargs)

        cased.processor = text_processor.__get__(cased)

    elif args.infer_mode == 1:
        # w/o y
        pass

    # get sentence-BERT model
    sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    sbert_model.cuda()

    acc = evaluate_accuracy(cased, sbert_model, dataset, template_list, args)
    print(f"Avg accuracy: {acc}")

    # save result to csv
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    file_path = os.path.join(args.save_path, args.save_name + '.csv')
    with open(file_path, mode='a') as file:
        writer = csv.writer(file)
        if args.eval_augmentation_2 != "None":
            writer.writerow([args.eval_augmentation] + [args.eval_augmentation_2] +[acc])
        else:
            writer.writerow([args.eval_augmentation] + [acc])
    print('results saved!')

    return

def evaluate_accuracy(model, text_sim_model, dataset, template_list, args):

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
        start = time.time()
        for i, data in batched_data:
            #print(f"batch: {i}/{len(dataloader)}")
            data = maybe_dictionarize(data)
            x = data['images'].to(device)
            y = data['labels'].to(device)
            
            if args.eval_augmentation != "None":
                x = transformer.augment_image(args.eval_augmentation, x,
                                              args.eval_augmentation_param)
            if args.eval_augmentation_2 != "None":
                x = transformer.augment_image(args.eval_augmentation_2, x,
                                              args.eval_augmentation_param)
            
            for j, img in enumerate(x):
                
                if args.infer_mode == 1:
                    # w/o y
                    pass #TODO
                
                pred = classify_single_image(model, img)
                #similarity = sentence_similarity(text_sim_model, pred, classnames[y[j]])
                #print(f'predicted: {pred}, actual: {classnames[y[j]]} --> similarity: {similarity}')
                sim += sentence_similarity(text_sim_model, pred, classnames[y[j]])
            n += len(x)
            print(f"Accuracy: {(sim / n)*100}%, batch: {i+1}/{len(dataloader)}")
            if i == 0:
                end = time.time()
                print(f"Time elapsed for one batch: {end-start}")
        acc = sim / n

    return acc

def sentence_similarity(text_sim_model, predicted, ground_truth):

    predicted_embeddings = text_sim_model.encode(predicted, convert_to_tensor=True)
    ground_truth_embeddings = text_sim_model.encode(ground_truth, convert_to_tensor=True)

    similarity_score = util.pytorch_cos_sim(predicted_embeddings, ground_truth_embeddings)

    return similarity_score.item()

def classify_single_image(model, img):

    image_tensor = torch.clamp(img, 0, 1)
    image = T.ToPILImage()(image_tensor)
    processed_image = model._old_processor(images=[image], return_tensors="pt", padding=True)
    outputs = model(processed_image, alpha=0.5)
    labels, scores = outputs["vocabularies"][0], outputs["scores"][0]
    max_score_index = scores.argmax().item()
    # as we have 2 templates, each label is used twice we iterate initially over the first template, 
    # then the second one, so the order of the labels is the following: 
    # [template0_label0, template0_label1, ..., template1_label0, template1_label1, ...]
    labels = labels*2
    pred = labels[max_score_index]
    
    return pred

if __name__ == '__main__':
    args = parse_arguments()
    print(args)
    main(args)
