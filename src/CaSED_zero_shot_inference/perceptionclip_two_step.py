import sys
from typing import Optional
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
import clip
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
        default=16,
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
        "--factors",
        default=None,
        type=lambda x: x.split(","),
    )
    parser.add_argument(
        "--main_template",
        type=str,
        default="main_template",
        help="text prompt for Y",
    )
    parser.add_argument(
        "--factor_templates",
        type=str,
        default="factor_templates",
        help="text prompt for Zs",
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

def first_step(clip_model, images, attributes):

    #image_inputs = torch.stack(images)  # Create a batch tensor
    image_inputs = images["pixel_values"]
    text_inputs = clip.tokenize(attributes)  # Tokenize text for CLIP

    image_inputs = image_inputs.to(args.device)
    text_inputs = text_inputs.to(args.device)
    # Encode the image and text features
    with torch.no_grad():
        image_features = clip_model.encode_image(image_inputs)  # Encodes all images in batch
        text_features = clip_model.encode_text(text_inputs) # Encodes all text prompts in batch

    # Compute cosine similarity between image features and text features
    similarity = image_features @ text_features.T  # Matrix multiplication for similarity

    # Get the best match for each image
    best_matches = similarity.argmax(dim=1)  # Index of best text prompt for each image

    return best_matches


def main(args):
    # load clip
    clip_model, preprocess = clip.load(args.model)
    clip_model = clip_model.cuda()

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
    if args.factors is not None:
        main_template = getattr(templates, args.main_template)
        factor_templates = getattr(templates, args.factor_templates)
        template_list = compose_template(main_template, factor_templates, args.factors)
    else:
        template_list = getattr(templates,
                            args.template)  # todo: here, we only consider the case where one factor value has only one text description. Need to extendent to multiple descriptions.'''

    print(f"infer_mode: {args.infer_mode}")
    base_text = args.convert_text
    attributes = [template(base_text) for template in template_list]
    print(f"template_list: {attributes}, len: {len(attributes)}")

    # get classifier
    cased = AutoModel.from_pretrained("altndrr/cased", trust_remote_code=True)
    cased = cased.cuda()
    cased._old_processor = cased.processor

    if args.infer_mode == 0:
        # w/ y  
        pass


    elif args.infer_mode == 1:
        # w/o y
        def forward_attr(self, images: dict, attributes = attributes, alpha: Optional[float] = None) -> torch.Tensor:
            """Forward pass.
            Args:
                images (dict): Dictionary with the images. The expected keys are:
                    - pixel_values (torch.Tensor): Pixel values of the images.
                alpha (Optional[float]): Alpha value for the interpolation.
            """
            alpha = alpha or self.hparams["alpha"]

            # forward the images
            images["pixel_values"] = images["pixel_values"].to(self.device)
            images_z = self.vision_proj(self.vision_encoder(**images)[1])
            images_z = images_z / images_z.norm(dim=-1, keepdim=True)
            attributes = [attributes[:] for _ in range(images_z.size(0))]
            vocabularies = self.get_vocabulary(images_z=images_z)

            # encode unfiltered words
            unfiltered_words = sum(vocabularies, [])
            texts_z = self._old_processor(unfiltered_words, return_tensors="pt", padding=True)
            texts_z["input_ids"] = texts_z["input_ids"][:, :77].to(self.device)
            texts_z["attention_mask"] = texts_z["attention_mask"][:, :77].to(self.device)
            texts_z = self.language_encoder(**texts_z)[1]
            texts_z = self.language_proj(texts_z)
            texts_z = texts_z / texts_z.norm(dim=-1, keepdim=True)

            # generate a text embedding for each image from their unfiltered words
            unfiltered_words_per_image = [len(vocab) for vocab in vocabularies]
            texts_z = torch.split(texts_z, unfiltered_words_per_image)
            texts_z = torch.stack([text_z.mean(dim=0) for text_z in texts_z])
            texts_z = texts_z / texts_z.norm(dim=-1, keepdim=True)

            # filter the words and embed them
            words = sum(attributes, [])
            words_z = self._old_processor(words, return_tensors="pt", padding=True)
            words_z = {k: v.to(self.device) for k, v in words_z.items()}
            words_z = self.language_encoder(**words_z)[1]
            words_z = self.language_proj(words_z)
            words_z = words_z / words_z.norm(dim=-1, keepdim=True)

            # create a one-hot relation mask between images and words
            words_per_image = [len(vocab) for vocab in attributes]
            col_indices = torch.arange(sum(words_per_image))
            row_indices = torch.arange(len(images_z)).repeat_interleave(torch.tensor(words_per_image))
            mask = torch.zeros(len(images_z), sum(words_per_image), device=self.device)
            mask[row_indices, col_indices] = 1

            # get the image and text similarities
            images_z = images_z / images_z.norm(dim=-1, keepdim=True)
            texts_z = texts_z / texts_z.norm(dim=-1, keepdim=True)
            words_z = words_z / words_z.norm(dim=-1, keepdim=True)
            images_sim = self.logit_scale * images_z @ words_z.T
            texts_sim = self.logit_scale * texts_z @ words_z.T

            # mask unrelated words
            images_sim = torch.masked_fill(images_sim, mask == 0, float("-inf"))
            texts_sim = torch.masked_fill(texts_sim, mask == 0, float("-inf"))

            # get the image and text predictions
            images_p = images_sim.softmax(dim=-1)
            texts_p = texts_sim.softmax(dim=-1)

            # average the image and text predictions
            samples_p = alpha * images_p + (1 - alpha) * texts_p

            return {"scores": samples_p, "words": words, "vocabularies": vocabularies}

        def forward(self, images: dict, alpha: Optional[float] = None) -> torch.Tensor:
            """Forward pass.
            Args:
                images (dict): Dictionary with the images. The expected keys are:
                    - pixel_values (torch.Tensor): Pixel values of the images.
                alpha (Optional[float]): Alpha value for the interpolation.
            """
            alpha = alpha or self.hparams["alpha"]

            # forward the images
            images["pixel_values"] = images["pixel_values"].to(self.device)
            images_z = self.vision_proj(self.vision_encoder(**images)[1])
            images_z = images_z / images_z.norm(dim=-1, keepdim=True)
            vocabularies = self.get_vocabulary(images_z=images_z)

            # encode unfiltered words
            unfiltered_words = sum(vocabularies, [])
            texts_z = self._old_processor(unfiltered_words, return_tensors="pt", padding=True)
            texts_z["input_ids"] = texts_z["input_ids"][:, :77].to(self.device)
            texts_z["attention_mask"] = texts_z["attention_mask"][:, :77].to(self.device)
            texts_z = self.language_encoder(**texts_z)[1]
            texts_z = self.language_proj(texts_z)
            texts_z = texts_z / texts_z.norm(dim=-1, keepdim=True)

            # generate a text embedding for each image from their unfiltered words
            unfiltered_words_per_image = [len(vocab) for vocab in vocabularies]
            texts_z = torch.split(texts_z, unfiltered_words_per_image)
            texts_z = torch.stack([text_z.mean(dim=0) for text_z in texts_z])
            texts_z = texts_z / texts_z.norm(dim=-1, keepdim=True)

            # filter the words and embed them
            vocabularies = self.vocab_transform(vocabularies)
            vocabularies = [vocab or ["object"] for vocab in vocabularies]
            filtered_words = vocabularies[:]
            words = sum(vocabularies, [])
            words_z = self.processor(filtered_words, return_tensors="pt", padding=True)
            words_z = {k: v.to(self.device) for k, v in words_z.items()}
            words_z = self.language_encoder(**words_z)[1]
            words_z = self.language_proj(words_z)
            words_z = words_z / words_z.norm(dim=-1, keepdim=True)

            # create a one-hot relation mask between images and words
            words_per_image = [len(vocab) for vocab in vocabularies]
            col_indices = torch.arange(sum(words_per_image))
            row_indices = torch.arange(len(images_z)).repeat_interleave(torch.tensor(words_per_image))
            mask = torch.zeros(len(images_z), sum(words_per_image), device=self.device)
            mask[row_indices, col_indices] = 1

            # get the image and text similarities
            images_z = images_z / images_z.norm(dim=-1, keepdim=True)
            texts_z = texts_z / texts_z.norm(dim=-1, keepdim=True)
            words_z = words_z / words_z.norm(dim=-1, keepdim=True)
            images_sim = self.logit_scale * images_z @ words_z.T
            texts_sim = self.logit_scale * texts_z @ words_z.T

            # mask unrelated words
            images_sim = torch.masked_fill(images_sim, mask == 0, float("-inf"))
            texts_sim = torch.masked_fill(texts_sim, mask == 0, float("-inf"))

            # get the image and text predictions
            images_p = images_sim.softmax(dim=-1)
            texts_p = texts_sim.softmax(dim=-1)

            # average the image and text predictions
            samples_p = alpha * images_p + (1 - alpha) * texts_p

            return {"scores": samples_p, "words": words, "vocabularies": vocabularies}

        cased._attr_forward = forward_attr.__get__(cased)
        cased.forward = forward.__get__(cased)

    # get sentence-BERT model
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
        acc = evaluate_accuracy(cased, clip_model, sbert_model, dataset, attributes, args)

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
                pass
                '''if args.factors is not None:
                    writer.writerow([args.factors] + [acc] + [worst])
                else:
                    writer.writerow([args.template] + [acc] + [worst])'''
            else:
                if args.factors is not None:
                    writer.writerow([args.factors] + [acc])
                else:
                    writer.writerow([args.template] + [acc])

    print('results saved!')

    return

def evaluate_accuracy(model, clip_model, text_sim_model, dataset, attributes, args):

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
            pred = classify_batch(model, clip_model, x, attributes)

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

def classify_batch(model, clip_model, images, attributes):
    #images = [torch.clamp(xi, 0, 1) for xi in images]
    images = [T.ToPILImage()(xi) for xi in images]
    images = model._old_processor(images=images, return_tensors="pt", padding=True)

    # extract z
    outputs = first_step(clip_model, images, attributes)
    templates = [attributes[match.item()] for match in outputs]

    # change text processor to incorporate z
    def text_processor(self, text, templ_list=templates, **kwargs):
        for i, pred_classes in enumerate (text):
            text[i] = [templates[i].replace(str(args.convert_text), c) for c in pred_classes]
        text = sum(text, [])
        return self._old_processor(text=text, **kwargs)

    model.processor = text_processor.__get__(model)

    # extract y
    outputs = model(images, alpha=0.7)
    labels = outputs["words"]
    pred = [labels[scores.argmax().item()] for scores in outputs["scores"]]
    
    return pred

'''def classify_batch(model, images, template_list):
    #images = [torch.clamp(xi, 0, 1) for xi in images]
    images = [T.ToPILImage()(xi) for xi in images]
    images = model._old_processor(images=images, return_tensors="pt", padding=True)

    # extract z
    outputs = model._attr_forward(images, alpha=0.7)
    labels = outputs["words"]
    templates = [labels[scores.argmax().item()] for scores in outputs["scores"]]

    # change text processor to incorporate z
    def text_processor(self, text, templ_list=templates, **kwargs):
        for i, pred_classes in enumerate (text):
            text[i] = [templates[i].replace(str(args.convert_text), c) for c in pred_classes]
        text = sum(text, [])
        return self._old_processor(text=text, **kwargs)

    model.processor = text_processor.__get__(model)

    # extract y
    outputs = model(images, alpha=0.7)
    labels = outputs["words"]
    pred = [labels[scores.argmax().item()] for scores in outputs["scores"]]
    
    return pred'''

if __name__ == '__main__':
    args = parse_arguments()
    print(args)
    main(args)
