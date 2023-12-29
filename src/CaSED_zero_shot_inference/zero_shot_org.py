import torch
import argparse
import os
import csv
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
        default=256,
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
        default='./results/CaSED_zero_shot_inference/eval_acc',
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
        "--random_descrip",
        type=bool,
        default=False,
        help="randomize the description of the factors or not",
    )

    parsed_args = parser.parse_args()

    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return parsed_args

def sentence_similarity(text_sim_model, predicted, ground_truth):
    predicted_embeddings = text_sim_model.encode(predicted, convert_to_tensor=True)
    ground_truth_embeddings = text_sim_model.encode(ground_truth, convert_to_tensor=True)

    similarity_score = util.pytorch_cos_sim(predicted_embeddings, ground_truth_embeddings)

    return similarity_score.item()


def evaluate_accuracy(model, text_sim_model, dataset, args):

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
            print(f"batch: {i}/{len(dataloader)}")
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
                image_tensor = torch.clamp(img, 0, 1)
                #permute = [2, 1, 0]
                #image_tensor = image_tensor[:, permute]
                image = T.ToPILImage()(image_tensor)
                if j == 0 and i == 0:
                    print(f'type {type(x)}')
                    print(type(img))
                    image.save('clamp_test.png')

                logits = model._old_processor(images=[image], return_tensors="pt", padding=True)
                outputs = model(logits, alpha=0.5)
                labels, scores = outputs["vocabularies"][0], outputs["scores"][0]
                max_score_index = scores.argmax().item()
                pred = labels[max_score_index]
                #similarity = sentence_similarity(text_sim_model, pred, classnames[y[j]])
                #print(f'predicted: {pred}, actual: {classnames[y[j]]} --> similarity: {similarity}')
                sim += sentence_similarity(text_sim_model, pred, classnames[y[j]])
                #correct += 1 if pred == classnames[y[j]] else 0
                break
            n += len(x)
            print(f"Accuracy: {(sim / n)*100}%, batch: {i+1}/{len(dataloader)}")
            break
        acc = sim / n

    return acc

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

    # get template
    assert args.template is not None
    template = getattr(templates, args.template)
    if args.random_descrip:
        template = randomize_template(template)

    # create classifier
    cased = AutoModel.from_pretrained("altndrr/cased", trust_remote_code=True)
    cased = cased.cuda()
    print(cased)

    template = template[0] if args.template == "simple_template" else template[1]

    def text_processor(self, text, templ=template, **kwargs):
        text = [templ(t) for t in text]
        return self._old_processor(text=text, **kwargs)

    cased._old_processor = cased.processor
    cased.processor = text_processor.__get__(cased)

    # get sentence-BERT model
    sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    sbert_model.cuda()

    # print template for debugging
    print(template)                   

    # eval accuracy
    acc = evaluate_accuracy(cased, sbert_model, dataset, args)
    print(f"Avg accuracy: {acc}")

    # save results
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    file_path = os.path.join(args.save_path, args.save_name + '.csv')
    with open(file_path, mode='a') as file:
        writer = csv.writer(file)
        if args.eval_augmentation_2 != "None":
            writer.writerow([args.eval_augmentation] + [args.eval_augmentation_2] + [acc])
        else:
            writer.writerow([args.eval_augmentation] + [acc])
    print('results saved!')

    return

def test_main(args):
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
    #model = model.cuda()

    # load data
    dataset_class = getattr(datasets, args.dataset)
    dataset = dataset_class(model.val_preprocess,
                            location=args.data_location,
                            batch_size=args.batch_size,
                            num_workers=args.workers)
    print(f"Eval dataset: {args.dataset}")

    # get template
    assert args.template is not None
    template = getattr(templates, args.template)
    if args.random_descrip:
        template = randomize_template(template)
    
    dataloader = get_dataloader(dataset,
                                is_train=args.eval_trainset,
                                args=args,
                                image_encoder=None)
    batched_data = enumerate(dataloader)
    device = "cpu"

    transform = T.ToPILImage()

    classnames = get_classnames('openai')

    with torch.no_grad():
        acc, correct, n = 0., 0., 0.
        for i, data in batched_data:
            print(f"batch: {i}/{len(dataloader)}")
            data = maybe_dictionarize(data)
            x = data['images'].to(device)
            y = data['labels'].to(device)
            
            if args.eval_augmentation != "None":
                x = transformer.augment_image(args.eval_augmentation, x,
                                              args.eval_augmentation_param)
            if args.eval_augmentation_2 != "None":
                x = transformer.augment_image(args.eval_augmentation_2, x,
                                              args.eval_augmentation_param)
            
            if x[0].dtype == torch.uint8:
                print("Tensor is uint8")
                # Convert the uint8 tensor to a NumPy array
                image_array = x[0].numpy()

                # Create a PIL Image from the NumPy array
                image = Image.fromarray(image_array)

                # Save the image
                image.save("uint8_image.png")
            elif x[0].dtype == torch.float32 or x[0].dtype == torch.float64:
                print(f"Tensor is float: {x[0].dtype}")
                print(f"Tensor shape: {x[0].shape}")
                # Convert the float tensor to a NumPy array
                #save_image(x[0], "float_image00.png")
                image_tensor = torch.clamp(x[0], 0, 1)

                save_image(image_tensor, "float_image0.png")

                print(f"Tensor is float: {image_tensor.dtype}")
                print(f"Tensor shape: {image_tensor.shape}")

                image = T.ToPILImage()(image_tensor)

                # Save the PIL Image to a file
                
                image.save("float_image.png")
                
            else:
                print("Tensor is of another type")
            
            #x = [TF.to_pil_image(image) for image in x]
            break

if __name__ == '__main__':
    args = parse_arguments()
    print(args)
    main(args)
