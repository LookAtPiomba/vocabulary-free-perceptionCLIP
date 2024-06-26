import torch
from tqdm import tqdm
import random
import string
import clip.clip as clip

from src.models.modeling import ClassificationHead
from itertools import product

def get_zeroshot_classifier(args, clip_model, classnames, template):
    """
    Calculate text embeddings before training and save them as a fixed linear layer.
    For multi-template cases, average the text embeddings.
    """
    logit_scale = clip_model.logit_scale

    device = args.device
    clip_model.eval()
    clip_model.to(device)

    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = []
            for t in template:
                texts.append(t(classname))
            texts = clip.tokenize(texts).to(device)  # tokenize
            embeddings = clip_model.encode_text(texts)  # embed with text encoder
            embeddings /= embeddings.norm(dim=-1, keepdim=True)  # normalize text embedding

            # average text embeddings derived from multi-templates
            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings /= embeddings.norm()  # normalize text embedding

            zeroshot_weights.append(embeddings)

        zeroshot_weights = torch.cat(zeroshot_weights, dim=0).to(device)
        zeroshot_weights *= logit_scale.exp()

    classification_head = ClassificationHead(normalize=True,
                                             weights=zeroshot_weights)

    return classification_head


def get_zeroshot_classifier_flat(args, clip_model, classnames, template):
    """
    Calculate text embeddings before training and save them as a fixed linear layer.
    For multi-template cases, not average the text embeddings.
    """
    logit_scale = clip_model.logit_scale

    device = args.device
    clip_model.eval()
    clip_model.to(device)

    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = []
            for t in template:
                texts.append(t(classname))
            texts = clip.tokenize(texts).to(device)  # tokenize
            embeddings = clip_model.encode_text(texts)  # embed with text encoder
            embeddings /= embeddings.norm(dim=-1, keepdim=True)  # normalize text embedding

            zeroshot_weights.append(embeddings)

        zeroshot_weights = torch.cat(zeroshot_weights, dim=0).to(device)
        zeroshot_weights *= logit_scale.exp()

    classification_head = ClassificationHead(normalize=True,
                                             weights=zeroshot_weights)

    return classification_head


def get_zeroshot_classifier_puretext(args, clip_model, template_texts):
    """
    Calculate text embeddings before training and save them as a fixed linear layer.
    Use pure text templates, not consider classes.
    Use for infer generative factor values.
    """

    device = args.device
    clip_model.eval()
    clip_model.to(device)
    logit_scale = clip_model.logit_scale

    with torch.no_grad():
        zeroshot_weights = []
        for text in template_texts:
            text = clip.tokenize(text).to(device)
            embeddings = clip_model.encode_text(text)  # embed with text encoder
            embeddings /= embeddings.norm(dim=-1, keepdim=True)  # normalize text embedding
            zeroshot_weights.append(embeddings)

        zeroshot_weights = torch.cat(zeroshot_weights, dim=0).to(device)
        zeroshot_weights *= logit_scale.exp()

    classification_head = ClassificationHead(normalize=True,
                                             weights=zeroshot_weights)

    return classification_head


def get_zeroshot_classifier_puretext_advance(args, clip_model, template_texts_list):
    """
    Calculate text embeddings before training and save them as a fixed linear layer.
    Use pure text templates, not consider classes.
    Use for infer generative factor values.
    template_texts_list contains a list of factor values for one factor.
    Every factor value has multi-descriptions.
    """

    device = args.device
    clip_model.eval()
    clip_model.to(device)
    logit_scale = clip_model.logit_scale

    with torch.no_grad():
        zeroshot_weights = []
        for template_texts in template_texts_list:
            texts = template_texts
            texts = clip.tokenize(texts).to(device)
            embeddings = clip_model.encode_text(texts)  # embed with text encoder
            embeddings /= embeddings.norm(dim=-1, keepdim=True)  # normalize text embedding

            # average text embeddings derived from multi-templates
            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings /= embeddings.norm()  # normalize text embedding

            zeroshot_weights.append(embeddings)

        zeroshot_weights = torch.cat(zeroshot_weights, dim=0).to(device)
        zeroshot_weights *= logit_scale.exp()

    classification_head = ClassificationHead(normalize=True,
                                             weights=zeroshot_weights)

    return classification_head


def get_zeroshot_classifier_flat_advance(args, clip_model, classnames, template_list):
    """
    Calculate text embeddings before training and save them as a fixed linear layer.
    Use for infer generative factor values.
    template_list contains a list of factor values for one factor.
    Every factor value has multi-descriptions.
    """
    logit_scale = clip_model.logit_scale

    device = args.device
    clip_model.eval()
    clip_model.to(device)

    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            for template in template_list:
                texts = []
                for t in template:
                    texts.append(t(classname))
                texts = clip.tokenize(texts).to(device)  # tokenize
                embeddings = clip_model.encode_text(texts)  # embed with text encoder
                embeddings /= embeddings.norm(dim=-1, keepdim=True)  # normalize text embedding
                # average text embeddings derived from multi-templates
                embeddings = embeddings.mean(dim=0, keepdim=True)
                embeddings /= embeddings.norm()  # normalize text embedding

                zeroshot_weights.append(embeddings)

        zeroshot_weights = torch.cat(zeroshot_weights, dim=0).to(device)
        zeroshot_weights *= logit_scale.exp()

    classification_head = ClassificationHead(normalize=True,
                                             weights=zeroshot_weights)

    return classification_head


def template_convert(template, convert_text):
    """
    Convert template to pure texts.
    Example for delete_text is ' of a '
    """
    return [func(convert_text) for func in template]


def randomize_text(text):
    """
    Replace each word with random letters and keep the word length unchanged
    """
    text = text.split()  # convert a string to a list of words
    randomized_words = []
    for word in text:
        randomized_word = ''.join(
                random.choices(string.ascii_letters + string.digits, k=len(word)))
        randomized_words.append(randomized_word)
    randomized_text = ' '.join(randomized_words)
    return randomized_text


def randomize_function(template):
    sentence = template("")

    if ',' not in sentence:
        return template

    # select the part between "," and "."
    first_part, rest = sentence.split(',', 1)
    rest_start, second_part = rest.split('.', 1)

    # randomize
    randomized_descrip = randomize_text(rest_start)

    def new_function(c):
        original_string = template(c)
        first_part, _ = original_string.rsplit(',', 1)
        return f"{first_part}, {randomized_descrip}."

    return new_function


def randomize_puretext(template):

    if ',' not in template:
        return template

    # select the part between "," and "."
    first_part, rest = template.split(',', 1)
    rest_start, second_part = rest.split('.', 1)

    # randomize
    randomized_descrip = randomize_text(rest_start)

    return f"{first_part}, {randomized_descrip}. {second_part}"


def randomize_template(template_list):
    randomized_template = []
    for template in template_list:
        if isinstance(template, str):
            randomized_template.append(randomize_puretext(template))
        else:
            randomized_template.append(randomize_function(template))
    return randomized_template


def group_accuracy(args, output, target, attribute):
    with torch.no_grad():
        batch_size = target.size(0)
        exact_acc = output.eq(target.view_as(output)).squeeze()
        group_correct = torch.zeros((args.num_attrs, args.num_labels))
        group_cnt = torch.zeros((args.num_attrs, args.num_labels))
        for g in range(args.num_attrs):
            for y in range(args.num_labels):
                group_exact_acc = exact_acc[(attribute == g) * (target == y)]
                group_correct[g, y] = group_exact_acc.sum()
                group_cnt[g, y] = len(group_exact_acc)
        if batch_size != group_cnt.sum().item():
            err_msg = "Errors in computing group accuracy!"
            raise ValueError(err_msg)

    return group_correct, group_cnt


def compose_template(org_templates, factor_templates, selected_factors):
    selected_templates = [factor_templates[category] for category in selected_factors if
                          category in factor_templates]
    new_templates = []
    for factors in product(*selected_templates):
        new_templates.append(
            lambda c, main=org_templates[0], factors=factors: main(c) + ''.join(factors) + "."
        )

    return new_templates
