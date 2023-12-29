from typing import Optional
from transformers import AutoModel
import torch


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
    texts_z = self.processor(unfiltered_words, return_tensors="pt", padding=True)
    texts_z["input_ids"] = texts_z["input_ids"][:, :77].to(self.device)
    texts_z["attention_mask"] = texts_z["attention_mask"][:, :77].to(self.device)
    texts_z = self.language_encoder(**texts_z)[1]
    texts_z = self.language_proj(texts_z)
    texts_z = texts_z / texts_z.norm(dim=-1, keepdim=True)

    # generate a text embedding for each image from their unfiltered words
    unfiltered_words_per_image = [len(vocab) for vocab in vocabularies]
    texts_z = torch.split(texts_z, unfiltered_words_per_image*3)
    texts_z = torch.stack([text_z.mean(dim=0) for text_z in texts_z])
    texts_z = texts_z / texts_z.norm(dim=-1, keepdim=True)

    # filter the words and embed them
    vocabularies = self.vocab_transform(vocabularies)
    vocabularies = [vocab or ["object"] for vocab in vocabularies]
    words = sum(vocabularies, [])
    words_z = self.processor(words, return_tensors="pt", padding=True)
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
