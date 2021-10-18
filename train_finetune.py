import torch
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from data.text_image_dm import TextImageDataModule
from models import CustomCLIPWrapper
from torchvision.models import resnet50
from transformers import AutoTokenizer, AutoModel

from models.state import merge_fine_tune_CLIP_into_CLIP_ViL, extract_CLIP_from_CLIP_ViL
from models.simple_tokenizer import SimpleTokenizer as _Tokenizer


def main(hparams):
    img_encoder = resnet50(pretrained=True)
    img_encoder.fc = torch.nn.Linear(2048, 768)

    tokenizer = AutoTokenizer.from_pretrained("johngiorgi/declutr-sci-base")
    txt_encoder = AutoModel.from_pretrained("johngiorgi/declutr-sci-base")

    if hparams.minibatch_size < 1:
        hparams.minibatch_size = hparams.batch_size

    img_encoder = None
    txt_encoder = None
    model = CustomCLIPWrapper(img_encoder, txt_encoder, hparams.minibatch_size, avg_word_embs=True, model_name=hparams.model_name)

    if hparams.merge_base_clip_into_clip_vil:
        # This just embeds the above loaded clip into CLIP-ViL so it can be tested
        # in the VQA pipeline.
        assert hparams.load_checkpoint is not None
        assert hparams.save_path is not None
        merge_fine_tune_CLIP_into_CLIP_ViL(hparams.load_checkpoint, model, hparams.save_path)
        return
    else:
        # Load CLIP using weights from the CLIP-ViL checkpoint.
        state_dict = extract_CLIP_from_CLIP_ViL(hparams.load_checkpoint, model.model)
        tokenizer = _Tokenizer()
        model = CustomCLIPWrapper(img_encoder, txt_encoder, hparams.minibatch_size, avg_word_embs=True, model_name=hparams.model_name, state_dict=state_dict)
        # Need to populate the missing params here from config...
        hparams.image_size = model.model.visual.input_resolution # Fixes issue with previously commented out self.attnpool in ModifiedRes
        dm = TextImageDataModule.from_argparse_args(hparams, custom_tokenizer=tokenizer)
        trainer = Trainer.from_argparse_args(hparams, precision=16, max_epochs=32)
        trainer.fit(model, dm)
        # TODO: Once fine-tuned, merge_fine_tune_CLIP_into_CLIP_ViL and save.


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--minibatch_size', type=int, default=0)
    parser.add_argument('--model_name', type=str, default="RN50")
    parser.add_argument('--load_checkpoint', type=str, default=None)
    # We can fine-tune on the models loaded here. Lets see if we can just
    # Transplant the CLIP model loaded here into the VQA CLIP-ViL.
    parser.add_argument('--merge_base_clip_into_clip_vil', default=False, action='store_true')
    parser.add_argument('--save_path', type=str, default=None)
    parser = TextImageDataModule.add_argparse_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
