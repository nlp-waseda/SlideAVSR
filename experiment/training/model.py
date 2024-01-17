import os
import io
from typing import Dict, Iterable, Optional, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from transformers import DonutProcessor, VisionEncoderDecoderModel
# from llava.model.builder import load_pretrained_model
# from llava.mm_utils import get_model_name_from_path

# from peft import inject_adapter_in_model, LoraConfig

from whisper.whisper import _MODELS, _ALIGNMENT_HEADS, _download, available_models
from whisper.whisper.model import Whisper, Conv1d, ResidualAttentionBlock, LayerNorm, ModelDimensions, sinusoids


class AudioVisualEncoder(nn.Module):
    def __init__(
        self, args, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

        self.vision_encoder = args.vision_encoder
        if args.vision_encoder == "clip":
            # HACK 18.48-second to fit encoder
            self.register_buffer("positional_embedding_sw", sinusoids(924, n_state))
            model_path = "liuhaotian/llava-v1.5-7b"
            _, model, image_processor, _ = load_pretrained_model(
                model_path=model_path,
                model_base=None,
                model_name=get_model_name_from_path(model_path)
            )
            self.image_processor = image_processor
            self.clip_encoder = model.get_model().get_vision_tower()
            for p in self.clip_encoder.parameters():
                p.requires_grad = False
            if args.no_llava_mlp:
                self.llava_proj = nn.Identity()
                self.additional_proj = nn.Linear(
                    self.clip_encoder.vision_tower.vision_model.post_layernorm.weight.shape[0],
                    n_state
                ).half()
            else:
                self.llava_proj = model.get_model().mm_projector
                self.additional_proj = nn.Sequential(
                    nn.GELU(),
                    nn.Linear(self.llava_proj[-1].weight.shape[0], n_state).half()
                )
        elif args.vision_encoder == "donut":
            self.register_buffer("positional_embedding_sw", sinusoids(1499, n_state))
            self.image_processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
            model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
            self.donut_encoder = model.encoder.half()
            for p in self.donut_encoder.parameters():
                p.requires_grad = False
            self.additional_proj = nn.Sequential(
                nn.Linear(1024, n_state),
                nn.GELU(),
                nn.Linear(n_state, n_state),
            ).half()
        else:
            raise NotImplementedError

        del model

    def forward(self, x: Tensor, y: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        y : torch.Tensor
            the image feature
        """
        # audio feature
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        assert x.shape[1:] == self.positional_embedding_sw.shape, "incorrect audio shape"
        x = (x + self.positional_embedding_sw).to(x.dtype)

        # image feature
        if self.vision_encoder == "clip":
            y = self.clip_encoder(y)
            y = self.llava_proj(y)
            y = self.additional_proj(y)
        elif self.vision_encoder == "donut":
            y = self.donut_encoder(y)
            y = self.additional_proj(y.pooler_output.unsqueeze(1))

        # concat image and audio features
        x = torch.cat([y, x], dim=1).to(x.dtype)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x


class SightWhisper(Whisper):
    def __init__(self, args, dims: ModelDimensions):
        super().__init__(dims)
        self.encoder = AudioVisualEncoder(
            args,
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )

    def forward(
        self, mel: torch.Tensor, image_tensor: torch.Tensor, tokens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return self.decoder(tokens, self.encoder(mel, image_tensor))


def load_model(
    args,
    name: str,
    device: Optional[Union[str, torch.device]] = None,
    download_root: str = None,
    in_memory: bool = False,
) -> SightWhisper:
    """
    Load a Whisper AVSR model

    Parameters
    ----------
    name : str
        one of the official model names listed by `whisper.available_models()`, or
        path to a model checkpoint containing the model dimensions and the model state_dict.
    device : Union[str, torch.device]
        the PyTorch device to put the model into
    download_root: str
        path to download the model files; by default, it uses "~/.cache/whisper"
    in_memory: bool
        whether to preload the model weights into host memory

    Returns
    -------
    model : Whisper
        The Whisper ASR model instance
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if download_root is None:
        default = os.path.join(os.path.expanduser("~"), ".cache")
        download_root = os.path.join(os.getenv("XDG_CACHE_HOME", default), "whisper")

    if name in _MODELS:
        checkpoint_file = _download(_MODELS[name], download_root, in_memory)
        alignment_heads = _ALIGNMENT_HEADS[name]
    elif os.path.isfile(name):
        checkpoint_file = open(name, "rb").read() if in_memory else name
        alignment_heads = None
    else:
        raise RuntimeError(
            f"Model {name} not found; available models = {available_models()}"
        )

    with (
        io.BytesIO(checkpoint_file) if in_memory else open(checkpoint_file, "rb")
    ) as fp:
        checkpoint = torch.load(fp, map_location=device)
    del checkpoint_file

    dims = ModelDimensions(**checkpoint["dims"])
    model = SightWhisper(args, dims)

    if not args.inference:
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint["model_state_dict"].items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        del model_dict, pretrained_dict

    if alignment_heads is not None:
        model.set_alignment_heads(alignment_heads)

    if args.use_lora:
        # insert lora layers
        # TODO only save lora to checkpoints
        target_modules = ["query", "key", "value", "out"]
        if args.lora_clip:
            target_modules += ["q_proj", "k_proj", "v_proj", "out_proj"]
        lora_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_rank,
            bias="none",
            target_modules=target_modules,
        )
        model = inject_adapter_in_model(lora_config, model)
        if not args.inference:
            # unfreeze vision encoder's mlp
            for p in model.encoder.llava_proj.parameters():
                p.requires_grad = True
            for p in model.encoder.additional_proj.parameters():
                p.requires_grad = True

    print(model)

    return model.to(device), model.encoder.image_processor
