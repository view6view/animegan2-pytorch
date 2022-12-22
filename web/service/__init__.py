import argparse
import os

from model import Generator
import torch

pytorch_AnimeGAN2_parser = argparse.ArgumentParser()
pytorch_AnimeGAN2_parser.add_argument(
    '--checkpoint',
    type=str,
    default='./weights/pytorch/AnimeGAN2',
)
pytorch_AnimeGAN2_parser.add_argument(
    '--temp_dir',
    type=str,
    default='./temp/pytorch/AnimeGAN2',
)
pytorch_AnimeGAN2_parser.add_argument(
        '--device',
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
pytorch_AnimeGAN2_parser.add_argument(
    '--upsample_align',
    type=bool,
    default=False,
    help="Align corners in decoder upsampling layers"
)
pytorch_AnimeGAN2_parser.add_argument(
    '--x32',
    action="store_true",
    help="Resize images to multiple of 32"
)
pytorch_AnimeGAN2_args = pytorch_AnimeGAN2_parser.parse_args()
pytorch_AnimeGAN2_device = pytorch_AnimeGAN2_args.device
pytorch_AnimeGAN2_temp_dir = pytorch_AnimeGAN2_args.temp_dir
os.makedirs(pytorch_AnimeGAN2_temp_dir, exist_ok=True)
pytorch_AnimeGAN2_map = {}
for model_name in os.listdir(pytorch_AnimeGAN2_args.checkpoint):
    model_name_split = model_name.split(".")
    if len(model_name_split) != 2 or model_name_split[1] != "pt":
        continue
    net = Generator()
    net.load_state_dict(torch.load(os.path.join(pytorch_AnimeGAN2_args.checkpoint, model_name), map_location="cpu"))
    net.to(pytorch_AnimeGAN2_device).eval()
    pytorch_AnimeGAN2_map[model_name_split[0]] = net
