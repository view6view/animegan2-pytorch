import argparse
from model import Generator
import torch

pytorch_aniMegan2_parser = argparse.ArgumentParser()
pytorch_aniMegan2_parser.add_argument(
    '--checkpoint',
    type=str,
    default='./weights/paprika.pt',
)
pytorch_aniMegan2_parser.add_argument(
        '--temp_dir',
        type=str,
        default='./temp',
    )
pytorch_aniMegan2_parser.add_argument(
        '--device',
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
pytorch_aniMegan2_parser.add_argument(
    '--upsample_align',
    type=bool,
    default=False,
    help="Align corners in decoder upsampling layers"
)
pytorch_aniMegan2_parser.add_argument(
    '--x32',
    action="store_true",
    help="Resize images to multiple of 32"
)
pytorch_aniMegan2_args = pytorch_aniMegan2_parser.parse_args()
pytorch_aniMegan2_device = pytorch_aniMegan2_args.device
pytorch_aniMegan2_net = Generator()
pytorch_aniMegan2_net.load_state_dict(torch.load(pytorch_aniMegan2_args.checkpoint, map_location="cpu"))
pytorch_aniMegan2_net.to(pytorch_aniMegan2_device).eval()
pytorch_aniMegan2_torch = torch
