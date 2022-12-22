import torch
from web.service import pytorch_AnimeGAN2_args, pytorch_AnimeGAN2_map
from torchvision.transforms.functional import to_tensor, to_pil_image
from web.util import image_util


def AnimeGAN2_run(file_path, model_name):
    image = image_util.load_image(
        file_path,
        pytorch_AnimeGAN2_args.x32)
    with torch.no_grad():
        image = to_tensor(image).unsqueeze(0) * 2 - 1
        net = pytorch_AnimeGAN2_map[model_name]
        out = net(
            image.to(pytorch_AnimeGAN2_args.device),
            pytorch_AnimeGAN2_args.upsample_align).cpu()
        out = out.squeeze(0).clip(-1, 1) * 0.5 + 0.5
        out = to_pil_image(out)

    return out
