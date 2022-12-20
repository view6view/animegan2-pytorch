from web.service import pytorch_aniMegan2_net, pytorch_aniMegan2_args, pytorch_aniMegan2_torch
from torchvision.transforms.functional import to_tensor, to_pil_image
from web.util import image_util


def aniMegan2_run(file_path):
    image = image_util.load_image(
        file_path,
        pytorch_aniMegan2_args.x32)
    with pytorch_aniMegan2_torch.no_grad():
        image = to_tensor(image).unsqueeze(0) * 2 - 1
        out = pytorch_aniMegan2_net(
            image.to(pytorch_aniMegan2_args.device),
            pytorch_aniMegan2_args.upsample_align).cpu()
        out = out.squeeze(0).clip(-1, 1) * 0.5 + 0.5
        out = to_pil_image(out)

    return out
