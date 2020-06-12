import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import random
import torch
import yaml
from torchvision import transforms
from gan_module import Generator
from PIL import Image

parser = ArgumentParser()
parser.add_argument('--config', default='configs/aging_gan.yaml', help='Config to use for training')
parser.add_argument('--image_dir', default='/Users/hasnainraza/Downloads/CACD_VS/', help='The CACD200 images dir')


@torch.no_grad()
def main():
    args = parser.parse_args()
    image_paths = [os.path.join(args.image_dir, x) for x in os.listdir(args.image_dir) if
                   x.endswith('.png') or x.endswith('.jpg')]
    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    model = Generator(ngf=config['ngf'], n_blocks=config['n_blocks'])
    ckpt = torch.load('lightning_logs/version_0/checkpoints/epoch=26.ckpt', map_location='cpu')
    new_state_dict = {}
    for k, v in ckpt['state_dict'].items():
        if str(k).startswith('genA2B'):
            new_state_dict[str(k).replace('genA2B.', '')] = v
    model.load_state_dict(new_state_dict)
    model.eval()

    trans = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    random.shuffle(image_paths)
    fig, ax = plt.subplots(2, 6, figsize=(20, 10))
    for i in range(6):
        img = Image.open(image_paths[i])
        img = trans(img).unsqueeze(0)
        aged_face, _, _ = model(img)
        aged_face = (aged_face.squeeze().permute(1, 2, 0).numpy() + 1.0) / 2.0
        ax[0, i].imshow((img.squeeze().permute(1, 2, 0).numpy() + 1.0) / 2.0)
        ax[1, i].imshow(aged_face)
    plt.show()


if __name__ == '__main__':
    main()
