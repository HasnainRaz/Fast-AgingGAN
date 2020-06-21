import os
import shutil
from argparse import ArgumentParser

from scipy.io import loadmat

parser = ArgumentParser()
parser.add_argument('--image_dir',
                    default='/Users/hasnainraza/Downloads/CACD2000/',
                    help='The CACD200 images dir')
parser.add_argument('--metadata',
                    default='/Users/hasnainraza/Downloads/celebrity2000_meta.mat',
                    help='The metadata for the CACD2000')
parser.add_argument('--output_dir',
                    default='/Users/hasnainraza/Downloads/CACDDomains',
                    help='The directory to write processed images')


def main():
    args = parser.parse_args()
    metadata = loadmat(args.metadata)['celebrityImageData'][0][0]
    ages = [x[0] for x in metadata[0]]
    names = [x[0][0] for x in metadata[-1]]

    ages_to_keep_a = [x for x in range(18, 30)]
    ages_to_keep_b = [x for x in range(55, 100)]

    domainA, domainB = [], []
    for age, name in zip(ages, names):
        if age in ages_to_keep_a:
            domainA.append(name)
        if age in ages_to_keep_b:
            domainB.append(name)

    N = min(len(domainA), len(domainB))
    domainA = domainA[:N]
    domainB = domainB[:N]
    print(f'Images in A {len(domainA)} and B {len(domainB)}')

    domainA_dir = os.path.join(args.output_dir, 'trainA')
    domainB_dir = os.path.join(args.output_dir, 'trainB')

    os.makedirs(domainA_dir, exist_ok=True)
    os.makedirs(domainB_dir, exist_ok=True)

    for imageA, imageB in zip(domainA, domainB):
        shutil.copy(os.path.join(args.image_dir, imageA), os.path.join(domainA_dir, imageA))
        shutil.copy(os.path.join(args.image_dir, imageB), os.path.join(domainB_dir, imageB))


if __name__ == '__main__':
    main()
