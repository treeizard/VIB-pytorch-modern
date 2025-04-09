import torch, os
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

class UnknownDatasetError(Exception):
    def __init__(self, name):
        super().__init__(f"Unknown dataset: {name}")

def return_data(args):
    name = args.dataset
    dset_dir = args.dset_dir
    batch_size = args.batch_size
    num_workers = os.cpu_count() if torch.cuda.is_available() else 0
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))
                                    ])

    if name == 'MNIST':
        root = os.path.join(dset_dir, 'MNIST')
        dset = MNIST
        train_kwargs = {'root': root, 'train': True, 'transform': transform, 'download': True}
        test_kwargs = {'root': root, 'train': False, 'transform': transform, 'download': False}
    else: 
        raise UnknownDatasetError()

    train_loader = DataLoader(dset(**train_kwargs), batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True)
    test_loader = DataLoader(dset(**test_kwargs), batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, drop_last=False)

    return {'train': train_loader, 'test': test_loader}

if __name__ == '__main__' :
    import argparse
    os.chdir('..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='MNIST', type=str)
    parser.add_argument('--dset_dir', default='datasets', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    args = parser.parse_args()

    data_loader = return_data(args)
    import ipdb; ipdb.set_trace()
