from torchvision import transforms as T
import torchvision.transforms.functional as F

def create_data_transforms(args, split='train'):
    image_size = getattr(args, 'image_size', 224)
    mean       = getattr(args, 'mean',  [0.485, 0.456, 0.406])
    std        = getattr(args, 'std',   [0.229, 0.224, 0.225])

    size_hw = (image_size, image_size)        # (H, W)
    crop_hw = int((256 / 224) * image_size)

    if split == 'train':
        train_transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.Resize(size_hw, interpolation=T.InterpolationMode.BILINEAR,
                     antialias=True),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])

        return train_transform

    # --------------------------------------------------------
    # Test / validation
    # --------------------------------------------------------
    test_transform = T.Compose([
        T.Resize(size_hw, interpolation=T.InterpolationMode.BILINEAR,
                 antialias=True),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    return test_transform