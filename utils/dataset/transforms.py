import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def train_transforms(mean, std):
    return A.Compose(
        [
            # A.RandomCrop(p=0.5),
            A.OneOf([
                A.HueSaturationValue(p=0.9),
                A.RandomBrightnessContrast(p=0.9),
                A.RandomGamma(p=0.9),
            ], p=0.9),
            A.OneOf([
                A.IAAAdditiveGaussianNoise(p=0.5),
                A.IAASharpen(alpha=(0.1, 0.3), p=0.5),
                A.CLAHE(p=0.8),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.GaussianBlur(blur_limit=3, p=0.5),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),
            ], p=0.0),
            A.Cutout(num_holes=4, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
            A.ToGray(p=0.01),
            A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            A.Normalize(mean=mean, std=std, p=1.),
            ToTensorV2(p=1.),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )
    )


def valid_transforms(mean, std):
    return A.Compose([
            A.Normalize(mean=mean, std=std, p=1.),
            ToTensorV2(p=1.),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )
    )


def test_transform(mean, std):
    return A.Compose(
        [
            A.Normalize(mean=mean, std=std, p=1.),
            ToTensorV2(p=1.),
        ],
        p=1.0
    )
