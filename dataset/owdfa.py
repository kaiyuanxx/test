import os
import cv2
import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from loguru import logger

from .utils.face_cropper import dlib_crop_face

try:
    import dlib
except:
    print('Please install dlib when using face detection!')


# -------------------------------------------------
# # OWDFADataset
# -------------------------------------------------
# Note that real-face images from different sources (e.g., Celeb-DF and FaceForensics++) 
# are intentionally merged and treated as one class.
def prepare_owdfa_samples(
    root,
    train=True,
    test_ratio=0.2,
    known_classes=None,
    train_classes=None,
    seed=2026
):
    """
    Prepare OWDFA samples by class-wise sampling and train/test splitting.

    The function performs the following steps:
      1. Data preparation: traverse subfolders under `root`, parse class labels
         from folder names, recursively collect image paths, and organize them
         by label (including special handling for the real-face class).
      2. Class-wise sampling: apply different sampling budgets for known and
         unknown classes, with a special rule for label 0.
      3. Train/test split: split sampled data into training and testing subsets
         according to `test_ratio`, and return the requested split based on the
         `train` flag.

    Args:
        root (str): Root directory containing class-wise subfolders.
        train (bool): If True, return training samples; otherwise return testing samples.
        test_ratio (float): Proportion of samples used for testing.
        known_classes (list[int] or None): List of known class labels.
        train_classes (list[int] or None): If provided, only these classes are included.
        seed (int): Random seed for reproducible sampling.

    Returns:
        list[list]: A list of [image_path, label] pairs.
    """

    # ---------------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------------
    np.random.seed(seed)
    samples_by_label = {}   # label -> list of image paths

    # ---------------------------------------------------------------------
    # Step 0: Traverse folders and collect image paths per label
    # ---------------------------------------------------------------------
    for folder in os.listdir(root):
        folder_path = os.path.join(root, folder)
        if not os.path.isdir(folder_path):
            continue

        # Parse class label from folder name prefix
        try:
            label = int(folder.split('_')[0])
        except Exception as e:
            logger.info(f"Skip folder '{folder}': cannot parse label ({e})")
            continue

        # Recursively collect all images under the folder
        imgs = []
        for dirpath, _, filenames in os.walk(folder_path):
            imgs.extend([
                os.path.join(dirpath, f)
                for f in filenames
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])

        print(f"[Collect] Label {label} | Folder '{folder}' | Images found: {len(imgs)}")

        # Special handling for real-face class (label == 0):
        if label == 0:
            imgs = np.random.choice(imgs, size=10000, replace=False).tolist()

        # Merge images into label-level container
        if label not in samples_by_label:
            samples_by_label[label] = []
        samples_by_label[label].extend(imgs)

    # ---------------------------------------------------------------------
    # Summary: number of collected images per label
    # ---------------------------------------------------------------------
    label_stats = {label: len(imgs) for label, imgs in samples_by_label.items()}
    print("[Summary] Total images per label:", label_stats)

    # ---------------------------------------------------------------------
    # Step 1 & 2: Class-wise sampling and train/test split
    # ---------------------------------------------------------------------
    final_samples = []

    for label, paths in samples_by_label.items():

        # Optionally restrict to selected training classes
        if train_classes is not None and label not in train_classes:
            continue

        paths = np.array(paths)

        # -------------------------------------------------------------
        # Step 1: Determine sampling budget per class
        # -------------------------------------------------------------
        # Q: 每个类样本是不是太少了 
        if known_classes is not None and label in known_classes:
            if label == 0:
                sampled_n = 20000   # special case: real-face class
            else:
                sampled_n = 2000
        else:
            sampled_n = 1500

        # Apply sampling if necessary
        if len(paths) > sampled_n:
            paths = np.random.choice(paths, size=sampled_n, replace=False)

        # -------------------------------------------------------------
        # Step 2: Train / Test split
        # -------------------------------------------------------------
        np.random.shuffle(paths)
        n = len(paths)
        split_idx = int(n * (1 - test_ratio))

        train_paths = paths[:split_idx]
        test_paths = paths[split_idx:]

        selected_paths = train_paths if train else test_paths
        for p in selected_paths:
            final_samples.append([p, label])

    return final_samples


class OWDFADataset(Dataset):
    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
        train=True,
        test_ratio=0.2,
        known_classes=None,
        train_classes=None,
        crop_face=True,
        predictor_path=None,
        seed=2026,
    ):
        """
        OWDFADataset: a dataset wrapper that prepares OWDFA samples and loads images.

        Args:
            root (str):
                Root directory of the dataset. Subfolders are expected to be class-wise.
            transform:
                Optional transform (e.g., Albumentations). Expected to be callable on `img`.
            target_transform:
                Optional transform applied to the label.
            train (bool):
                If True, use the training split; otherwise use the testing split.
            test_ratio (float):
                Ratio of samples used for testing.
            known_classes (list[int] or None):
                List of known class labels (used by sampling routine).
            train_classes (list[int] or None):
                If provided, only these classes are included.
            crop_face (bool):
                Whether to crop faces using dlib before applying transforms.
            predictor_path (str or None):
                Path to dlib 68-landmark predictor. Required when crop_face=True.
            seed (int):
                Random seed used by the sampling routine.
        """
        # 为每个类采样并根据train/test标志进行划分
        # Prepare samples (path, label) using the dataset split logic
        self.samples = prepare_owdfa_samples(
            root=root,
            train=train,
            test_ratio=test_ratio,
            known_classes=known_classes,
            train_classes=train_classes,
            seed=seed,
        )

        # Ensure targets are integers
        self.samples = [[s[0], int(s[1])] for s in self.samples]

        # log for total number of samples
        logger.info(f"[Dataset] total number of samples: {len(self.samples)}")

        # Cache targets for downstream utilities
        self.targets = [s[1] for s in self.samples]

        # Store transforms
        self.transform = transform
        self.target_transform = target_transform

        # Face-cropping configuration
        self.crop_face = crop_face
        self.predictor_path = predictor_path

        # Unique indices (often used by SSL / clustering pipelines)
        self.uq_idxs = np.arange(len(self.samples))

        # Initialize dlib components if face cropping is enabled
        if self.crop_face:
            assert predictor_path is not None, "predictor_path must be provided when crop_face=True"
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(predictor_path)

    def __len__(self):
        """Return the number of samples in the current split."""
        return len(self.samples)

    def load_img(self, path):
        """
        Load an image from disk, optionally crop the face, and convert BGR (OpenCV) to RGB (PIL).

        Args:
            path (str): Absolute path to an image.

        Returns:
            PIL.Image: Loaded (and optionally face-cropped) image in RGB format.
        """
        img = cv2.imread(path)
        if self.crop_face:
            img = dlib_crop_face(img, self.detector, self.predictor, align=False, margin=1.2)
        img = img[:, :, ::-1]  # BGR -> RGB
        return Image.fromarray(img)

    def __getitem__(self, idx):
        """
        Fetch one item.

        Returns a dict with:
            - image: transformed image object
            - target: label (possibly transformed)
            - idx: unique index
            - img_path: original image path
        """
        path, label = self.samples[idx]
        
        try:
            img = self.load_img(path)

            if self.transform is not None:
                transformed = self.transform(img)
                
                img = transformed if isinstance(transformed, list) else transformed

            if self.target_transform is not None:
                label = self.target_transform(label)

            item = {
                'image': img,
                'target': label,
                'idx': self.uq_idxs[idx],
                'img_path': path
            }
        except Exception as e:
            raise RuntimeError(f"⚠️ Failed to load image: {e} | Path: {path}")

        return item
