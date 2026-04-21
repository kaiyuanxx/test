from copy import deepcopy
from loguru import logger
import numpy as np

from .owdfa import OWDFADataset
from .utils.data_utils import MergedDataset, dataset_stats

def get_dataset(args, train_transform, test_transform):

    datasets = get_owdfa_datasets(train_transform=train_transform, test_transform=test_transform, train_classes=args.train_classes,
                                 dataset_root= args.dataset.data_root, predictor_path=args.dataset.predictor_path, 
                                 prop_train_labels=args.dataset.prop_train_labels, known_classes=args.known_classes)

    train_dataset = MergedDataset(labelled_dataset=deepcopy(datasets['train_labelled']),
                                  unlabelled_dataset=deepcopy(datasets['train_unlabelled']))
    test_dataset = datasets['test']

    return train_dataset, test_dataset



def get_owdfa_datasets(train_transform, test_transform, train_classes, dataset_root, 
                       prop_train_labels=0.75, seed=2026, split_train_val=False,
                       crop_face=True, predictor_path=None, known_classes=None):
    """
    Construct datasets for the open-world setting.

    This function builds the following datasets:
      - train_labelled:
            Labelled training data sampled from known classes only.
            Optionally further split into train/validation subsets.
      - train_unlabelled:
            Unlabelled training data consisting of:
              (1) remaining samples from known classes, and
              (2) all samples from unknown (novel) classes.
      - val (optional):
            Validation set sampled from labelled training data.
      - test:
            Test set constructed from the same class pool as training,
            following the same class filtering rules.

    The final label space is unified across labelled, unlabelled,
    and test datasets via a shared target_transform.
    """
    np.random.seed(seed)
    
    # ------------------------------------------------------------------
    # Step 0: Construct the full training dataset (before label/unlabel split)
    # ------------------------------------------------------------------
    train_dataset = OWDFADataset(root=dataset_root,
                                 transform=train_transform, 
                                 train=True,
                                 known_classes=known_classes,
                                 train_classes=train_classes,
                                 crop_face=crop_face, predictor_path=predictor_path, seed=seed)
    
    
    # ------------------------------------------------------------------
    # Step 1: Construct the labelled training dataset
    #   - Keep only samples from known classes
    #   - Randomly subsample a proportion of instances as labelled data
    # ------------------------------------------------------------------

    # Filter the training dataset to known classes only   从完整的训练数据集中筛选出已知类的样本
    # 返回dataset对象，包含筛选出来的已知类samples、targets、uq_idxs等属性，并且target_transform被设置为将原始标签映射到连续标签空间
    train_dataset_labelled = subsample_classes(deepcopy(train_dataset), include_classes=known_classes)
    
    # Randomly select a subset of instances as labelled samples  
    # 从已知类的样本中随机选择一部分作为有标签的数据，比例为0.75，subsample_indices是被选中的样本的索引
    subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)
    
    # ------------------------------------------------------------------
    # Step 2 (optional): Split labelled data into train / validation sets
    # ------------------------------------------------------------------
    if split_train_val:
        train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled, val_instances_per_class=5)
        train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
        val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
        # Use test-time transforms for validation
        val_dataset_labelled_split.transform = test_transform
    else:
        train_dataset_labelled_split, val_dataset_labelled_split = None, None


    # ------------------------------------------------------------------
    # Step 3: Construct the unlabelled training dataset
    #   - Remove labelled instances from the full training dataset
    #   - Remaining samples (known + unknown classes) are treated as unlabelled
    # ------------------------------------------------------------------
    labelled_uq = set(train_dataset_labelled.uq_idxs)
    unlabelled_indices = np.array(list(set(train_dataset.uq_idxs) - labelled_uq))
    train_dataset_unlabelled = subsample_dataset(deepcopy(train_dataset), unlabelled_indices)

    # ------------------------------------------------------------------
    # Step 4: Construct the test dataset
    # ------------------------------------------------------------------
    test_dataset = OWDFADataset(root=dataset_root,
                                 transform=test_transform, 
                                 train=False,
                                 known_classes=known_classes,
                                 train_classes=train_classes,
                                 crop_face=crop_face, predictor_path=predictor_path, seed=seed)

    # ------------------------------------------------------------------
    # Step 5: Define a unified target_transform for open-world classification
    #   - Known and unknown classes are mapped into a single contiguous label space
    #   - The same mapping is applied to labelled, unlabelled, and test datasets
    #   - target_xform_dict 的前k个键是已知类标签，后面是未知类标签，值是新的连续标签索引
    # ------------------------------------------------------------------
    unlabelled_classes = list(set(train_dataset.targets) - set(known_classes))
    target_xform_dict = {k: i for i, k in enumerate(list(known_classes) + unlabelled_classes)}

    train_dataset_labelled.target_transform = lambda x: target_xform_dict[x]
    train_dataset_unlabelled.target_transform = lambda x: target_xform_dict[x]
    test_dataset.target_transform = lambda x: target_xform_dict[x]

    # Select final labelled / validation datasets based on configuration
    train_dataset_labelled = train_dataset_labelled_split if split_train_val else train_dataset_labelled
    val_dataset_labelled = val_dataset_labelled_split if split_train_val else None

    # ------------------------------------------------------------------
    # Step 6: Log dataset statistics for verification
    # ------------------------------------------------------------------
    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val': val_dataset_labelled,
        'test': test_dataset,
    }

    logger.info("[Dataset Statistics]")
    dataset_stats(all_datasets, known_classes=known_classes, target_xform_dict=target_xform_dict)

    return all_datasets


# -------------------------------------------------
# Open-World Dataset Partitioning and Subsampling
# -------------------------------------------------

# Subsample a dataset by selecting a subset of indices.
def subsample_dataset(dataset, idxs):
    # Create a boolean mask indicating which indices are kept
    mask = np.zeros(len(dataset), dtype=bool)
    mask[idxs] = True

    # Apply the mask to samples, targets, and unique indices
    dataset.samples = np.array(dataset.samples)[mask].tolist()
    dataset.targets = np.array(dataset.targets)[mask].tolist()
    dataset.uq_idxs = dataset.uq_idxs[mask]

    # Ensure labels stored in samples are of type int
    dataset.samples = [[x[0], int(x[1])] for x in dataset.samples]

    return dataset

# Subsample a dataset to include only specified classes and remap labels.
def subsample_classes(dataset, include_classes):
    # find indices of samples whose labels are in the include_classes list
    cls_idxs = [i for i, l in enumerate(dataset.targets) if l in include_classes]

    # Define a label remapping: original_label -> new_contiguous_label
    target_xform_dict = {k: i for i, k in enumerate(include_classes)}

    # Subsample the dataset to keep only selected classes
    dataset = subsample_dataset(dataset, cls_idxs)

    # Apply the target transform to map labels into a contiguous space
    dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset

# Randomly select a subset of instance indices from a dataset.
def subsample_instances(dataset, prop_indices_to_subsample=0.8):
    num = int(len(dataset.targets) * prop_indices_to_subsample)
    return np.random.choice(np.arange(len(dataset.targets)), size=num, replace=False)

# Split a labelled training dataset into train and validation indices
def get_train_val_indices(train_dataset, val_instances_per_class=5):
    # Unique class labels in the dataset
    train_classes = list(set(train_dataset.targets))
    train_idxs = []
    val_idxs = []
    for cls in train_classes:
        # Indices of all samples belonging to the current class
        cls_idxs = np.where(np.array(train_dataset.targets) == cls)[0]

        # --- Minimum sample check per class ---
        if len(cls_idxs) < val_instances_per_class:
            raise ValueError(
                f"Class {cls} has only {len(cls_idxs)} samples, "
                f"which is fewer than val_instances_per_class={val_instances_per_class}."
            )
        
        # Randomly select validation instances for this class
        v_ = np.random.choice(cls_idxs, replace=False, size=val_instances_per_class)

        # Remaining instances are used for training
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs

if __name__ == '__main__':
    # Root directory
    dataset_root = "/home/xky/Desktop/OW-DFA/OWDFA-CAL-main/data/OWDFA40-Benchmark/data"
    predictor_path = "/home/xky/Desktop/OW-DFA/OWDFA-CAL-main/data/OWDFA40-Benchmark/shape_predictor_68_face_landmarks.dat"

    # training and testing transforms
    train_transform = None
    test_transform = None

    # Protocol 1
    logger.info("-------------------------------------------------------------Protocol 1-------------------------------------------------------------")
    known_classes = [0, 1, 3, 6, 8, 11, 14, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39]
    train_classes = list(range(0,41))
    datasets = get_owdfa_datasets(train_transform, test_transform,
                                    dataset_root=dataset_root,
                                    train_classes=train_classes,
                                    prop_train_labels=0.75, seed=2026, split_train_val=False,
                                    crop_face=True, predictor_path=predictor_path, known_classes=known_classes)

    # Protocol 2
    logger.info("-------------------------------------------------------------Protocol 2-------------------------------------------------------------")
    known_classes = [0, 1, 3, 6, 8, 11, 14, 21, 23, 25, 27, 29, 31]
    train_classes = list(range(0,41))
    datasets = get_owdfa_datasets(train_transform, test_transform,
                                    dataset_root=dataset_root,
                                    train_classes=train_classes,
                                    prop_train_labels=0.75, seed=2025, split_train_val=False,
                                    crop_face=True, predictor_path=predictor_path, known_classes=known_classes)

    # Protocol 3
    logger.info("-------------------------------------------------------------Protocol 3-------------------------------------------------------------")
    known_classes = [0, 1, 3, 5, 6, 7, 8, 10, 11, 13, 14, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 29, 31, 33, 35, 36, 37, 38, 39]
    train_classes = list(range(0,41))
    datasets = get_owdfa_datasets(train_transform, test_transform,
                                    dataset_root=dataset_root,
                                    train_classes=train_classes,
                                    prop_train_labels=0.75, seed=2025, split_train_val=False,
                                    crop_face=True, predictor_path=predictor_path, known_classes=known_classes)