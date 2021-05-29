import csv
import os
import random
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pdb
START = "<SOS>" # 0
END = "<EOS>" # 1
PAD = "<PAD>" # 2
SPECIAL_TOKENS = [START, END, PAD]


# Rather ignorant way to encode the truth, but at least it works.
def encode_truth(truth, token_to_id):

    truth_tokens = truth.split()
    for token in truth_tokens:
        if token not in token_to_id:
            raise Exception("Truth contains unknown token")
    truth_tokens = [token_to_id[x] for x in truth_tokens]
    if '' in truth_tokens: truth_tokens.remove('')
    return truth_tokens


def load_vocab(tokens_paths):
    tokens = []
    tokens.extend(SPECIAL_TOKENS)
    for tokens_file in tokens_paths:
        with open(tokens_file, "r") as fd:
            reader = fd.read()
            for token in reader.split("\n"):
                if token not in tokens:
                    tokens.append(token)
    token_to_id = {tok: i for i, tok in enumerate(tokens)}
    id_to_token = {i: tok for i, tok in enumerate(tokens)}
    return token_to_id, id_to_token


def split_gt(groundtruth, proportion=1.0, test_percent=None):
    root = os.path.join(os.path.dirname(groundtruth), "images")
    with open(groundtruth, "r") as fd:
        data=[]
        for line in fd:
            data.append(line.strip().split("\t"))
        random.shuffle(data)
        dataset_len = round(len(data) * proportion)
        data = data[:dataset_len]
        data = [[os.path.join(root, x[0]), x[1]] for x in data]
    
    if test_percent:
        test_len = round(len(data) * test_percent)
        return data[test_len:], data[:test_len]
    else:
        return data


def collate_batch(data):
    max_len = max([len(d["truth"]["encoded"]) for d in data])
    # Padding with -1, will later be replaced with the PAD token
    padded_encoded = [
        d["truth"]["encoded"] + (max_len - len(d["truth"]["encoded"])) * [-1]
        for d in data
    ]
    return {
        "path": [d["path"] for d in data],
        "image": torch.stack([d["image"] for d in data], dim=0),
        "truth": {
            "text": [d["truth"]["text"] for d in data],
            "encoded": torch.tensor(padded_encoded)
        },
    }

def collate_eval_batch(data):
    max_len = max([len(d["truth"]["encoded"]) for d in data])
    # Padding with -1, will later be replaced with the PAD token
    padded_encoded = [
        d["truth"]["encoded"] + (max_len - len(d["truth"]["encoded"])) * [-1]
        for d in data
    ]
    return {
        "path": [d["path"] for d in data],
        "file_path":[d["file_path"] for d in data],
        "image": torch.stack([d["image"] for d in data], dim=0),
        "truth": {
            "text": [d["truth"]["text"] for d in data],
            "encoded": torch.tensor(padded_encoded)
        },
    }

class LoadDataset(Dataset):
    """Load Dataset"""

    def __init__(
        self,
        groundtruth,
        tokens_file,
        crop=False,
        transform=None,
        rgb=3,
    ):
        """
        Args:
            groundtruth (string): Path to ground truth TXT/TSV file
            tokens_file (string): Path to tokens TXT file
            ext (string): Extension of the input files
            crop (bool, optional): Crop images to their bounding boxes [Default: False]
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(LoadDataset, self).__init__()
        self.crop = crop
        self.transform = transform
        self.rgb = rgb
        self.token_to_id, self.id_to_token = load_vocab(tokens_file) ## 토큰: index형태인 딕셔너리와 index:토큰 형태인  딕셔너리.
        self.data = [
            {
                "path": p,
                "truth": {
                    "text": truth,
                    "encoded": [
                        self.token_to_id[START],
                        *encode_truth(truth, self.token_to_id),
                        self.token_to_id[END],
                    ],
                },
            }
            for p, truth in groundtruth
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = self.data[i]
        image = Image.open(item["path"])
        if self.rgb == 3:
            image = image.convert("RGB")
        elif self.rgb == 1:
            image = image.convert("L")
        else:
            raise NotImplementedError

        if self.crop:
            # Image needs to be inverted because the bounding box cuts off black pixels,
            # not white ones.
            bounding_box = ImageOps.invert(image).getbbox()
            image = image.crop(bounding_box)

        if self.transform:
            image = self.transform(image)

        return {"path": item["path"], "truth": item["truth"], "image": image}

class LoadEvalDataset(Dataset):
    """Load Dataset"""

    def __init__(
        self,
        groundtruth,
        token_to_id,
        id_to_token,
        crop=False,
        transform=None,
        rgb=3,
    ):
        """
        Args:
            groundtruth (string): Path to ground truth TXT/TSV file
            tokens_file (string): Path to tokens TXT file
            ext (string): Extension of the input files
            crop (bool, optional): Crop images to their bounding boxes [Default: False]
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(LoadEvalDataset, self).__init__()
        self.crop = crop
        self.rgb = rgb
        self.token_to_id = token_to_id
        self.id_to_token = id_to_token
        self.transform = transform
        self.data = [
            {
                "path": p,
                "file_path":p1,
                "truth": {
                    "text": truth,
                    "encoded": [
                        self.token_to_id[START],
                        *encode_truth(truth, self.token_to_id),
                        self.token_to_id[END],
                    ],
                },
            }
            for p, p1,truth in groundtruth
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = self.data[i]
        image = Image.open(item["path"])
        if self.rgb == 3:
            image = image.convert("RGB")
        elif self.rgb == 1:
            image = image.convert("L")
        else:
            raise NotImplementedError

        if self.crop:
            # Image needs to be inverted because the bounding box cuts off black pixels,
            # not white ones.
            bounding_box = ImageOps.invert(image).getbbox()
            image = image.crop(bounding_box)

        if self.transform:
            image = self.transform(image)

        return {"path": item["path"], "file_path":item["file_path"],"truth": item["truth"], "image": image}

def dataset_loader(options, transformed):


    # Read data
    train_data, valid_data = [], [] 
    if options.data.random_split:
        for i, path in enumerate(options.data.train):
            prop = 1.0
            if len(options.data.dataset_proportions) > i: ## data path가 여러개 일 경우(외부 데이터를 쓸 경우).
                prop = options.data.dataset_proportions[i]
            train, valid = split_gt(path, prop, options.data.test_proportions)
            '''(Pdb) valid
[['/opt/ml/input/data/train_dataset/images/train_00008.jpg', 'F \\left( 0 , \\sqrt { a ^ { 2 } + b ^ { 2 } } \\right) , \\left( 0 , - \\sqrt { a ^ { 2 } + b ^ { 2 } }
 \\right)'], ['/opt/ml/input/data/train_dataset/images/train_00014.jpg', 'M P _ { e } = \\lim _ { \\Delta l \\to 0 } \\frac { g \\left( l + \\Delta l \\right) - g 
 \\left( l \\right) } { \\Delta l } = \\frac { d g } { d l }'], ['/opt/ml/input/data/train_dataset/images/train_00004.jpg', 'I = d q / d t'], 
 ['/opt/ml/input/data/train_dataset/images/train_00024.jpg', '\\left( 1 2 x - 1 6 \\right) \\times \\left( - \\frac { 3 } { 4 } \\right) = 1 2 x \\times \\left( - \\frac 
 { 3 } { 4 } \\right)'], ['/opt/ml/input/data/train_dataset/images/train_00006.jpg', 'i ^ { 2 } = - 1 \\left( i = \\sqrt { - 1 } \\right)'], 
 ['/opt/ml/input/data/train_dataset/images/train_00016.jpg', '= \\frac { \\left( n + 2 \\right) \\left( n + 1 \\right) } { 2 } = 7 8 = 2 \\times 3 9 = 2 \\times 3 
 \\times 1 3']]  -> [file_path, ground truth]의 형태.
            '''
            train_data += train
            valid_data += valid
    else:
        for i, path in enumerate(options.data.train):
            prop = 1.0
            if len(options.data.dataset_proportions) > i:
                prop = options.data.dataset_proportions[i]
            train_data += split_gt(path, prop)
        for i, path in enumerate(options.data.test):
            valid = split_gt(path)
            valid_data += valid

    # Load data
    train_dataset = LoadDataset(
        train_data, options.data.token_paths, crop=options.data.crop, transform=transformed, rgb=options.data.rgb
    )
    '''(Pdb) train_dataset.data
[{'path': '/opt/ml/input/data/train_dataset/images/train_00005.jpg', 'truth': {'text': '\\sum \\overrightarrow { F } _ { e x t } = d', 
'encoded': [0, 117, 126, 224, 178, 213, 10, 224, 243, 204, 123, 213, 180, 215, 1]}}, {'path': '/opt/ml/input/data/train_dataset/images/train_00015.jpg', 
'truth': {'text': '\\sum F _ { \\theta } = m a _ { \\theta }', 'encoded': [0, 117, 178, 10, 224, 33, 213, 180, 149, 205, 10, 224, 33, 213, 1]}}, 
{'path': '/opt/ml/input/data/train_dataset/images/train_00027.jpg', 'truth': {'text': '= x \\left( x - 1 \\right) \\cdot \\left( x - 2 \\right) !', 
'encoded': [0, 180, 204, 229, 204, 214, 125, 137, 170, 229, 204, 214, 177, 137, 48, 1]}}, {'path': '/opt/ml/input/data/train_dataset/images/train_00023.jpg', 
'truth': {'text': 't = - 1', 'encoded': [0, 123, 180, 214, 125, 1]}}, {'path': '/opt/ml/input/data/train_dataset/images/train_00012.jpg', 
'truth': {'text': '7 \\div 4', 'encoded': [0, 160, 121, 85, 1]}}, {'path': '/opt/ml/input/data/train_dataset/images/train_00013.jpg', 
'truth': {'text': 'f \\left( x \\right) = 4 x ^ { 3 }', 'encoded': [0, 25, 229, 204, 137, 180, 85, 204, 236, 224, 111, 213, 1]}}, 
{'path': '/opt/ml/input/data/train_dataset/images/train_00019.jpg', 'truth': {'text': 'a _ { n } = a _ { 1 } r ^ { n - 1 }', 
'encoded': [0, 205, 10, 224, 61, 213, 180, 205, 10, 224, 125, 213, 164, 236, 224, 61, 214, 125, 213, 1]}}, ...
    '''
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=options.batch_size,
        shuffle=True,
        num_workers=options.num_workers,
        collate_fn=collate_batch,
    )

    valid_dataset = LoadDataset(
        valid_data, options.data.token_paths, crop=options.data.crop, transform=transformed, rgb=options.data.rgb
    )
    '''(Pdb) valid_dataset.data
[{'path': '/opt/ml/input/data/train_dataset/images/train_00008.jpg', 'truth': {'text': 'F \\left( 0 , \\sqrt { a ^ { 2 } + b ^ { 2 } } \\right) , 
\\left( 0 , - \\sqrt { a ^ { 2 } + b ^ { 2 } } \\right)', 'encoded': [0, 178, 229, 14, 93, 13, 224, 205, 236, 224, 177, 213, 108, 41, 236, 224, 177, 213, 213, 137, 93, 
229, 14, 93, 214, 13, 224, 205, 236, 224, 177, 213, 108, 41, 236, 224, 177, 213, 213, 137, 1]}}, {'path': '/opt/ml/input/data/train_dataset/images/train_00014.jpg', 
'truth': {'text': 'M P _ { e } = \\lim _ { \\Delta l \\to 0 } \\frac { g \\left( l + \\Delta l \\right) - g \\left( l \\right) } { \\Delta l } = \\frac { d g } { d l }', 
'encoded': [0, 54, 57, 10, 224, 243, 213, 180, 168, 10, 224, 211, 240, 163, 14, 213, 82, 224, 18, 229, 240, 108, 211, 240, 137, 214, 18, 229, 240, 137, 213, 224, 211, 240,
 213, 180, 82, 224, 215, 18, 213, 224, 215, 240, 213, 1]}}, ...
    '''
    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=options.batch_size,
        shuffle=False,
        num_workers=options.num_workers,
        collate_fn=collate_batch,
    )

    return train_data_loader, valid_data_loader, train_dataset, valid_dataset
