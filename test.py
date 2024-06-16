import argparse
import time
from tqdm import tqdm
import evaluate
import random
import re
import unimernet.tasks as tasks

import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from tabulate import tabulate
from rapidfuzz.distance import Levenshtein
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from unimernet.common.config import Config

# imports modules for registration
from unimernet.datasets.builders import *
from unimernet.models import *
from unimernet.processors import *
from unimernet.tasks import *
from unimernet.processors import load_processor


class MathDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        raw_image = Image.open(self.image_paths[idx])
        if self.transform:
            image = self.transform(raw_image)
        return image


def load_data(image_path, math_file):
    """
    Load a list of image paths and their corresponding formulas.
    The function skips empty lines and lines without corresponding images.

    Args:
        image_path (str): The path to the directory containing the image files.
        math_file (str): The path to the text file containing the formulas.

    Returns:
        list, list: A list of image paths and a list of corresponding formula
    """
    image_names = [f for f in sorted(os.listdir(image_path))]
    image_paths = [os.path.join(image_path, f) for f in image_names]

    math_gts = []
    with open(math_file, 'r') as f:
        # load maths which 
        for i, line in enumerate(f, start=1):
            image_name = f'{i-1:07d}.png'
            if line.strip() and image_name in image_names:
                math_gts.append(line.strip())
    
    if len(image_paths) != len(math_gts):
        raise ValueError("The number of images does not match the number of formulas.")
    
    return image_paths, math_gts


def normalize_text(text):
    """Remove unnecessary whitespace from LaTeX code."""
    text_reg = r'(\\(operatorname|mathrm|text|mathbf)\s?\*? {.*?})'
    letter = '[a-zA-Z]'
    noletter = '[\\w_^\\d]'
    names = [x[0].replace(' ', '') for x in re.findall(text_reg, text)]
    text = re.sub(text_reg, lambda match: str(names.pop(0)), text)
    news = text
    while True:
        text = news
        news = re.sub(r'(?!\\ )(%s)\s+?(%s)' % (noletter, noletter), r'\1\2', text)
        news = re.sub(r'(?!\\ )(%s)\s+?(%s)' % (noletter, letter), r'\1\2', news)
        news = re.sub(r'(%s)\s+?(%s)' % (letter, noletter), r'\1\2', news)
        if news == text:
            break
    return text


def score_text(predictions, references):
    bleu = evaluate.load("bleu", keep_in_memory=True, experiment_id=random.randint(1,1e8))
    bleu_results = bleu.compute(predictions=predictions, references=references)

    lev_dist = []
    for p, r in zip(predictions, references):
        lev_dist.append(Levenshtein.normalized_distance(p, r))

    return {
        'bleu': bleu_results["bleu"],
        'edit': sum(lev_dist) / len(lev_dist)
    }


def setup_seeds(seed=3):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--result_path", type=str, help="Path to json file to save result to.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def main():

    setup_seeds()
    # Load Model and Processor
    start = time.time()
    cfg = Config(parse_args())
    task = tasks.setup_task(cfg)
    model = task.build_model(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vis_processor = load_processor('formula_image_eval', cfg.config.datasets.formula_rec_eval.vis_processor.eval)
    model.to(device)

    print(f'arch_name:{cfg.config.model.arch}')
    print(f'model_type:{cfg.config.model.model_type}')
    print(f'checkpoint:{cfg.config.model.finetuned}')
    print(f'='*100)

    end1 = time.time()

    # Generate prediction with MFR model
    print(f'Device:{device}')
    print(f'Load model: {end1 - start:.3f}s')

    # Load Data (image and corresponding annotations)
    val_names = [
        "Simple Print Expression(SPE)",
        "Complex Print Expression(CPE)",
        "Screen Capture Expression(SCE)",
        "Handwritten Expression(HWE)"
    ]
    image_paths = [
        "./data/PixSolve-test/spe",
        "./data/PixSolve-test/cpe",
        "./data/PixSolve-test/sce",
        "./data/PixSolve-test/hwe"
            ]
    math_files = [
        "./data/PixSolve-test/spe.txt",
        "./data/PixSolve-test/cpe.txt",
        "./data/PixSolve-test/sce.txt",
        "./data/PixSolve-test/hwe.txt"
    ]


    for val_name, image_path, math_file in zip(val_names, image_paths, math_files):
        image_list, math_gts = load_data(image_path, math_file)

        transform = transforms.Compose([
            vis_processor,
        ])

        dataset = MathDataset(image_list, transform=transform)
        dataloader = DataLoader(dataset, batch_size=32, num_workers=8)
        
        math_preds = []
        for images in tqdm(dataloader):
            images = images.to(device)
            with torch.no_grad():
                output = model.generate({"image": images})
            math_preds.extend(output["pred_str"])
            
        # Compute BLEU/METEOR/EditDistance
        norm_gts = [normalize_text(gt) for gt in math_gts]
        norm_preds = [normalize_text(pred) for pred in math_preds] 
        print(f'len_gts:{len(norm_gts)}, len_preds={len(norm_preds)}')
        print(f'norm_gts[0]:{norm_gts[0]}')
        print(f'norm_preds[0]:{norm_preds[0]}')

        p_scores = score_text(norm_preds, norm_gts)

        write_data= {
            "scores": p_scores,
            "text": [{"prediction": p, "reference": r} for p, r in zip(norm_preds, norm_gts)]
        }

        score_table = []
        score_headers = ["bleu", "edit"]
        score_dirs = ["⬆", "⬇"]

        score_table.append([write_data["scores"][h] for h in score_headers])

        score_headers = [f"{h} {d}" for h, d in zip(score_headers, score_dirs)]

        end2 = time.time()

        print(f'Evaluation Set:{val_name}')
        print(f'Inference Time: {end2 - end1}s')
        print(tabulate(score_table, headers=[*score_headers]))
        print('='*100)

if __name__ == "__main__":
    main()