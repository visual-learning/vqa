"""Dataset class for VQA."""

from collections import Counter
import os
import re

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from vqa_api import VQA


class VQADataset(Dataset):
    """VQA dataset class."""

    def __init__(self, image_dir, question_json_file_path,
                 annotation_json_file_path, image_filename_pattern,
                 answer_to_id_map=None, answer_list_length=5216, size=224):
        """
        Initialize dataset.

        Args:
            image_dir (str): Path to the directory with COCO images
            question_json_file_path (str): Path to json of questions
            annotation_json_file_path (str): Path to json of mapping
                images, questions, and answers together
            image_filename_pattern (str): The pattern the filenames
                (eg "COCO_train2014_{}.jpg")
        """
        self._vqa = TODO  # load the VQA api
        # also initialize whatever you need from self._vqa
        self._image_dir = image_dir
        self._image_filename_pattern = image_filename_pattern

        # Publicly accessible dataset parameters
        self.answer_list_length = answer_list_length + 1
        self.unknown_answer_index = answer_list_length
        self.size = size

        # Create the answer map if necessary
        keys = sorted(self._vqa.qa.keys())
        if answer_to_id_map is None:
            all_answers = [
                ' '.join([
                    re.sub(r'\W+', '', word)
                    for word in a['answer'].lower().split()
                ])
                for key in keys
                for a in self._vqa.qa[key]['answers']
            ]
            self.answer_to_id_map = self._create_id_map(
                all_answers, answer_list_length
            )
        else:
            self.answer_to_id_map = answer_to_id_map

    def _create_id_map(self, word_list, max_list_length):
        """
        Create a str-id map for most common words.

        Args:
            word_list: a list of str, with most frequent elements picked out
            max_list_length: the number of strs picked

        Returns:
            A map (dict) from str to id (rank)
        """
        common = Counter(word_list).most_common(max_list_length)
        return {tup[0]: t for t, tup in enumerate(common)}

    def __len__(self):
        return TODO

    def __getitem__(self, idx):
        """
        Load an item of the dataset.

        Args:
            idx: index of the data item

        Returns:
            A dict containing torch tensors for image, question and answers
        """
        q_anno = TODO  # load annotation
        q_str = TODO  # question in str format

        # Load and pre-process image
        name = str(q_anno['image_id'])
        if len(name) < 12:
            name = '0' * (12 - len(name)) + name
        img_name = self._image_filename_pattern.format(name)
        _img = Image.open(
            os.path.join(self._image_dir, img_name)
        ).convert('RGB')
        width, height = _img.size
        max_wh = max(width, height)
        mean_ = [0.485, 0.456, 0.406]
        std_ = [0.229, 0.224, 0.225]
        preprocessing = transforms.Compose([
            transforms.Pad((0, 0, max_wh - width, max_wh - height)),
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
            transforms.Normalize(mean_, std_)
        ])
        img = preprocessing(_img)
        orig_prep = transforms.Compose([
            transforms.Pad((0, 0, max_wh - width, max_wh - height)),
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor()
        ])
        orig_img = orig_prep(_img)

        # Encode answer to tensor
        a_tensor = torch.zeros(len(q_anno['answers']), self.answer_list_length)
        for a, ans in enumerate(q_anno['answers']):
            a_tensor[
                a, self.answer_to_id_map.get(
                    ' '.join([
                        re.sub(r'\W+', '', word)
                        for word in ans['answer'].lower().split()
                    ]),
                    self.unknown_answer_index
                )
            ] = 1
        a_tensor = a_tensor.any(0).float()  # keep all answers!

        return {
            'image': img,
            'question': q_str,
            'answers': a_tensor,
            'orig_img': orig_img
        }
