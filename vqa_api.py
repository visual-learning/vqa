"""
Modified/cleaner version of the VQA API for Python3.

Re-written by Nikos Gkanatsios for the purposes of CMU 16824.
------------------------------------------------------------

Copyright (c) 2014, Aishwarya Agrawal
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are
those
of the authors and should not be interpreted as representing official
policies,
either expressed or implied, of the FreeBSD Project.
"""

import json


class VQA:
    """VQA utilities class."""

    def __init__(self, annotation_file=None, question_file=None):
        """Initialize constructor."""
        self.dataset = {}
        self.questions = {}
        self.qa = {}
        self.qqa = {}
        self.img2qa = {}
        if annotation_file is not None and question_file is not None:
            print('loading VQA annotations and questions into memory...')
            dataset = json.load(open(annotation_file, 'r'))
            questions = json.load(open(question_file, 'r'))
            self.dataset = dataset
            self.questions = questions
            self.create_index()

    def create_index(self):
        """
        Create index dicts.

        Example of annotation:
        {
            'question_type': 'what',
            'multiple_choice_answer': 'curved',
            'answers': [
                {'answer': 'oval', 'answer_confidence': 'yes', 'answer_id': 1}
            ],
            'image_id': 487025,
            'answer_type': 'other',
            'question_id': 4870250
        }

        Example of question:
        {
            'question': 'What shape is the bench seat?',
            'image_id': 487025,
            'question_id': 4870250
        }

        Creates the following dictionaries:
            self.img2qa: {487025: [annos with image_id 487025]}
            self.qa: {4870250: anno dict for this question_id 4870250}
            self.qqa: {4870250: question dict for this question_id 4870250}
        """
        print('creating index...')
        img2qa = {ann['image_id']: [] for ann in self.dataset['annotations']}
        qa = {ann['question_id']: [] for ann in self.dataset['annotations']}
        qqa = {ann['question_id']: [] for ann in self.dataset['annotations']}
        for ann in self.dataset['annotations']:
            img2qa[ann['image_id']] += [ann]
            qa[ann['question_id']] = ann
        for ques in self.questions['questions']:
            qqa[ques['question_id']] = ques
        print('index created!')

        # create class members
        self.qa = qa
        self.qqa = qqa
        self.img2qa = img2qa

    def get_ques_ids(self, img_ids=[], ques_types=[], ans_types=[]):
        """
        Get question ids that satisfy given filter conditions.

        Args:
            img_ids    (int array): get question ids for given imgs
            ques_types (str array): get question ids for given question types
            ans_types  (str array): get question ids for given answer types

        Returns:
            ids (int array): integer array of question ids
        """
        img_ids = img_ids if type(img_ids) == list else [img_ids]
        ques_types = ques_types if type(ques_types) == list else [ques_types]
        ans_types = ans_types if type(ans_types) == list else [ans_types]

        if len(img_ids) == len(ques_types) == len(ans_types) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(img_ids) == 0:  # specific image ids
                anns = sum([
                    self.img2qa[imgId]
                    for imgId in img_ids if imgId in self.img2qa
                ], [])
            else:
                anns = self.dataset['annotations']
            anns = (
                anns if len(ques_types) == 0
                else
                [ann for ann in anns if ann['question_type'] in ques_types]
            )
            anns = (
                anns if len(ans_types) == 0
                else [ann for ann in anns if ann['answer_type'] in ans_types]
            )
        ids = [ann['question_id'] for ann in anns]
        return ids

    def get_img_ids(self, ques_ids=[], ques_types=[], ans_types=[]):
        """
        Get image ids that satisfy given filter conditions.

        Args:
            ques_ids   (int array): get image ids for given question ids
            ques_types (str array): get image ids for given question types
            ans_types  (str array): get image ids for given answer types

        Returns:
            ids (int array): integer array of image ids
        """
        ques_ids = ques_ids if type(ques_ids) == list else [ques_ids]
        ques_types = ques_types if type(ques_types) == list else [ques_types]
        ans_types = ans_types if type(ans_types) == list else [ans_types]

        if len(ques_ids) == len(ques_types) == len(ans_types) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(ques_ids) == 0:
                anns = [
                    self.qa[ques_id]
                    for ques_id in ques_ids if ques_id in self.qa
                ]
            else:
                anns = self.dataset['annotations']
            anns = (
                anns if len(ques_types) == 0
                else
                [ann for ann in anns if ann['question_type'] in ques_types]
            )
            anns = (
                anns if len(ans_types) == 0
                else [ann for ann in anns if ann['answer_type'] in ans_types]
            )
        ids = [ann['image_id'] for ann in anns]
        return ids

    def load_qa(self, ids=[]):
        """
        Load questions and answers with the specified question ids.

        Args:
            ids (int array): integer ids specifying question ids

        Returns:
            qa (object array): loaded qa objects
        """
        if type(ids) == list:
            return [self.qa[id_] for id_ in ids]
        elif type(ids) == int:
            return [self.qa[ids]]

    def show_qa(self, anns):
        """
        Display the specified annotations.

        Args:
            anns (array of object): annotations to display
        """
        if len(anns) == 0:
            return 0
        for ann in anns:
            ques_id = ann['question_id']
            print("Question: %s" % (self.qqa[ques_id]['question']))
            for ans in ann['answers']:
                print("Answer %d: %s" % (ans['answer_id'], ans['answer']))


if __name__ == "__main__":
    # Examples
    data_path = './'  # change this to your data path
    anno_file = data_path + 'mscoco_train2014_annotations.json'
    q_file = data_path + 'OpenEnded_mscoco_train2014_questions.json'
    vqa_api = VQA(anno_file, q_file)

    # How the data looks like
    print(len(vqa_api.qa))  # equal to the number of questions
    print(len(vqa_api.qqa))  # equal to the number of questions
    print(len(vqa_api.img2qa))  # equal to the number of images

    # Let's see an example
    print('Image id for question 4870250')
    img_id = vqa_api.get_img_ids([4870250])
    print(img_id)
    print('Question ids for image 487025')
    q_ids = vqa_api.get_ques_ids([487025])
    print(q_ids)
    print('Annotations for question 4870250')
    annos = vqa_api.load_qa([4870250])
    print(annos)
