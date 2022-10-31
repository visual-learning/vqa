"""Base experiment runner class for VQA experiments."""
import argparse
import os

import torch
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import BaselineNet, TransformerNet
from vqa_dataset import VQADataset


class Trainer:
    """Train/test models on manipulation."""

    def __init__(self, model, data_loaders, args):
        self.model = model
        self.data_loaders = data_loaders
        self.args = args

        self.writer = SummaryWriter('runs/' + args.tensorboard_dir)

        self.optimizer = Adam(
            model.parameters(), lr=args.lr, betas=(0.0, 0.9), eps=1e-8
        )

        self._id2answer = {
            v: k
            for k, v in data_loaders['val'].dataset.answer_to_id_map.items()
        }
        self._id2answer[len(self._id2answer)] = 'Other'

    def run(self):
        # Set
        start_epoch = 0
        val_acc_prev_best = -1.0

        # Load
        if os.path.exists(self.args.ckpnt):
            start_epoch, val_acc_prev_best = self._load_ckpnt()

        # Eval?
        if self.args.eval or start_epoch >= self.args.epochs:
            self.model.eval()
            self.train_test_loop('val')
            return self.model

        # Go!
        for epoch in range(start_epoch, self.args.epochs):
            print("Epoch: %d/%d" % (epoch + 1, self.args.epochs))
            self.model.train()
            # Train
            self.train_test_loop('train', epoch)
            # Validate
            print("\nValidation")
            self.model.eval()
            with torch.no_grad():
                val_acc = self.train_test_loop('val', epoch)

            # Store
            if val_acc >= val_acc_prev_best:
                print("Saving Checkpoint")
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_acc": val_acc
                }, self.args.ckpnt)
                val_acc_prev_best = val_acc
            else:
                print("Updating Checkpoint")
                checkpoint = torch.load(self.args.ckpnt)
                checkpoint["epoch"] += 1
                torch.save(checkpoint, self.args.ckpnt)

        return self.model

    def _load_ckpnt(self):
        ckpnt = torch.load(self.args.ckpnt)
        self.model.load_state_dict(ckpnt["model_state_dict"], strict=False)
        self.optimizer.load_state_dict(ckpnt["optimizer_state_dict"])
        start_epoch = ckpnt["epoch"]
        val_acc_prev_best = ckpnt['best_acc']
        return start_epoch, val_acc_prev_best

    def train_test_loop(self, mode='train', epoch=1000):
        n_correct, n_samples = 0, 0
        for step, data in tqdm(enumerate(self.data_loaders[mode])):

            # Forward pass
            scores = self.model(
                data['image'].to(self.args.device),
                data['question']
            )
            answers = data['answers'].to(self.args.device)

            # Losses
            # Uncomment these if you want to assign less weight to 'other'
            # pos_weight = torch.ones_like(answers[0])
            # pos_weight[-1] = 0.1  # 'Other' has lower weight
            # and use the pos_weight argument
            # ^OPTIONAL: the expected performance can be achieved without this
            loss = TODO

            # Update
            if mode == 'train':
                # optimize loss
                TODO

            # Accuracy
            n_samples += len(scores)
            found = (
                F.one_hot(scores.argmax(1), scores.size(1))
                * answers
            ).sum(1)
            n_correct += (
                F.one_hot(scores.argmax(1), scores.size(1))
                * answers
            ).sum().item()  # checks if argmax matches any ground-truth

            # Logging
            self.writer.add_scalar(
                'Loss/' + mode, loss.item(),
                epoch * len(self.data_loaders[mode]) + step
            )
            if mode == 'val' and step == 0:  # change this to show other images
                _n_show = 3  # how many images to plot
                for i in range(_n_show):
                    self.writer.add_image(
                        'Image%d' % i, data['orig_img'][i].cpu().numpy(),
                        epoch * _n_show + i, dataformats='CHW'
                    )
                    # add code to show the question
                    # the gt answer
                    # and the predicted answer
                    # if your model's prediction is correct,
                    # show the predicted answer as gt
                    # so as to avoid confusion due to multiple correct answers
            # add code to plot the current accuracy
        acc = n_correct / n_samples
        print(acc)
        return acc


def main():
    """Run main training/test pipeline."""
    # Feel free to add more args, or change/remove these.
    parser = argparse.ArgumentParser(description='Load VQA.')
    parser.add_argument('--model', type=str, default='simple')
    parser.add_argument('--tensorboard_dir', type=str, default=None)
    parser.add_argument('--ckpnt', type=str, default=None)
    parser.add_argument('--data_path', type=str, default='./')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()
    data_path = args.data_path

    # Other variables
    args.train_image_dir = data_path + 'train2014/'
    args.train_q_path = data_path + 'OpenEnded_mscoco_train2014_questions.json'
    args.train_anno_path = data_path + 'mscoco_train2014_annotations.json'
    args.test_image_dir = data_path + 'val2014/'
    args.test_q_path = data_path + 'OpenEnded_mscoco_val2014_questions.json'
    args.test_anno_path = data_path + 'mscoco_val2014_annotations.json'
    if args.tensorboard_dir is None:
        args.tensorboard_dir = args.model
    if args.ckpnt is None:
        args.ckpnt = args.model + '.pt'
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Loaders
    train_dataset = VQADataset(
        image_dir=args.train_image_dir,
        question_json_file_path=args.train_q_path,
        annotation_json_file_path=args.train_anno_path,
        image_filename_pattern="COCO_train2014_{}.jpg"
    )
    val_dataset = VQADataset(
        image_dir=args.test_image_dir,
        question_json_file_path=args.test_q_path,
        annotation_json_file_path=args.test_anno_path,
        image_filename_pattern="COCO_val2014_{}.jpg",
        answer_to_id_map=train_dataset.answer_to_id_map
    )
    print(len(train_dataset), len(val_dataset))
    data_loaders = {
        mode: DataLoader(
            train_dataset if mode == 'train' else val_dataset,
            batch_size=args.batch_size,
            shuffle=mode == 'train',
            drop_last=mode == 'train',
            num_workers=4
        )
        for mode in ('train', 'val')
    }

    # Models
    if args.model == "simple":
        model = BaselineNet()
    elif args.model == "transformer":
        model = TransformerNet()
    else:
        raise ModuleNotFoundError()

    trainer = Trainer(model.to(args.device), data_loaders, args)
    trainer.run()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
