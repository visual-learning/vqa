# Assignment 4: Visual Question Answering with PyTorch

- [Visual Learning and Recognition (16-824) Spring 2021](https://visual-learning.cs.cmu.edu/index.html)
- Modified by: [Qichen Fu](https://fuqichen1998.github.io/), [Ziyan Wang](https://ziyanw1.github.io/)
- Created By: [Sam Powers](https://www.ri.cmu.edu/ri-people/samantha-powers/), [Kenny Marino](http://kennethmarino.weebly.com/), [Donglai Xiang](https://xiangdonglai.github.io/)
- TAs: [Qichen Fu](<https://fuqichen1998.github.io/>)
- Please post questions, if any, on the piazza for HW4.
- Total points: 90 + 20(bonus)
- Due Date: April 20, 2022 at 11:59pm EST.
- Please start EARLY!

In this assignment you will do three main things:

1. Load the VQA dataset [1]
2. Implement Simple Baseline for Visual Question Answering [2]
3. Implement Hierarchical Question-Image Co-Attention for Visual Question Answering [3]

**Submission Requirements**:

- Please submit your report as well as your code.
- At the beginning of the report, please include a section which lists all the commands for TAs to run your code.
- You should include a Google Drive link to download your trained models and tensorboard files.
- You should also mention any collaborators or other sources used for different parts of the assignment.

## Software setup

We will use the following python libraries for the homework:

1. PyTorch 1.6.0
2. VQA Python API (https://github.com/GT-Vision-Lab/VQA)
3. tensorboardX
4. Any other standard libraries you wish to use (numpy, scipy, PIL, etc). If you're unsure whether a library is allowed, please ask.

Please use Python 3.6. The VQA API is in Python 2.7; for convenience we have provided a version that has been
converted to work with Python 3.

Everything you are asked to implement is in the folder student_code. What is already provided there is intended as a launching point for
your implementation. Feel free to add arguments, restructure, etc.

## Task 1: Data Loader (30 points)

In this task you will write a dataset loader for VQA (1.0 Real Images, Open-Ended). You should look over the original VQA paper [1] to get an idea for this dataset and the task you're about to do.

More specifically your goal for this task is to implement a subclass of `torch.utils.data.Dataset` (https://pytorch.org/docs/stable/data.html) to provide easy access to the VQA data for the latter parts of this assignment.

### Before you start

The full dataset itself is quite large (the COCO training images are ~13 GB, for instance). For convenience we've provided an extremely trimmed down version in the test folder (one image and a couple corresponding questions and annotations) for your convenience. The test_vqa_dataset.py file in the test folder has a couple simple unit tests that call your implementation using this small dataset. These can be run using `python -m unittest discover test`. You can modify the code to adapt to your own need.

The unit tests are quite easy, just a sanity check. The real test will come as you start using it to train your nets. Don't worry if you find yourself coming back to Task 1 and improving it as you progress through the assignment.

### Download dataset

For this assignment, you need to download the train and validation data: https://visualqa.org/vqa_v1_download.html (we only need 'Real Images', not 'Abstract Scenes').

1. You'll need to get all three things: the annotations, the questions, and the images for both the training and validation sets.
    1. We're just using the validation set for testing, for simplicity. (In other words, we're not creating a separate set for parameter tuning.)
1. If you're using AWS Volumes we suggest getting a volume with at least 50 GB for caching (more details in Task 3 below).
1. We're using VQA v1.0 Open-Ended for easy comparison to the baseline papers. Feel free to experiment with, for example, VQA 2.0 [4] if you feel inclined, though it is not required.

You can use `wget` to download the files to a single folder in AWS volumes. After that, `unzip` the files. Now you should have a directory `$DATA` containing the following items.

    mscoco_train2014_annotations.json
    mscoco_val2014_annotations.json
    OpenEnded_mscoco_train2014_questions.json
    OpenEnded_mscoco_val2014_questions.json
    train2014/
    val2014/

### Understand VQA API

Now let's take a look at the Python API for VQA. Suppose we are at the root directory of this cloned repository. Launch a Python 3 interpretor and run:

    >>> from external.vqa.vqa import VQA
    >>> vqa = VQA(annotation_json_file_path, question_json_file_path)

where `annotation_json_file_path` and `question_json_file_path` are paths to `mscoco_train2014_annotations.json` and `OpenEnded_mscoco_train2014_questions.json` in the `$DATA` directory. Use the `vqa` object to answer the following question.

**1.1 Which member function of the `VQA` class returns the IDs of all questions in this dataset? How many IDs are there?**

**1.2 What is the content of the question of ID `409380`? What is the ID of the image associated with this question?**
    Hint for later tasks: This image ID can be padded with 0s (and prefix and suffix) to obtain the image file name.

**1.3 What is the mostly voted answer for this question?**

### Prepare the dataset for PyTorch

If you can answer the above questions, you should be able to start writing the data loader code. In this section you will fill in the file `student_code/vqa_dataset.py`.

#### Closed vocabulary

For this assigment, we will use a closed vocabulary for the question embedding. In other words, we will choose a fixed vocabulary of words in the question sentences and feed them as input into our networks. In particular, we will choose a set of words that has the **highest frequency** in the training set. All the remaining words will be considered as an 'unknown' class. For more details, please refer to our course slides, or [these slides](https://www.dropbox.com/s/84dawq7agq6i6o5/Image%20Captioning%20and%20VQA.pdf?dl=0) from previous year.

**1.4 Finish the `_create_word_list` function. It should split a list of question sentences into a single list of words.** Hints: 1. Convert any upper case alphabet to lower case. 2. Remove all punctuations before splitting.

**1.5 Finish the `_create_id_map` function. It should pick out the most frequent strings and create a mapping to their IDs.**

**1.6 Using the previous functions, assign `self.question_word_to_id_map` in the `__init__` function.** It will be used to create one-hot embedding of questions later. Hint: the number of words to use is passed into the `__init__` as an argument.

**1.7 Assign `self.answer_to_id_map` in the `__init__` function. Different from the word-level question embedding, the answer embedding is sentence-level (one ID per sentence). Why is it?**

#### Utilize the PyTorch data loading mechanism

As you can see in our code, our `VqaDataset` class is [inherited](https://docs.python.org/3/tutorial/classes.html#inheritance) from `torch.utils.data.Dataset` class. This allows us to use the PyTorch data loading API that conveniently supports features including multi-thread data loading. Feel free to refer to the [official PyTorch tutorial](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html).

We need to implement two methods for the `VqaDataset` class: `__len__` and `__getitem__`. `__len__` should return the size of the dataset; it is called when you pass an instance of the class to `len()`. `__getitem__` returns the `idx`-th item of the dataset; it is called when an instance of the class is indexed using `[idx]`, or iterated through using for loop.

**1.8 Implement the `__len__` function of the `VqaDataset` class. Should the size of the dataset equal the number of images, questions or the answers? Show your reasoning.**

**1.9 Implement the `__getitem__` function. You need to (1) figure out what is the `idx`-th item of the dataset; (2) load the associated image; (3) encode the question and answers.**

1. For now, assume that `cache_location` and `pre_encoder` are both `None`. We will come back to this in Task 3.
2. After you load the image, apply `self._transform` if it exists; otherwise apply `torchvision.transforms.ToTensor`.
3. Create **word-level one-hot encoding** for the question. Make sure that your implementation handles words not in the vocabulary. You also need to handle sentensences of varying lengths. Check out the `self._max_question_length` parameter in the `__init__` function. How do you handle questions of different lengths? Describe in words, what is the dimension of your output tensor?
4. Create sentence-level **one-hot encoding** for the answers. 10 answers are provided for each question. Encode each of them and stack together. Again, make sure to handle the answers not in the answer list. What is the dimension of your output tensor?

## Task 2: Simple Baseline (30 points)

For this task you will implement the simple method described in [2]. This serves to validate your dataset and provide
a baseline to compare the future tasks against.

We've provided a skeleton to get you started. Here's a quick overview:

1. The entry point is main.py, which can be run with `python -m student_code.main` plus any arguments (refer to main.py for the list).
2. Main will, by default, create a SimpleBaselineExperimentRunner.py, which subclasses ExperimentRunnerBase.
3. ExperimentRunnerBase runs the training loop and validation.
4. SimpleBaselineExperimentRunner is responsible for creating the datasets, creating the model (SimpleBaselineNet), and running optimization.
    - In general anything SimpleBaseline-specific should be put in here, not in the base class.

You need not stick strictly to the established structure. Feel free to make changes as desired.

Be mindful of your AWS usage if applicable. If you find yourself using too much, you may wish to use a subset of the dataset for debugging,
for instance a particular question type (e.g "what color is the").

Feel free to refer to the official implementation in Torch (https://github.com/metalbubble/VQAbaseline),
for instance to find the parameters they used to avoid needing to do a comprehensive parameter search.

***

**2.1 This paper uses 'bag-of-words' for question representation. What are the advantage and disadvantage of this type of representation? How do you convert the one-hot encoding loaded in question 1.9 to 'bag-of-words'?**

Let's start with the network structure. This paper uses the output of pretrained GoogLeNet as visual features. An implementation of GoogLeNet is provided in `external/googlenet/googlenet.py`.

**2.2 What are the 3 major components of the network used in this paper? What are the dimensions of input and output for each of them (including batch size)? In `student_code/simple_baseline_net.py`, implement the network structure.**

**2.3 In `student_code/simple_baseline_experiment_runner.py`, set up transform applied to input images.** The transform will be passed into the dataset class. It should be a composition of

1. Resizing to fit network input size;
2. Normalize to [0, 1] and convert from (H, W, 3) to (3, H, W);
3. Subtract mean [0.485, 0.456, 0.406] and divide by standard deviation [0.229, 0.224, 0.225] computed from ImageNet for each channel.

Hint: check out `torchvision.transforms.Compose` and `torchvision.transforms.ToTensor`.

**2.4 In `student_code/simple_baseline_experiment_runner.py`, specify the arguments `question_word_to_id_map` and `answer_to_id_map` passed into `VqaDataset`. Explain how you are handling the training set and validation set differently.**

**2.5 In `student_code/simple_baseline_experiment_runner.py`, set up the PyTorch optimizer. In Section 3.2 of the paper, they explain that they use a different learning rate for word embedding layer and softmax layer. We recommend a learning rate of 0.8 for word embedding layer and 0.01 for softmax layer, both with SGD optimizer. Explain how this is achieved in your implementation.**

`SimpleBaselineExperimentRunner` is a subclass of `ExperimentRunnerBase`. This is a great way to enable code reuse and make your code more structured. Implementations in `ExperimentRunnerBase` should be generic, not specific to Task 2 or 3.

**2.6 In `student_code/experiment_runner_base.py`, get the predicted answer and ground truth answer.** Notice that 10 annotated answers are loaded for each question. You should do a majority voting to get a single ground truth answer for training.

The member function `ExperimentRunnerBase._optimize` is left to be implemented in its subclasses. This makes it a [pure virtual function](https://en.wikipedia.org/wiki/Virtual_function#Abstract_classes_and_pure_virtual_functions) from the perspective of Object-Oriented Programming (OOP).

**2.7 In `student_code/simple_baseline_experiment_runner.py`, implement the `_optimize` function. In Section 3.2 of the paper, they mention weight clip. This means to clip network weight data and gradients that have a large absolute value. We recommend a threshold of 1500 for the word embedding layer weights, 20 for the softmax layer weights, and 20 for weight gradients. What loss function do you use?**

**2.8 In `student_code/experiment_runner_base.py`, implement the `validate` function.** If you want to, you can shuffle the validation dataset and only use a subset of it (at least 1,000) each time.

**2.9 Use Tensorboard to graph your loss and validation accuracies as you train. During validation, also log the input image, input question, predicted answer and ground truth answer (one example per validation is enough). This helps you validate your network output.**

Now, we are ready to train the model. Aim for a validation accuracy of 50%, though anything over 40% is okay. Remember to specify `--log_validation` in your command line argument.

**2.10 Describe anything special about your implementation in the report. Include your figures of training loss and validation accuracy. Also show input, prediction and ground truth in 3 different iterations.**

## Task 3: Co-Attention Network (30 points)

In this task you'll implement [3]. This paper introduces three things not used in the Simple Baseline paper: hierarchical question processing, attention, and the use of recurrent layers.

The paper explains attention fairly thoroughly, so we encourage you to, in particular, closely read through section 3.3 of the paper.

To implement the Co-Attention Network you'll need to:

1. Implement the image caching method to allow large batch size.
2. Implement CoattentionExperimentRunner's optimize method.
3. Implement CoattentionNet
    - Encode the image in a way that maintains some spatial awareness; you may want to skim through [5] to get a sense for why they upscale the images.
    - Understand the hierarchical language embedding (words, phrases, question) and the alternating co-attention module we provided by referring to the paper.
    - Attend to each layer of the hierarchy, creating an attended image and question feature for each layer.
    - Combine these features to predict the final answer.

Once again feel free to refer to the [official Torch implementation](https://github.com/jiasenlu/HieCoAttenVQA).

***

The paper uses a batch_size of 300. One way you can make this work is to pre-compute the pretrained network's (e.g ResNet) encodings of your images and cache them, and then load those instead of the full images. This reduces the amount of data you need to pull into memory, and greatly increases the size of batches you can run. This is why we recommended you create a larger AWS Volume, so you have a convenient place to store this cache.

**3.1 Set up transform used in the Co-attention paper. The transform should be similar to question 2.3, except a different input size. What is the input size used in the Co-Attention paper [3]? Here, we use ResNet18 as the image feature extractor as we have prepared for you.** Similar to 2.4, specify the arguments `question_word_to_id_map` and `answer_to_id_map` passed into `VqaDataset`.

**3.2 In `student_code/vqa_dataset.py`, understand the caching and loading logic.** The basic idea is to check whether a cached file for an image exists. If not, load original image from the disk, **apply certain transform if necessary**, extract feature using the encoder, and cache it to the disk; if the cached file exists, directly load the cached feature. **Please feel free to modify this part if preferred.**

Once you understand this part, run `python -m student_code.run_resnet_encoder` plus any arguments (preferably with batch size 1).

1. It will call the data loader for both training and validation set, and start the caching process.
2. This process will take some time. You can check the progress by counting the number of files in the cache location.
3. Once all the images are pre-cached, the training process will run very fast despite the large batch size we use.
4. In the meanwhile, you can start working on the later questions.

**3.3 Implement Co-attention network in `student_code/coattention_net.py`. The paper proposes two types of co-attention: parallel co-attention and alternating co-attention. In this assignment, please focus on the alternating co-attention.**

We have implemented the **hierarchical question feature extractor** and the **alternating co-attention module** for you. Please make sure you understand them first by referring the paper, and then use them to implement the **\_\_init\_\_** and **forward** functions of the **CoattentionNet** class. You should add **no** new lines to the **\_\_init\_\_** function and input **less than 20** lines for the **forward** function.

**In the report, use you own words to answer the following questions.**

1. What are the three levels in the hierarchy of question representation? How do you obtain each level of representation?
2. What is attention? How does the co-attention mechanism work? Why do you think it can help with the VQA task?
3. Compared to networks we use in previous assignments, the co-attention network is quite complicated. How do you modularize your code so that it is easy to manage and reuse?

**3.4 In `student_code/coattention_experiment_runner.py`, set up the optimizer and implement the optimization step. The original paper uses RMSProp, but feel free to experiment with other optimizers.**

At this point, you should be able to train you network. You implementation in `student_code/experiment_runner_base.py` for Task 2 should be directly reusable for Task 3.

**3.5 Similar to question 2.10, describe anything special about your implementation in the report. Include your figures of training loss and validation accuracy. Compare the performance of co-attention network to the simple baseline.**

## Task 4: Custom Network  (20 bonus points)

Brainstorm some ideas for improvements to existing methods or novel ways to approach the problem.

For 10 extra points, pick at least one method and try it out. It's okay if it doesn't beat the baselines, we're looking for creativity here; not all interesting ideas work out.

**4.1 List a few ideas you think of (at least 3, the more the better).**

**(bonus) 4.2 Implementing at least one of the ideas. If you tweak one of your existing implementations, please copy the network to a new, clearly named file before changing it. Include the training loss and test accuracy graphs for your idea.**

## Relevant papers

[1] VQA: Visual Question Answering (Agrawal et al, 2016): https://arxiv.org/pdf/1505.00468v6.pdf

[2] Simple Baseline for Visual Question Answering (Zhou et al, 2015): https://arxiv.org/pdf/1512.02167.pdf

[3] Hierarchical Question-Image Co-Attention for Visual Question Answering (Lu et al, 2017):  https://arxiv.org/pdf/1606.00061.pdf

[4] Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering (Goyal, Khot et al, 2017):  https://arxiv.org/pdf/1612.00837.pdf

[5] Stacked Attention Networks for Image Question Answering (Yang et al, 2016): https://arxiv.org/pdf/1511.02274.pdf

## Submission Checklist

### Report

List of commands to run your code

Google Drive Link to your model and tensorboard file

Specification of collaborators and other sources

Your response to questions

- 1.1 (4 pts)
- 1.2 (4 pts)
- 1.3 (4 pts)
- 1.7 (4 pts)
- 1.8 (5 pts)
- 1.9.3 (5 pts)
- 1.9.4 (4 pts)
- 2.1 (4 pts)
- 2.2 (4 pts)
- 2.4 (4 pts)
- 2.5 (4 pts)
- 2.7 (4 pts)
- 2.10 (10 pts)
- 3.1 (4 pts)
- 3.3.1 (4 pts)
- 3.3.2 (4 pts)
- 3.3.3 (4 pts)
- 3.5 (14 pts)
- 4.1 (bonus 10 pts)
- 4.2 (bonus 10 pts)

### Files

Your `student_code` folder.
