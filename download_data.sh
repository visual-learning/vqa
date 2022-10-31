cd $DATA_PATH
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Annotations_Train_mscoco.zip
unzip Annotations_Train_mscoco.zip
rm Annotations_Train_mscoco.zip
# you should see now mscoco_train2014_annotations.json

wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Annotations_Val_mscoco.zip
unzip Annotations_Val_mscoco.zip
rm Annotations_Val_mscoco.zip
# you should see now mscoco_val2014_annotations.json

wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Questions_Train_mscoco.zip
unzip Questions_Train_mscoco.zip
rm Questions_Train_mscoco.zip
# you should see now OpenEnded_mscoco_train2014_questions.json

wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Questions_Val_mscoco.zip
unzip Questions_Val_mscoco.zip
rm Questions_Val_mscoco.zip
# you should see now OpenEnded_mscoco_val2014_questions.json

wget http://images.cocodataset.org/zips/train2014.zip
unzip train2014.zip
rm train2014.zip
# you should see now train2014/

wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip
rm val2014.zip
# you should see now val2014/