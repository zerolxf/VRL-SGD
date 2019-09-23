python plot.py --path record/LeNet/identical/ --pos 0 --title "LeNet, MNIST" --save "lenet_identical" --alpha 0.4
python plot.py --path record/TextCNN/identical/ --title "TextCNN, DBPedia" --save "textCNN_identical" --pos 0 --alpha 0.4
python plot.py --path record/TransferLearning/identical/ --title "Transfer Learning, Tiny ImageNet" --save "transfer_learning_identical" --pos 0 --alpha 0.4 --st 0
python plot.py --path record/LeNet/non-identical/ --pos 0 --title "LeNet, MNIST" --save "lenet_non_identical" --alpha 0.7
python plot.py --path record/TextCNN/non-identical/ --title "TextCNN, DBPedia" --save "textCNN_non_identical" --pos 0 --alpha 0.7
python plot.py --path record/TransferLearning/non-identical/ --title "Transfer Learning, Tiny ImageNet" --save "transfer_learning_non_identical" --pos 0 --alpha 0.7 --st 0