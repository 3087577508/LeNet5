Command Instructions:

Quesiton 1:
(1) python data.py: Provide the MNIST dataset.
(2) python export_data.py: Download the MNIST dataset and convert the 28×28 image into PNG. Generate the test and train files in data at the same time.
(3) python prepare_rbf.py: Take the average bitmap of each class from the training PNG and convert it into an 84-dimensional center vector; save it in rbf_centers.pt.
(4) python train.py: Train the model and generate the model LeNet1.pth. Generate the train/test error of 20 epochs and the error curve error_plot.png at the same time
(5) python test1.py: Used for scoring and calculating the test accuracy
(6) python most_confusing.py: Find out which image of each digit was misclassified with the highest confidence.
(7) python confusion_matrix.py: Generate a 10×10 confusion matrix

Quesiton2:
(1) python train2.py: Train LeNet2. And calculate the training error and validation error after each epoch
(2) pyhton test2.py: Used to evaluate LeNet2 and scoring


Homework4.tex is the Latex file of 2 questions.