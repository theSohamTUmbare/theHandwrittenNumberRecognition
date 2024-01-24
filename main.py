import network
import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# training_data_list = list(training_data)
# print(training_data_list[0])  # Access the first (image, label) tuple

net = network.Network([784, 40, 10])         ## 784 nuerons in 1st 30 in 2nd and 10 in last layer

net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
