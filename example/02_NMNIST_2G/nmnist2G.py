import sys, os
CURRENT_TEST_DIR = os.getcwd()
sys.path.append(CURRENT_TEST_DIR + "/../../src")

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import slayerSNN as snn
from learningStats import learningStats
import zipfile

netParams = snn.params('network.yaml')


# Dataset definition
class nmnistDataset(Dataset):
    def __init__(self, datasetPath, sampleFile, samplingTime, sampleLength):
        self.path = datasetPath
        self.samples = np.loadtxt(sampleFile).astype('int')
        self.samplingTime = samplingTime
        self.nTimeBins = int(sampleLength / samplingTime)

    def __getitem__(self, index):
        inputIndex = self.samples[index, 0]
        classLabel = self.samples[index, 1]

        inputSpikes = snn.io.read2Dspikes(
            self.path + str(inputIndex.item()) + '.bs2'
        ).toSpikeTensor(torch.zeros((2, 34, 34, self.nTimeBins)),
                        samplingTime=self.samplingTime)
        desiredClass = torch.zeros((10, 1, 1, 1))
        desiredClass[classLabel, ...] = 1
        return inputSpikes, desiredClass, classLabel

    def __len__(self):
        return self.samples.shape[0]


class Goole_module(torch.nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Goole_module, self).__init__()

        def down_kernel(in_channel, out_channel):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channel, out_channel, 1, 1),
                torch.nn.BatchNorm2d(out_channel),
                torch.nn.LeakyReLU(0.1),
            )

        self.conv_3 = torch.nn.Sequential(
            torch.nn.Conv2d(int(out_channel / 2), out_channel, 3, 1, 1),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.LeakyReLU(0.1),
        )

        self.conv_5 = torch.nn.Sequential(
            torch.nn.Conv2d(int(out_channel / 2), out_channel, 5, 1, 2),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.LeakyReLU(0.1),
        )
        self.conv_1_1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, out_channel, 1, 1),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.LeakyReLU(0.1),
        )  # 1 -> 16
        self.conv_1_2 = down_kernel(in_channel, int(out_channel / 2))  # 1 -> 8 -> 16 ?
        self.conv_1_3 = down_kernel(in_channel, int(out_channel / 2))  # 1 -> 8 -> 16 ?
        self.conv_1_4 = down_kernel(in_channel, out_channel)
        self.pool = torch.nn.MaxPool2d(2)

    def forward(self, x):

        x1 = self.conv_1_1(x)
        x2 = self.conv_3(self.conv_1_2(x))
        x3 = self.conv_5(self.conv_1_3(x))
        x4 = self.conv_1_4(self.pool(x))
        return torch.cat((x1, x2, x3, x4), 1)


# Network definition
class Network(torch.nn.Module):
    def __init__(self, netParams):
        super(Network, self).__init__()
        # initialize slayer
        slayer = snn.layer(netParams['neuron'], netParams['simulation'])
        self.slayer = slayer
        # define network functions
        self.conv1 = slayer.conv(2, 16, 5, padding=1)
        self.conv2 = slayer.conv(16, 32, 3, padding=1)
        self.conv3 = slayer.conv(32, 64, 3, padding=1)
        self.convg1 = Goole_module(34, 34).to(device)  # 1,512,512 -> 64,256,256
        self.convg2 = Goole_module(68, 8).to(device)  # 1,512,512 -> 64,256,256
        self.pool1 = slayer.pool(2)
        self.pool2 = slayer.pool(2)
        self.unpool1 = slayer.unpool(2)
        self.drop = slayer.dropout(0.5)
        self.fc1 = slayer.dense((2, 16, 2), 10)


    def forward(self, spikeInput):
        batch1 = len(spikeInput)
        tempGoogle1 = self.convg1(self.slayer.psp(spikeInput[0]))
        shape = list(tempGoogle1.shape)
        shape.insert(0, batch1)
        spikeGoogle1 = torch.zeros(shape)
        for batch_n in range(1, batch1):
            tempGoogle = self.convg1(self.slayer.psp(spikeInput[batch_n]))
            spikeGoogle1[batch_n] = tempGoogle

        spikeLayer2 = self.slayer.spike(self.pool1(self.slayer.psp(spikeGoogle1.to(device))))  # 16, 16, 16
        batch = len(spikeLayer2)
        tempGoogle1 = self.convg2(self.slayer.psp(spikeLayer2[0]))

        shape = list(tempGoogle1.shape)
        shape.insert(0, batch)
        spikeGoogle2 = torch.zeros(shape)
        for batch_n in range(1, batch):
            tempGoogle = self.convg2(self.slayer.psp(spikeLayer2[batch_n]))
            spikeGoogle2[batch_n] = tempGoogle
        # spikeLayer2 = self.slayer.spike(self.convG(self.slayer.psp(spikeGoogle.to(device))))  # 8,  8, 64
        # spikeLayer3 = self.slayer.spike(self.conv2(self.slayer.psp(self.drop(spikeLayer2))))  # 16, 16, 32
        spikeLayer3 = self.slayer.spike(self.pool2(self.slayer.psp(spikeGoogle2.to(device))))  # 8,  8, 32
        # spikeOut = self.slayer.spike(self.fc1(self.slayer.psp(spikeLayer3)))  # 10
        spikeOut = self.slayer.spike(self.fc1(self.slayer.psp(spikeLayer3)))  # 10
        return spikeOut


if __name__ == '__main__':
    # Extract NMNIST samples
    with zipfile.ZipFile('NMNISTsmall.zip') as zip_file:
        for member in zip_file.namelist():
            if not os.path.exists('./' + member):
                zip_file.extract(member, './')

    # Define the cuda device to run the code on.
    device = torch.device('cuda')
    # Use multiple GPU's if available
    # device = torch.device('cuda:2') # should be the first GPU of deviceIDs
    # deviceIds = [2, 3, 1]

    # Create network instance.
    net = Network(netParams).to(device)
    # Split the network to run over multiple GPUs
    # net = torch.nn.DataParallel(Network(netParams).to(device), device_ids=deviceIds)

    # Create snn loss instance.
    error = snn.loss(netParams).to(device)

    # Define optimizer module.
    # optimizer = torch.optim.AdamW(net.parameters(), lr=0.001,weight_decay=0.01)
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.001, amsgrad=True)
    # Dataset and dataLoader instances.
    trainingSet = nmnistDataset(datasetPath=netParams['training']['path']['train_in'],
                                sampleFile=netParams['training']['path']['train'],
                                samplingTime=netParams['simulation']['Ts'],
                                sampleLength=netParams['simulation']['tSample'])
    trainLoader = DataLoader(dataset=trainingSet, batch_size=12, shuffle=False, num_workers=4)

    testingSet = nmnistDataset(datasetPath=netParams['training']['path']['test_in'],
                               sampleFile=netParams['training']['path']['test'],
                               samplingTime=netParams['simulation']['Ts'],
                               sampleLength=netParams['simulation']['tSample'])
    testLoader = DataLoader(dataset=testingSet, batch_size=12, shuffle=False, num_workers=4)

    # Learning stats instance.
    stats = learningStats()

    # # Visualize the network.
    # for i in range(5):
    #   input, target, label = trainingSet[i]
    #   snn.io.showTD(snn.io.spikeArrayToEvent(input.reshape((2, 34, 34, -1)).cpu().data.numpy()))

    # training loop
    for epoch in range(100):
        tSt = datetime.now()

        # Training loop.
        for i, (input, target, label) in enumerate(trainLoader, 0):
            # Move the input and target to correct GPU.
            input = input.to(device)
            target = target.to(device)

            # Forward pass of the network.
            output = net.forward(input)

            # Gather the training stats.
            stats.training.correctSamples += torch.sum(snn.predict.getClass(output) == label).data.item()
            stats.training.numSamples += len(label)

            # Calculate loss.
            loss = error.numSpikes(output, target)

            # Reset gradients to zero.
            optimizer.zero_grad()

            # Backward pass of the network.
            loss.backward()

            # Update weights.
            optimizer.step()

            # Gather training loss stats.
            stats.training.lossSum += loss.cpu().data.item()

            # Display training stats.
            if i%50==0:stats.print(epoch, i, (datetime.now() - tSt).total_seconds())

        # Testing loop.
        # Same steps as Training loops except loss backpropagation and weight update.
        for i, (input, target, label) in enumerate(testLoader, 0):
            input = input.to(device)
            target = target.to(device)

            output = net.forward(input)

            stats.testing.correctSamples += torch.sum(snn.predict.getClass(output) == label).data.item()
            stats.testing.numSamples += len(label)

            loss = error.numSpikes(output, target)
            stats.testing.lossSum += loss.cpu().data.item()
            if i%10==0:stats.print(epoch, i)

        # Update stats.
        stats.update()

    # Plot the results.
    plt.figure(1)
    plt.semilogy(stats.training.lossLog, label='Training')
    plt.semilogy(stats.testing.lossLog, label='Testing')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.figure(2)
    plt.plot(stats.training.accuracyLog, label='Training')
    plt.plot(stats.testing.accuracyLog, label='Testing')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()
