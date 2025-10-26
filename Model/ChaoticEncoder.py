'''
    Copyright:      JarvisLee
    Date:           5/31/2021
    File Name:      ChaoticEncoder.py
    Description:    The Chaotic different types of RNNs based Encoder.
'''

# Import the necessary library.
import torch
import torch.nn as nn
from .old_ChaoticLSTM import ChaoticLSTM
from .old_LeeOscillator import LeeOscillator

cuda = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create the class for the Chaotic Encoder.
class ChaoticEncoder(nn.Module):
    '''
        The Chaotic different types of RNNs based Encoder.\n
        Params:\n
            - inputSize (integer), The input size of the Chaotic Encoder.\n
            - hiddenSize (integer), The output size of the Chaotic Encoder.\n
            - Lee (LeeOscillator), The Lee-Oscillator.\n
            - chaotic (bool), The boolean to check whether use the Chaotic Mode.\n
            - bidirection (bool), The boolean to check whether apply the Bi-Model.\n
            - LSTM (bool), The boolean to check whether use the LSTM unit.\n
            - GRU (bool), The boolean to check whether use the GRU unit.\n
            - RNN (bool), The boolean to check whether use the RNN unit.\n
    '''

    # Create the constructor.
    def __init__(self, inputSize, bidirection=False):
        # Create the super constructor.
        super(ChaoticEncoder, self).__init__()
        # Create the Chaotic Encoder.
        if bidirection == True:
            print("The Encoder applied Bi-LSTM unit.")
        else:
            print("The Encoder applied LSTM unit.")
        self.unit = ChaoticLSTM(inputSize=inputSize, bidirection=bidirection).to(cuda)

    # Create the forward propagation.
    def forward(self, x):
        # Compute the Chaotic Long Short-Term Memory Unit.
        # output, hidden = self.unit(x)
        hidden = self.unit(x)
        # Return the output and hidden.
        return hidden


# Create the main function to test the Chaotic Encoder.
if __name__ == "__main__":
    # Get the Lee-Oscillator.
    Lee = LeeOscillator()
    # Create the Encoder.
    CEncoder = ChaoticEncoder(46, bidirection=False).cuda()
    # Test the Encoder.
    x = torch.randn((32, 10, 46)).cuda()
    print(x.shape)
    output, hidden = CEncoder(x)
    print(output.shape)
    print(hidden[0].shape)
    print(hidden[1].shape)

