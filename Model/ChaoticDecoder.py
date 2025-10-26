'''
    Copyright:      JarvisLee
    Date:           5/1/2021
    File Name:      ChaoticDecoder.py
    Description:    The Chaotic different types of RNNs based Decoder.
'''

# Import the necessary library.
import torch
import torch.nn as nn
from Model.ChaoticAttention import ChaoticAttention
from .old_ChaoticLSTM import ChaoticLSTM
from .old_LeeOscillator import LeeOscillator


# Create the class for the Chaotic Decoder.
class ChaoticDecoder(nn.Module):
    '''
        The Chaotic different types of RNNs based Encoder.\n
        Params:\n
            - inputSize (integer), The output size of the Chaotic Decoder.\n
            - outputSize (integer), The output size of the Chaotic Decoder.\n 
            - Lee (LeeOscillator), The Lee-Oscillator.\n
            - bidirection (bool), The boolean to check whether apply the Bi-Model.\n
            - attention (bool), The boolean to check whether use the Attention Mechanism.\n
            - LSTM (bool), The boolean to check whether use the LSTM unit.\n
            - GRU (bool), The boolean to check whether use the GRU unit.\n
            - RNN (bool), The boolean to check whether use the RNN unit.\n
    '''

    # Create the constructor.
    def __init__(self, inputSize, outputSize, bidirection=False, attention=False):
        # Create the super constructor.
        super(ChaoticDecoder, self).__init__()
        # Get the member variables.
        if bidirection:
            self.inputSize = 2 * inputSize
        else:
            self.inputSize = inputSize
        # Create the Chaotic attention.
        if attention:
            print("The Decoder applied Attention.")
            self.CAttention = ChaoticAttention(self.inputSize).cuda()
        else:
            print("The Decoder didn't apply Attention.")
            self.CAttention = None
        # Create the Chaotic Decoder.
        print("The Decoder applied LSTM unit.")
        self.unit = ChaoticLSTM(inputSize=self.inputSize, bidirection=bidirection).cuda()
        # self.unit = ChaoticLSTM(inputSize=self.inputSize, bidirection=bidirection)

        # Create the Fully Connected Layer.
        self.fc = nn.Linear(4 * self.inputSize, outputSize).cuda()

    # Create the forward propagation.
    def forward(self, x, hs=None):
        # Get the output.
        outputs = []
        # Get the batch size.
        bs = x.shape[1]
        # Get the hidden.
        if hs is None:
            ht, ct = (torch.zeros(bs, self.inputSize).cuda(), torch.zeros(bs, self.inputSize).cuda())
        else:
            ht, ct = hs
        # Check whether apply the attention.
        if self.CAttention is None:
            # Get the output.
            for _ in range(4):
                output, (ht, ct) = self.unit(ht.unsqueeze(1), (ht, ct))
                outputs.append(output)
        else:
            # Get the output.
            for _ in range(4):
                # Compute the attention.
                context = self.CAttention(x, ht)
                # Compute the output.
                output, (ht, ct) = self.unit(context, (ht, ct))
                outputs.append(output)
        # Get the output.
        outputs = torch.cat(outputs, dim=1)
        outputs = outputs.view(outputs.shape[0], outputs.shape[1] * outputs.shape[2])
        outputs = self.fc(outputs)
        # Return the output.
        return outputs


# Create the main function to test the Chaotic Decoder.
if __name__ == "__main__":
    # Get the Lee-Oscillator.
    Lee = LeeOscillator()
    # Create the Chaotic LSTM Decoder with Attention.
    CDecoder = ChaoticDecoder(10, 1, bidirection=False, attention=False).cuda()
    # Test the Chaotic LSTM Decoder with Attention.
    # x = torch.randn((32, 10, 20)).cuda()
    x = torch.randn((10, 32, 20)).cuda()
    hs = (torch.zeros(32, 20).cuda(), torch.zeros(32, 20).cuda())
    output = CDecoder(x)
    print(output.shape)