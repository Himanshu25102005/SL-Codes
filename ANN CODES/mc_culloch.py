# Generated from: 02_second_ann.ipynb
# Converted at: 2026-05-04T20:29:51.849Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# ### Assignment 2
# 
# Title: To study and understand the concept AND NOT function using McCulloch-Pitts neural net by  using python. 
# 
# Problem Statement : Write a Python program to plot a few activation functions that are being used in neural networks.


# No installation required (uses only built-in Python)

# (Optional - if you later extend with libraries)
# !pip install numpy

#Option 1, SIMPLER Syntax
def andnot(x1, x2):
    w1 = 1     
    w2 = -1     
    threshold = 1
    
    net = x1*w1 + x2*w2
    
    if net >= threshold:
        return 1
    else:
        return 0


for x1 in [0, 1]:
    for x2 in [0, 1]:
        print(x1, x2, "->", andnot(x1, x2))

#Option 2, COMPLEX Syntax
class McCullochPittsNeuron:
    def __init__(self, weights, threshold):
        self.weights = weights
        self.threshold = threshold

    def activation(self, net):
        return 1 if net >= self.threshold else 0

    def forward(self, inputs):
        net = 0

        for i in range(len(inputs)):
            net = net + self.weights[i] * inputs[i]

        output = self.activation(net)
        return net, output


# ANDNOT neuron configuration
weights = [1, -1]   # x1 positive, x2 inhibitory
threshold = 1

neuron = McCullochPittsNeuron(weights, threshold)


# Generate truth table
print("x1 x2 | Net | Output (ANDNOT)")
print("-----------------------------")

for x1 in [0, 1]:
    for x2 in [0, 1]:
        net, output = neuron.forward([x1, x2])
        print(f"{x1}  {x2}  |  {net:>2}  |    {output}")