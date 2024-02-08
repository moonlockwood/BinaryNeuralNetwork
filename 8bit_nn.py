import torch
import torch.nn as nn
import torch.optim as optim

class BinaryNN(nn.Module):
    def __init__(self):
        super(BinaryNN, self).__init__()
        self.linear1 = nn.Linear(8, 16)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(16, 8)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

# convert a digit to an 8-bit binary representation
def encode_digit(digit): return [int(b) for b in '{:08b}'.format(digit)]
# convert an 8-bit binary representation back to a digit
def decode_digit(binary): return int(''.join(str(int(b)) for b in binary), 2)

# generate dataset with holdout digits not trained on for testing generalisation
def generate_dataset(holdout_digits):
    train_dataset = []
    test_dataset = []
    for digit in range(256):  # 0 to 255 for 8-bit representation
        next_digit = (digit + 1) % 256
        binary = encode_digit(digit)
        next_binary = encode_digit(next_digit)
        if digit in holdout_digits:
            test_dataset.append((digit, binary, next_binary))
        else:
            train_dataset.append((binary, next_binary))
    return train_dataset, test_dataset

def train_network(NeuralNetwork, dataset, epochs=100, learning_rate=0.1):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(NeuralNetwork.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        total_loss = 0
        for input_binary, target_binary in dataset:
            input_tensor = torch.tensor(input_binary, dtype=torch.float32)
            target_tensor = torch.tensor(target_binary, dtype=torch.float32)

            optimizer.zero_grad()
            output = NeuralNetwork(input_tensor)
            loss = criterion(output, target_tensor)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {total_loss/len(dataset):.4f}')
        
    print(f'final Loss: {total_loss/len(dataset):.6f}')

def threshold_output(output, threshold):
    return (output >= threshold).float()

def test_network(NeuralNetwork, digit):
    input_binary = encode_digit(digit)
    input_tensor = torch.tensor(input_binary, dtype=torch.float32)

    with torch.no_grad():
        output = NeuralNetwork(input_tensor)
    predicted_next_binary = threshold_output(output, 0.5).int().numpy()
    predicted_next_digit = decode_digit(predicted_next_binary)

    print(f'network output: {output}')
    print(f" Input Digit: {digit} (Binary: {input_binary})")
    print(f"Output Digit: {predicted_next_digit} (Binary: {predicted_next_binary.tolist()})")

holdout_digits = [7,64,93,129,167,185,203]  # Don't train on these digits, they will be the test dataset

print('generating dataset')
train_dataset, test_dataset = generate_dataset(holdout_digits)

print('training network')
NeuralNetwork = BinaryNN()
train_network(NeuralNetwork, train_dataset)

print('network ready')
print('-------------')
print(f'numbers not trained on: {", ".join(str(digit) for digit in holdout_digits)}')

while True:
    test_digit = input('number to test:')
    if test_digit == 'q':
        break
    test_network(NeuralNetwork, int(test_digit))
