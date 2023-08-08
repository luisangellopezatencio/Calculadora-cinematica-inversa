from torch.utils.data import Dataset
class MyDataset(Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        input_data = self.inputs[index]
        output_data = self.outputs[index]
        return input_data, output_data