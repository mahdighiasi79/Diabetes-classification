import preprocessing as pre
import torch


if __name__ == "__main__":
    t = torch.tensor([34, 56, 123])
    t = torch.softmax(t, dim=0)
    print(t)
