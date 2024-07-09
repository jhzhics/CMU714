import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    seq = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        norm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(drop_prob),
        nn.Linear(hidden_dim, dim),
        norm(dim)
    )
    res = nn.Residual(seq)
    block = nn.Sequential(res, nn.ReLU())
    return block


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):

    resnet = nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU(), 
                           *[ResidualBlock(dim=hidden_dim, hidden_dim=hidden_dim//2, norm=norm, drop_prob=drop_prob) for _ in range(num_blocks)],
                           nn.Linear(hidden_dim, num_classes))
    return resnet

def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    if opt is None:
        model.eval()
    else:
        model.train()
    loss_fn = nn.SoftmaxLoss()
    total_loss = []
    total_err = 0
    for X, y in dataloader:
        logits = model(X)
        loss = loss_fn(logits, y)
        total_err += np.sum(np.argmax(logits.numpy(), axis=1) != y.numpy())
        total_loss.append(loss.numpy().tolist())
        if model.training:
            opt.reset_grad()
            loss.backward()
            opt.step()
            print(f"Loss: {loss}")
    total_loss = np.array(total_loss)
    average_loss = np.mean(total_loss)
    average_err = total_err / len(dataloader.dataset)
    return average_err, average_loss



def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    train_img_file = data_dir+"/train-images-idx3-ubyte.gz"
    train_label_file = data_dir+"/train-labels-idx1-ubyte.gz"
    test_img_file = data_dir+"/t10k-images-idx3-ubyte.gz"
    test_label_file = data_dir+"/t10k-labels-idx1-ubyte.gz"
    train_dataset = ndl.data.mnist_dataset.MNISTDataset\
    (train_img_file,train_label_file)
    test_dataset = ndl.data.mnist_dataset.MNISTDataset\
    (test_img_file,test_label_file)
    
    train_dataloader=ndl.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader=ndl.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model = MLPResNet(784, hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for _ in range(epochs):
        train_err, train_loss = epoch(train_dataloader, model, opt)
    
    model.eval()
    test_err, test_loss = epoch(test_dataloader, model)

    return train_err, train_loss, test_err, test_loss



if __name__ == "__main__":
    train_mnist(data_dir="../data")
