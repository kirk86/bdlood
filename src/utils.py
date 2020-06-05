# coding: utf-8

import os
import torch
import random
import logging
import numpy as np
import pandas as pd
from models import algorithms
import matplotlib.pyplot as plt
import torch.nn.functional as F

logger = logging.getLogger('OoD' + __name__)


def split_data(
        train_data, test_data=None, batch_size=1024, valid_size=0.1,
        num_workers=4, pin_memory=True,
        sampler=torch.utils.data.sampler.SubsetRandomSampler):

    n_samples = len(train_data)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * n_samples))
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = sampler(train_idx)
    valid_sampler = sampler(valid_idx)

    if test_data is not None:
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=batch_size, num_workers=num_workers,
            pin_memory=pin_memory)
    else:
        train_idx, test_idx = train_idx[split:], train_idx[:split]
        train_sampler = sampler(train_idx)
        test_sampler = sampler(test_idx)

        test_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, sampler=test_sampler,
            num_workers=num_workers, pin_memory=pin_memory)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory)

    valid_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def update_classwise_accuracies(preds, labels, class_correct, class_totals):
    correct = np.squeeze(preds.eq(labels.data.view_as(preds)))
    for i in range(labels.shape[0]):
        label = labels.data[i].item()
        class_correct[label] += correct[i].item()
        class_totals[label] += 1


def get_accuracies(class_names, class_correct, class_totals):
    accuracy = (100 * np.sum(list(class_correct.values())) /
                np.sum(list(class_totals.values())))
    class_accuracies = [(class_names[i], 100.0*(class_correct[i]/class_totals[i]))
                        for i in class_names.keys() if class_totals[i] > 0]
    return accuracy, class_accuracies


def flatten_tensor(x):
    return x.view(x.shape[0], -1)


def calculate_img_stats(dataset):
    imgs_ = torch.stack([img for img, _ in dataset], dim=3)
    imgs_ = imgs_.view(3, -1)
    imgs_mean = imgs_.mean(dim=1)
    imgs_std = imgs_.std(dim=1)
    return imgs_mean, imgs_std


def create_csv_from_folder(folder_path, outfile, cols=['id', 'path']):
    import glob

    f = glob.glob(folder_path+'/*.*')

    ids = []
    for elem in f:
        t = elem[elem.rfind('/')+1:]
        ids.append(t[:t.rfind('.')])
    data = {cols[0]: ids, cols[1]: f}
    df = pd.DataFrame(data, columns=cols)
    df.to_csv(outfile, index=False)


def model_summary(model):
    total = 0
    logger.debug('Trainable parameters:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            # logger.debug("{}\t{}".format(name, param.numel()))
            total += param.numel()
    logger.debug("---------------------------")
    logger.debug("Total,\t{}\n".format(total))


def plot_img(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    plt.show()


def fix_state_dict_dataparallel(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == 'module.':
            name = k[7:]  # remove 'module.'
        else:
            name = k
        new_state_dict[name] = v


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # auto select best possible algorithm
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def prepare_model(algorithm_name, optimizer_name, chkpt,
                  optim_params,
                  device, num_classes=10):
    model = getattr(algorithms, algorithm_name)
    model = model(num_classes=num_classes)
    optimizer = getattr(torch.optim, optimizer_name)
    optimizer = optimizer(model.parameters(), **optim_params)
    # checkpoint = load_chkpt(filename, config)
    model.load_state_dict(chkpt['model_state_dict'])
    optimizer.load_state_dict(chkpt['optimizer_state_dict'])
    model.to(device)
    logger.info("Best trained model: "
                "epoch: {}, lr: {}, valid_loss: {}, valid_acc: {}".format(
                    chkpt['epoch'], chkpt['lr'],
                    chkpt['valid_loss'], chkpt['valid_acc']))
    return model, optimizer


def load_chkpt(filename, path):
    path = os.path.join(path, filename + '.pt')
    if os.path.exists(path):
        checkpoint = torch.load('{}'.format(path))
        logger.info("Loading model...{}".format(filename))
        return checkpoint
    else:
        logger.info("No checkpoint found in {}!".format(path))


def save_model(filename, config, **kwargs):
    logger.debug("checkpoint...")
    if not os.path.isdir(config.chkpt_path):
        os.mkdir(config.chkpt_path)
    torch.save(kwargs, '{}/{}.pt'.format(config.chkpt_path, filename))


def schedule(epoch, lr_init=0.05, epochs=300):
    # adapted from https://github.com/wjmaddox/swa_gaussian/swag/run_swag.py
    t = epoch / epochs
    # t = (epoch) / (args.swa_start if args.swa else args.epochs)
    # lr_ratio = args.swa_lr / args.lr_init if args.swa else 0.01
    # lr_ratio = 0.02 / 0.01
    lr_ratio = 0.01
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    # return args.lr_init * factor
    return lr_init * factor


def rotate_img(img, deg):
    import scipy.ndimage as nd
    return nd.rotate(img, deg, reshape=False)


# This method rotates an image counter-clockwise and classify it for
# different degress of rotation.  It plots the highest classification
# probability along with the class label for each rotation degree.
def rotating_img_prediction(
        chkpt,
        model,
        method,
        loaders,
        n_samples,
        classes,
        uncertainty=None,
        threshold=0.5,
        num_classes=10,
        title='NoTitle'
):
    img = loaders['test'].dataset.data.squeeze()
    w, h = img.shape[0], img.shape[1]
    assert(w == h), "image dims should be identical!"
    rotation = 180
    total_imgs = int(rotation / 10) + 1
    degrees = []
    probabilities = []
    uncertainties = []
    scores = np.zeros((1, num_classes))
    rimgs = np.zeros((w, h * total_imgs))
    for i, deg in enumerate(np.linspace(0, rotation, total_imgs)):
        rotated_img = rotate_img(img, deg).reshape(w, h, -1)
        rimgs[:, i * w: (i+1) * h] = rotated_img
        if uncertainty is None:
            entropies, predictions, targets = evaluate(
                chkpt, model, method, loaders, n_samples=n_samples)
        else:
            entropies, predictions, targets = evaluate(
                chkpt, model, method, loaders, n_samples=n_samples)
            uncertainties.append(entropies)
        scores += predictions >= threshold
        degrees.append(deg)
        probabilities.append(predictions.squeeze())

    rimgs /= 255.0
    labels = np.arange(10)[scores.squeeze().astype(bool)]
    probabilities = np.array(probabilities)[:, labels]
    c = ['black', 'blue', 'red', 'brown', 'purple', 'cyan']
    marker = ['s', '^', 'o'] * 2
    labels = labels.tolist()
    for i in range(len(labels)):
        plt.plot(degrees, probabilities[:, i], marker=marker[i], c=c[i])

    if uncertainty is not None:
        labels += ['uncertainty']
        plt.plot(degrees, uncertainties, marker='<', c='red')

    plt.legend(classes[labels])

    plt.xlim([0, deg])
    plt.xlabel('Rotation Degree')
    plt.ylabel('Classification Probability')
    plt.title(title)
    plt.show()

    plt.figure(figsize=[14, 100])
    plt.imshow(1 - rimgs, cmap='gray')
    plt.axis('off')
    plt.show()


def save_nparray(filename, **kwargs):
    np.savez(out_path + filename + '.npz', **kwargs)


def parse_nparray(filename):
    path = out_path + filename + '.npz'
    npz_arr = np.load(path)
    return npz_arr


def test(model, test_loader, device):
    cum_loss = 0.0
    correct = 0.0
    n_samples = 0.0
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            output = F.log_softmax(output, dim=1)
            cum_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            sftmx_probs, labels = output.max(dim=1, keepdim=True)  # labels
            correct += (labels.view(-1) == target).sum().item()
            n_samples += len(output)
    epoch_loss = cum_loss / n_samples  # avg. over all mini-batches
    epoch_acc = correct / n_samples
    return epoch_loss, epoch_acc


def train(model, optimizer, train_loader, device):
    cum_loss = 0.0
    proxy_loss = 0.0
    correct = 0.0
    n_samples = 0.0
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        output = F.log_softmax(output, dim=1)
        # default behaviour of nll_loss return mean loss over mini-batch
        loss = F.nll_loss(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)  # labels
        # sum up batch correct predictions
        correct += pred.eq(target.view_as(pred)).sum().item()
        # sum up loss for every data point
        cum_loss += loss.item() * data.size(0)
        proxy_loss += F.nll_loss(output, target).item()
        n_samples += len(output)
        # logging information.
        # cum_loss += loss.data[0]
        # max_scores, max_labels = outputs.data.max(1)
        # correct += (max_labels == labels.data).sum()
        # counter += inputs.size(0)
    epoch_loss = cum_loss / n_samples  # avg. over all data points
    # epoch_loss = cum_loss / len(train_loader)  # avg. over num mini-batches
    epoch_acc = correct / n_samples
    return epoch_loss, epoch_acc
