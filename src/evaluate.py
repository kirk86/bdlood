import os
# import copy
import fire
import torch
import utils
import stats
import torch.nn.functional as F


utils.set_seed(2019)
logger = utils.setup_logger()
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

out_path = os.getcwd() + '/out/'


def main(algorithm, optimizer, dataset, num_classes=10,
         optim_params={'lr': 0.05, 'weight_decay': 5e-4, 'momentum': 0.9}):

    filename = algorithm + '_' + optimizer + '_' + dataset

    # prepare dataset
    logger.info("====== Evaluation ======")
    logger.info("Preparing dataset...{}".format(dataset))
    db = utils.Datasets(dataset)
    train, valid, test = db.split_image_data(train_data=db.train, test_data=db.test)

    # prepare model
    model, optimizer = utils.prepare_model(algorithm, optimizer, filename,
                                           optim_params, device, num_classes)

    # get model's output
    data_size = test.dataset.data.shape[0]
    targets = test.dataset.targets if dataset == "CIFAR10" else test.dataset.labels
    predictions = torch.zeros(data_size, num_classes)
    labels = torch.zeros(data_size, 1)
    logger.info("data: {} - targets {}.".format(data_size, len(targets)))
    cum_loss = 0.0
    correct = 0.0
    n_samples = 0.0
    model.eval()
    with torch.no_grad():
        for idx, (data, target) in enumerate(test):
            start = idx * data.size(0)
            end = (idx + 1) * data.size(0)
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            output = F.log_softmax(output, dim=1)
            if target.max() == 9:
                cum_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            sftmx_probs, predicted_labels = output.max(dim=1, keepdim=True)  # labels
            correct += (predicted_labels.view(-1) == target).sum().item()
            n_samples += len(output)
            predictions[start:end] = output
            labels[start:end] = predicted_labels
    predictions = predictions.cpu().numpy()
    labels = labels.view(-1).cpu().numpy()
    epoch_loss = cum_loss / n_samples  # avg. over all mini-batches
    epoch_acc = correct / n_samples
    logger.info("Loss = {}, Accuracy = {}, test set!".format(
        epoch_loss, epoch_acc))
    logger.info("Computing entropy on... test")
    stats.entropy(predictions, targets, filename + '_ENTROPY')
    # save model's outputs and targets valid data used in training
    logger.info("Computing calibration on... test")
    # compute and save reliability stats
    calibration = stats.calibration_curve(filename + '_ENTROPY')
    utils.save_nparray(filename + '_CALIBRATION', **calibration)
    logger.info("====== Evaluation End ======\n\n")


if __name__ == "__main__":
    fire.Fire(main)
