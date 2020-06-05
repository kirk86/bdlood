import torch
import utils
import numpy as np


def entropy(predictions, targets, filename):
    # predictions are output of log_softmax
    if predictions.min() < 0.0:
        predictions = np.exp(predictions)
    entropy = -np.sum(np.log(predictions + 1e-8) * predictions, axis=1)
    utils.save_nparray(filename, entropy=entropy, predictions=predictions, targets=targets)
    # np.savez(path, entropy=entropy, predictions=predictions, targets=targets)


def calibration_plot(logits, labels, n_bins=20):
    # adapted from https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py
    import torch.nn.functional as F
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    softmaxes = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    accuracies = predictions.eq(labels)

    ece = torch.zeros(1, device=logits.device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = confidences.gt(
            bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(
                avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


def calibration_curve(filename):
    npz_arr = utils.parse_nparray(filename)
    num_bins = 20
    outputs, labels = npz_arr["predictions"], npz_arr["targets"]
    if outputs is None:
        out = None
    else:
        confidences = np.max(outputs, axis=1)
        step = (confidences.shape[0] + num_bins - 1) // num_bins
        bins = np.sort(confidences)[::step]
        if confidences.shape[0] % step != 1:
            bins = np.concatenate((bins, [np.max(confidences)]))
        #bins = np.linspace(0.1, 1.0, 30)
        predictions = np.argmax(outputs, 1)
        bin_lowers = bins[:-1]
        bin_uppers = bins[1:]

        accuracies = predictions == labels

        xs = []
        ys = []
        zs = []

        #ece = Variable(torch.zeros(1)).type_as(confidences)
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = (confidences > bin_lower) * (confidences < bin_upper)
            prop_in_bin = in_bin.mean()
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin -
                              accuracy_in_bin) * prop_in_bin
                xs.append(avg_confidence_in_bin)
                ys.append(accuracy_in_bin)
                zs.append(prop_in_bin)
        xs = np.array(xs)
        ys = np.array(ys)
        zs = np.array(zs)

        out = {
            'confidence': xs,
            'accuracy': ys,
            'prop_in_bin': zs,
            'ece': ece,
        }

    # save_nparray(filename + '_calibration', **out)
    return out
