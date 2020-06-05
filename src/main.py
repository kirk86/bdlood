# coding: utf-8

import fire
import utils
import torch
import datetime
import pandas as pd
from models import algorithms
from config import Config
from loader import Datasets
from trainer import Trainer


def main(epochs=150, batch_size=512, num_workers=1, pin_memory=True,
         log_interval=1, step=50, gamma=0.1, dist=False, **kwargs):

    # create dirs and setup logger
    config = Config()
    logger = config.setup_logger()

    # reproducible
    utils.set_seed(2019)

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # metrics
    df = pd.DataFrame(columns=['epoch', 'lr', 'train_loss',
                               'train_acc', 'valid_loss', 'valid_acc'])

    algorithm_name = kwargs['algorithm']
    optimizer_name = kwargs['optimizer']
    dataset_name = kwargs['dataset']
    num_classes = kwargs['num_classes'] if 'num_classes' in kwargs.keys() else 10

    # setup model
    model = getattr(algorithms, algorithm_name)

    # setup dataset
    if dist and torch.distributed.is_available():
        torch.distributed.init_process_group(backend='nccl', init_method='env://')  # init torch dist.
        sampler = torch.utils.data.distributed.DistributedSampler
        logger.info("Preparing distributed loader...")
    else:
        sampler = torch.utils.data.SubsetRandomSampler
        logger.info("Preparing loader...")

    db = Datasets(dataset_name, config, pin_memory=pin_memory, sampler=sampler)
    train_loader, valid_loader, test_loader = db.split_image_data(
        train_data=db.train, test_data=db.test, dist=dist)
    config.data = {'train': train_loader,
                   'valid': valid_loader,
                   'test': test_loader}

    logger.info("train ==> {}, valid ==> {}, test ==> {}".format(
        len(train_loader.sampler),
        len(valid_loader.sampler),
        len(test_loader.sampler))
    )
    model = model(num_classes=num_classes)
    filename = f"{str(model)}_{optimizer_name}_{dataset_name}"
    logger.info("Preparing model ==> {}\n".format(str(model)))

    # setup optimizer
    optimizer = getattr(torch.optim, optimizer_name)(
        model.parameters(), lr=kwargs['lr'])
    optim_params = {k: v for k, v in kwargs.items()
                    if k in optimizer.param_groups[0]}

    logger.debug("======= Experiment Start: {} =======".format(
        datetime.datetime.now()))
    if torch.distributed.is_initialized():
        # single-machine multi-gpu
        # device ids will include all gpus by default
        model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model)
        logger.info("Distributed training...")
    elif torch.cuda.device_count() > 1:
        model = torch.nn.parallel.DataParallel(model)
        model.to(device)
        logger.info("Parallel training...")
    else:
        model.to(device)

    # assign params to optimizer
    optimizer = getattr(torch.optim, optimizer_name)(
        model.parameters(), **optim_params)

    logger.debug("Model: {}, Dataset: {}, Optimizer: {}, params: {}".format(
        algorithm_name, dataset_name, optimizer_name, optim_params))

    utils.model_summary(model)

    # setup learning rate scheduler
    # scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer, step_size=step, gamma=gamma
    # )
    # lr_scheduler = GradualWarmupScheduler(
    # optimizer, multiplier=1.1, total_epoch=20)
    #########
    # train #
    #########
    # first_batch = next(iter(train_loader))
    # first_valid = next(iter(valid_loader))
    # xtr, ytr = first_batch.__iter__()
    # xval, yval = first_valid.__iter__()
    # show(tv.utils.make_grid(xtr))
    # show(tv.utils.make_grid(xval))
    logger.info(', '.join(df.columns.to_list()))
    trainer = Trainer(config, model, optimizer, device)
    idx = 0
    for epoch in range(1, epochs + 1):
        if torch.distributed.is_initialized():
            train_loader.sampler.set_epoch(epoch)

        newlr = utils.schedule(epoch, lr_init=kwargs['lr'])
        for group in optimizer.param_groups:
            group['lr'] = newlr

        # train_loss, train_acc = utils.train(model, optimizer, train_loader, device)
        train_loss, train_acc = trainer.train(epoch)
        if epoch % log_interval == 0:
            # valid_loss, valid_acc = utils.test(model, valid_loader, device)
            valid_loss, valid_acc = trainer.evaluate()

            # update metrics
            lr = optimizer.param_groups[0]['lr']
            df = df.append({
                'epoch': epoch,
                'lr': optimizer.param_groups[0]['lr'],
                'train_loss': train_loss.average,
                'train_acc': train_acc.accuracy,
                'valid_loss': valid_loss.average,
                'valid_acc': valid_acc.accuracy}, ignore_index=True)

            # save model
            if valid_loss.average <= df['valid_loss'].min():
                model_state_dict = model.module.state_dict() \
                    if torch.distributed.is_initialized() or isinstance(
                            model, torch.nn.parallel.DataParallel
                    ) else model.state_dict()
                save_params = {
                    'epoch': epoch,
                    'valid_loss': valid_loss,
                    'valid_acc': valid_acc,
                    'lr': optimizer.param_groups[0]['lr'],
                    'optimizer_state_dict': optimizer.state_dict(),
                    'model_state_dict': model_state_dict
                }
                if torch.distributed.is_initialized() and \
                   torch.distributed.get_rank() == 0:
                    utils.save_model(filename, config, **save_params)
                else:
                    utils.save_model(filename, config, **save_params)

            # logger.info("{}".format(', '.join(
            #     df.iloc[idx].values.round(4).astype(str).tolist()))
            # )
            logger.info("{}".format(', '.join(map(
                str, [epoch, lr, train_loss, train_acc, valid_loss, valid_acc]))))
            idx += 1
        # scheduler.step(valid_loss)
        # lr_scheduler.step(epoch)
    # save metrics to csv
    df.to_csv(config.out_path + filename + '.csv', index=False)

    ########
    # test #
    ########
    chkpt = utils.load_chkpt(filename, config.chkpt_path)
    model, optimizer = utils.prepare_model(algorithm_name, optimizer_name,
                                           chkpt, optim_params, device)
    test_loss, test_acc = utils.test(model, test_loader, device)
    logger.info("Evaluate test set: test_loss: {}, test_acc: {}\n".format(
        test_loss, test_acc))
    logger.debug("======= Experiment End: {} =======\n\n".format(
        datetime.datetime.now()))


if __name__ == "__main__":
    # logger = utils.setup_logger()
    fire.Fire(main)
