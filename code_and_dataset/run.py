import datetime
import os
import sys

import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

from callbacks import EarlyStopCallBack, ModelCheckPointCallBack
from config_parser import Config
from data_provider import DataProvider
from logger import log_to_file, setup_logging
from model import Model, count_parameters, weight_initial
from result_writer import (
    weeekly_result_writer,
    write_binding_core_results,
    write_metrics_file,
    write_metrics_file_IGNORE_IEDBID_AND_LENGTH,
)
from seq_encoding import ENCODING_METHOD_MAP

#############################################################################################
#
# Train
#
#############################################################################################


def batch_train(model, device, data, config):
    hla_a, hla_a_mask, hla_a_length,\
        hla_b, hla_b_mask, hla_b_length,\
        pep, pep_mask, pep_length, ic50 = data

    pred_ic50, _ = model(
        hla_a.to(device), hla_a_mask.to(device), hla_a_length.to(device),
        hla_b.to(device), hla_b_mask.to(device), hla_b_length.to(device),
        pep.to(device), pep_mask.to(device), pep_length.to(device)
    )
    loss = nn.MSELoss()(pred_ic50.to(config.cpu_device), ic50.view(ic50.size(0), 1))

    return loss


def batch_validation(model, device, data, config):
    with torch.no_grad():
        return batch_train(model, device, data, config)


def train(config, data_provider):
    # skip training if test mode
    if not config.do_train:
        log_to_file('Skip train', 'Not enabled training')
        return

    device = config.device
    log_to_file('Device', device)

    # log pytorch version
    log_to_file('PyTorch version', torch.__version__)

    # prepare model
    model = Model(config)
    weight_initial(model, config)
    model.to(device)

    # log param count
    log_to_file('Trainable params count', count_parameters(model))

    # OPTIMIZER
    optimizer = optim.SGD(model.parameters(),
                          lr=config.start_lr, weight_decay=config.weight_decay)
    log_to_file("Optimizer", "SGD")

    # call backs
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, threshold=config.loss_delta, patience=4,
                                               cooldown=4, verbose=True, min_lr=config.min_lr, factor=0.2)
    model_check_callback = ModelCheckPointCallBack(
        model,
        config.model_save_path,
        period=1,
        delta=config.loss_delta,
    )
    early_stop_callback = EarlyStopCallBack(
        patience=10, delta=config.loss_delta)

    # some vars
    epoch_loss = 0
    validation_loss = 0
    steps = data_provider.train_steps()
    log_to_file('Start training', datetime.datetime.now())
    for epoch in range(config.epochs):
        epoch_start_time = datetime.datetime.now()

        # train batches
        model.train(True)
        for _ in range(steps):
            data = data_provider.batch_train()
            loss = batch_train(model, device, data, config)
            loss.backward()

            # clip grads
            nn.utils.clip_grad_value_(model.parameters(), config.grad_clip)

            # update params
            optimizer.step()

            # record loss
            epoch_loss += loss.item()

            # reset grad
            optimizer.zero_grad()

        # time compute
        time_delta = datetime.datetime.now() - epoch_start_time

        # validation on epoch end
        model.eval()
        for _ in range(data_provider.val_steps()):
            data = data_provider.batch_val()
            validation_loss += batch_validation(model,
                                                device, data, config).item()

        # log
        log_to_file("Training process", "[Epoch {0:04d}] - time: {1:4d} s, train_loss: {2:0.5f}, val_loss: {3:0.5f}".format(
            epoch, time_delta.seconds, epoch_loss / steps, validation_loss / data_provider.val_steps()))

        # call back
        model_check_callback.check(
            epoch, validation_loss / data_provider.val_steps())
        if early_stop_callback.check(epoch, validation_loss / data_provider.val_steps()):
            break

        # LR schedule
        scheduler.step(loss.item())

        # reset loss
        epoch_loss = 0
        validation_loss = 0

        # reset data provider
        data_provider.new_epoch()

    # save last epoch model
    torch.save(model.state_dict(), os.path.join(
        config.working_dir, 'last_epoch_model.pytorch'))

#############################################################################################
#
# Test
#
#############################################################################################


def batch_test(model, device, data, config):
    hla_a, hla_a_mask, hla_a_length,\
        hla_b, hla_b_mask, hla_b_length,\
        pep, pep_mask, pep_length, uid_list = data

    pred_ic50, atten_weight = model(
        hla_a.to(device), hla_a_mask.to(device), hla_a_length.to(device),
        hla_b.to(device), hla_b_mask.to(device), hla_b_length.to(device),
        pep.to(device), pep_mask.to(device), pep_length.to(device)
    )
    return pred_ic50, uid_list, atten_weight


def test(config, data_provider):
    """Test on weekly
    """
    # skip testing
    if not config.do_test:
        log_to_file('Skip testing', 'Not enabled testing')
        return
    else:
        log_to_file('Testing start', 'Weekly benchmark')

    device = config.device

    # load and prepare model
    state_dict = torch.load(config.model_save_path)
    model = Model(config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    result_dict = {}
    for _ in range(data_provider.test_steps()):
        data = data_provider.batch_test()
        with torch.no_grad():
            pred_ic50, uid_list, _ = batch_test(model, device, data, config)
            for i, uid in enumerate(uid_list):
                result_dict[uid] = pred_ic50[i].item()

    result_file = weeekly_result_writer(result_dict, config)
    log_to_file('Testing result file', result_file)

    metric_file = write_metrics_file(result_file, config)
    log_to_file('Testing metric result file', metric_file)


    metric_file_ignore_iedb_and_length = write_metrics_file_IGNORE_IEDBID_AND_LENGTH(
        result_file, config)
    log_to_file('Testing metric result file IGNORE IEDB_ID AND length',
                metric_file_ignore_iedb_and_length)

#############################################################################################
#
# Output binding core
#
#############################################################################################


def output_binding_core(config, data_provider):
    config.device = config.cpu_device
    device = config.device

    # load and prepare model
    state_dict = torch.load(config.model_save_path)
    model = Model(config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    attn_weight_dict = {}
    for _ in range(data_provider.bind_core_steps()):
        data = data_provider.batch_bind_core()
        with torch.no_grad():
            _, uid_list, attn_weight_list = batch_test(
                model, device, data, config)
            for i, uid in enumerate(uid_list):
                attn_weight_dict[uid] = [x.item() for x in attn_weight_list[i]]

    bind_core_reuslt_file = write_binding_core_results(
        attn_weight_dict, config)
    log_to_file('Binding core result', bind_core_reuslt_file)

#############################################################################################
#
# LOMO test
#
#############################################################################################


def batch_test_LOMO(model, device, data, config):
    hla_a, hla_a_mask, hla_a_length,\
        hla_b, hla_b_mask, hla_b_length,\
        pep, pep_mask, pep_length, ic50_list, pep_seq_list = data

    pred_ic50, atten_weight = model(
        hla_a.to(device), hla_a_mask.to(device), hla_a_length.to(device),
        hla_b.to(device), hla_b_mask.to(device), hla_b_length.to(device),
        pep.to(device), pep_mask.to(device), pep_length.to(device)
    )
    return pred_ic50, ic50_list, atten_weight, pep_seq_list


def test_LOMO(config, data_provider):
    """Test on on allele
    """
    # skip testing
    if not config.do_test:
        log_to_file('Skip testing', 'Not enabled testing')
        return

    device = config.device

    # load and prepare model
    state_dict = torch.load(config.model_save_path)
    model = Model(config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    result_list = []
    for _ in range(data_provider.LOMO_test_steps()):
        data = data_provider.batch_test()
        with torch.no_grad():
            pred_ic50, real_ic50, _, pep_seq_list = batch_test_LOMO(
                model, device, data, config)
            for i in range(len(pred_ic50)):
                result_list.append(
                    (pep_seq_list[i], real_ic50[i], pred_ic50[i].item()))

    # write to file
    result_file_path = os.path.join(config.working_dir, 'test_result.txt')
    out_file = open(result_file_path, 'w')
    out_file.write('pep\treal\tpred\n')
    for info in result_list:
        out_file.write('{}\t{}\t{}\n'.format(info[0], info[1], info[2]))
    log_to_file('Wrote LOMO test result to file',
                '[{}]'.format(result_file_path))

#############################################################################################
#
# Main
#
#############################################################################################


def main():
    # parse config
    config_file = sys.argv[1]
    config = Config(config_file)

    # setup logger
    setup_logging(config.working_dir)

    # encoding func
    encoding_func = ENCODING_METHOD_MAP[config.encoding_method]
    log_to_file('Encoding method', config.encoding_method)

    data_provider = DataProvider(
        encoding_func,
        config.data_file,
        config.test_file,
        config.bind_core_file,
        config.batch_size,
        max_len_hla_A=config.max_len_hla_A,
        max_len_hla_B=config.max_len_hla_B,
        max_len_pep=config.max_len_pep,
        validation_ratio=config.validation_ratio,
        LOMO=config.is_LOMO,
        LOMO_allele=config.test_allele,
        shuffle_before_epoch_enable=config.shuffle_before_epoch_enable,
    )
    log_to_file('Traning samples', len(data_provider.train_samples))
    log_to_file('Val samples', len(data_provider.validation_samples))
    log_to_file('Traning steps', data_provider.train_steps())
    log_to_file('Val steps', data_provider.val_steps())
    log_to_file('Batch size', data_provider.batch_size)
    log_to_file('max_len_hla_A', data_provider.max_len_hla_A)
    log_to_file('max_len_hla_B', data_provider.max_len_hla_B)
    log_to_file('max_len_pep', data_provider.max_len_pep)
    log_to_file('validation_ratio', data_provider.validation_ratio)
    log_to_file('shuffle_before_epoch_enable',
                data_provider.shuffle_before_epoch_enable)
    if config.is_LOMO:
        log_to_file('LOMO allele:', config.test_allele)
        log_to_file('LOMO allele sample count:', len(
            data_provider.lomo_test_samples))

    if config.is_LOMO:
        train(config, data_provider)
        test_LOMO(config, data_provider)
    else:
        train(config, data_provider)
        test(config, data_provider)
        output_binding_core(config, data_provider)


if __name__ == '__main__':
    main()
