from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, SCINet, Film, FITS, Real_FITS
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric
from utils.distributed import (DistributedContext, barrier, broadcast_bool,
                               reduce_sum_count, unwrap_model)
from utils.graceful_shutdown import raise_if_requested

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from utils.augmentations import augmentation
import os
import time
import csv

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.distributed_context = getattr(
            args, 'distributed_context', DistributedContext()
        )
        if self.distributed_context.enabled:
            from torch.nn.parallel import DistributedDataParallel

            # Aurora's one-rank-per-tile path moves the model first and wraps it
            # second. The reverse order is only required for multi-CCS sharing.
            self.model = DistributedDataParallel(self.model)
            # Model parameters have now been broadcast from rank zero. Give
            # stochastic layers a reproducible, rank-local random stream.
            torch.manual_seed(self.args.seed + self.args.rank)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'SCINet': SCINet,
            'Film': Film,
            'FITS': FITS,
            'Real_FITS': Real_FITS
        }
        model = model_dict[self.args.model].Model(self.args).float()

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        if self.distributed_context.is_main:
            print('!!!!!!!!!!!!!!learning rate!!!!!!!!!!!!!!!')
            print(self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _get_profile(self, model):
        try:
            from thop import profile
        except ImportError as exc:
            raise RuntimeError(
                "THOP is an optional profiling dependency; install it to use "
                "_get_profile()."
            ) from exc
        _input=torch.randn(self.args.batch_size, self.args.seq_len, self.args.enc_in).to(self.device)
        macs, params = profile(model, inputs=(_input,))
        print('FLOPs: ', macs)
        print('params: ', params)
        return macs, params

    def vali(self, vali_data, vali_loader, criterion):
        total_squared_error = 0.0
        total_elements = 0
        self.model.eval()
        evaluation_model = (
            unwrap_model(self.model)
            if self.distributed_context.enabled
            else self.model
        )
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                raise_if_requested()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)[:,-self.args.pred_len:,:]
                batch_xy = torch.cat([batch_x, batch_y], dim=1)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if 'FITS' in self.args.model:
                    outputs, low = evaluation_model(batch_x)
                elif 'SCINet' in self.args.model:
                    outputs = evaluation_model(batch_x)
                else:
                    if self.args.output_attention:
                        outputs = evaluation_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = evaluation_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                squared_error = torch.sum((outputs - batch_y) ** 2)
                total_squared_error += float(squared_error.item())
                total_elements += batch_y.numel()
                raise_if_requested()

        total_squared_error, total_elements = reduce_sum_count(
            total_squared_error,
            total_elements,
            self.device,
            self.distributed_context,
        )
        if total_elements == 0:
            raise RuntimeError('validation split contains no elements')
        total_loss = total_squared_error / total_elements
        self.model.train()
        return total_loss

    def train(self, setting, ft=False):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        if self.distributed_context.is_main:
            print(self.model)
            print('Trainable parameters: ', sum(p.numel() for p in self.model.parameters() if p.requires_grad))

        path = os.path.join(self.args.checkpoints, setting)
        if self.distributed_context.is_main:
            os.makedirs(path, exist_ok=True)
        barrier(self.distributed_context)

        time_now = time.time()

        train_steps = len(train_loader)
        if train_steps == 0:
            raise RuntimeError(
                'training produced zero batches on rank {}'.format(self.args.rank)
            )
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            raise_if_requested()
            iter_count = 0
            train_squared_error = 0.0
            train_elements = 0

            self.model.train()
            epoch_time = time.time()
            if hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)
            if self.args.in_dataset_augmentation:
                train_loader.dataset.regenerate_augmentation_data()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                raise_if_requested()
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)[:,-self.args.pred_len:,:]
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                # print(batch_x.shape, batch_y.shape)
                batch_xy = torch.cat([batch_x, batch_y], dim=1)

                # if self.args.in_batch_augmentation:
                #     aug = augmentation('batch')
                #     methods = {'f_mask':aug.freq_mask, 'f_mix': aug.freq_mix, 'noise':aug.noise,'noise_input':aug.noise_input}
                #     for step in range(self.args.aug_data_size):
                #         xy = methods[self.args.aug_method](batch_x, batch_y[:, -self.args.pred_len:, :], rate=self.args.aug_rate, dim=1)
                #         batch_x2, batch_y2 = xy[:, :self.args.seq_len, :], xy[:, -self.args.label_len-self.args.pred_len:, :]
                #         if 'noise' not in self.args.aug_method:
                #             batch_x = torch.cat([batch_x,batch_x2],dim=0)
                #             batch_y = torch.cat([batch_y,batch_y2],dim=0)
                #             batch_x_mark = torch.cat([batch_x_mark,batch_x_mark],dim=0)
                #             batch_y_mark = torch.cat([batch_y_mark,batch_y_mark],dim=0)
                #         else:
                #             print('noise')
                #             batch_x = batch_x2
                #             batch_y = batch_y2

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if 'FITS' in self.args.model:
                        outputs, low = self.model(batch_x)
                elif 'SCINet' in self.args.model:
                        outputs = self.model(batch_x)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)

                # print(outputs.shape,batch_y.shape)
                f_dim = -1 if self.args.features == 'MS' else 0
                if ft:
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    # print(outputs.shape,batch_xy.shape)
                    #loss = criterion(outputs, batch_xy)
                    loss = criterion(outputs, batch_y)
                else: 
                    outputs = outputs[:, :, f_dim:]
                    # batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device) #???
                    loss = criterion(outputs, batch_xy)
                train_squared_error += loss.item() * outputs.numel()
                train_elements += outputs.numel()

                if self.distributed_context.is_main and (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()
                # The optimizer step is a safe boundary: all work for this
                # batch has been submitted and the next batch has not begun.
                raise_if_requested()

            train_squared_error, train_elements = reduce_sum_count(
                train_squared_error,
                train_elements,
                self.device,
                self.distributed_context,
            )
            if train_elements == 0:
                raise RuntimeError('training epoch contains no elements')
            train_loss = train_squared_error / train_elements
            vali_loss = self.vali(vali_data, vali_loader, criterion)

            stop = False
            if self.distributed_context.is_main:
                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
                print("Epoch: {0}, Steps/rank: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss))
                early_stopping(
                    vali_loss, unwrap_model(self.model), path
                )
                stop = early_stopping.early_stop
            stop = broadcast_bool(
                stop, self.device, self.distributed_context, src=0
            )
            if stop:
                if self.distributed_context.is_main:
                    print("Early stopping")
                break

            adjust_learning_rate(
                model_optim,
                epoch + 1,
                self.args,
                verbose=self.distributed_context.is_main,
            )

        best_model_path = path + '/' + 'checkpoint.pth'
        barrier(self.distributed_context)
        state = torch.load(
            best_model_path, map_location=self.device, weights_only=True
        )
        unwrap_model(self.model).load_state_dict(state)
        barrier(self.distributed_context)

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            checkpoint = os.path.join(
                self.args.checkpoints, setting, 'checkpoint.pth'
            )
            unwrap_model(self.model).load_state_dict(
                torch.load(checkpoint, map_location=self.device, weights_only=True)
            )

        preds = []
        trues = []
        inputx = []
        reconx = []
        inputxy = []
        reconxy = []
        lows = []
        results_root = self.args.results_root
        folder_path = os.path.join(results_root, setting, 'plots')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                raise_if_requested()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)[:,-self.args.pred_len:,:]
                batch_xy = torch.cat([batch_x, batch_y], dim=1).float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                
                if 'FITS' in self.args.model:
                        outputs, low = self.model(batch_x)
                elif 'SCINet' in self.args.model:
                        outputs = self.model(batch_x)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs_ = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs_ = outputs_.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()


                pred = outputs_  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                inputxy.append(batch_xy.detach().cpu().numpy())
                reconx.append(outputs[:, :-self.args.pred_len, f_dim:].detach().cpu().numpy())
                reconxy.append(outputs.detach().cpu().numpy())
                lows.append(low.detach().cpu().numpy())
                raise_if_requested()
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)
        # inputx = np.array(inputx)
        # reconx = np.array(reconx)
        # reconxy = np.array(reconxy)
        # inputxy = np.array(inputxy)
        # lows = np.array(lows)


        # preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        # trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])
        # reconx = reconx.reshape(-1, reconx.shape[-2], reconx.shape[-1])
        # reconxy = reconxy.reshape(-1, reconxy.shape[-2], reconxy.shape[-1])
        # inputxy = inputxy.reshape(-1, inputxy.shape[-2], inputxy.shape[-1])
        # lows = lows.reshape(-1, lows.shape[-2], lows.shape[-1])

        # try: 
        #     for i in range(0,2800,300):
                
        #         # create a figure with 3 subplots
        #         fig, axs = plt.subplots(3, 1, figsize=(10, 10))
        #         # plot pred and true in the first subplot
        #         axs[0].plot(trues[i, :, -1], label='true')
        #         axs[0].plot(preds[i, :, -1], label='pred')
        #         axs[0].set_title('pred and true')
        #         # plot inputx and reconx in the second subplot
        #         axs[1].plot(inputx[i, :, -1], label='inputx')
        #         axs[1].plot(reconx[i, :, -1], label='reconx')
        #         axs[1].set_title('inputx and reconx')
        #         # plot inputxy and reconxy in the third subplot
        #         axs[2].plot(inputxy[i, :, -1], label='inputxy')
        #         axs[2].plot(reconxy[i, :, -1], label='reconxy')
        #         axs[2].plot(lows[i, :, -1])
        #         axs[2].set_title('inputxy and reconxy')
        #         # show the legend
        #         plt.legend()
        #         # save the figure to file
        #         fig.savefig(os.path.join(folder_path, str(i) + '_F.png'))
        #         # print('plottting')
        # except:
        #     pass

        # result save
        folder_path = os.path.join(results_root, setting)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}, corr:{}'.format(mse, mae, rse, corr))
        # One metrics file per setting avoids a shared append target when
        # independent evaluations run concurrently on different XPU tiles.
        metrics_path = os.path.join(folder_path, 'metrics.csv')
        with open(metrics_path, 'w', newline='') as metrics_file:
            writer = csv.DictWriter(
                metrics_file,
                fieldnames=[
                    'model', 'setting', 'data', 'seq_len', 'pred_len', 'seed',
                    'train_mode', 'h_order', 'mse', 'mae', 'rmse', 'mape',
                    'mspe', 'rse', 'corr_mean',
                ],
            )
            writer.writeheader()
            writer.writerow({
                'model': self.args.model,
                'setting': setting,
                'data': self.args.data,
                'seq_len': self.args.seq_len,
                'pred_len': self.args.pred_len,
                'seed': self.args.seed,
                'train_mode': self.args.train_mode,
                'h_order': self.args.H_order,
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'mspe': mspe,
                'rse': rse,
                'corr_mean': float(np.mean(corr)),
            })

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(os.path.join(folder_path, 'pred.npy'), preds)
        np.save(os.path.join(folder_path, 'true.npy'), trues)
        np.save(os.path.join(folder_path, 'x.npy'), inputx)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            unwrap_model(self.model).load_state_dict(
                torch.load(best_model_path, map_location=self.device, weights_only=True)
            )

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                raise_if_requested()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if 'Linear' in self.args.model:
                    outputs = self.model(batch_x)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)
                raise_if_requested()

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
