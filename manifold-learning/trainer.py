import torch
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from exceptions import NanException, EarlyStoppingException
from torch.nn.utils import clip_grad_norm_
from torch import nn


class SubsetRandomSampler:
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class ForwardTrainer:
    """Trainer for likelihood-based flow training when the model is not conditional."""

    def train(
        self,
        dataset,
        loss_functions,
        loss_weights=None,
        loss_labels=None,
        epochs=50,
        batch_size=100,
        optimizer=optim.AdamW,
        optimizer_kwargs=None,
        initial_lr=1.0e-3,
        scheduler=optim.lr_scheduler.CosineAnnealingLR,
        scheduler_kwargs=None,
        restart_scheduler=None,
        validation_split=0.25,
        early_stopping=True,
        early_stopping_patience=None,
        clip_gradient=1.0,
        verbose="all",
        parameters=None,
        callbacks=None,
        forward_kwargs=None,
        custom_kwargs=None,
        compute_loss_variance=False,
        seed=None,
        initial_epoch=None,
    ):
        if initial_epoch is not None and initial_epoch >= epochs:
            logging.info(
                "Initial epoch is larger than epochs, nothing to do in this training phase!"
            )
        elif initial_epoch is not None and initial_epoch <= 0:
            initial_epoch = None

        if loss_labels is None:
            loss_labels = [fn.__name__ for fn in loss_functions]

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        print("Initialising training data")
        train_loader, val_loader = self.make_dataloader(
            dataset, validation_split, batch_size
        )

        print("Setting up optimizer")
        optimizer_kwargs = {} if optimizer_kwargs is None else optimizer_kwargs
        if parameters is None:
            parameters = list(self.model.parameters())
        opt = optimizer(parameters, lr=initial_lr, **optimizer_kwargs)

        print("Setting up LR scheduler")
        if epochs < 2:
            scheduler = None
            print("Deactivating scheduler for only %s epoch", epochs)
        scheduler_kwargs = {} if scheduler_kwargs is None else scheduler_kwargs
        sched = None
        epochs_per_scheduler = (
            restart_scheduler if restart_scheduler is not None else epochs
        )
        if scheduler is not None:
            try:
                sched = scheduler(
                    optimizer=opt, T_max=epochs_per_scheduler, **scheduler_kwargs
                )
            except:
                sched = scheduler(optimizer=opt, **scheduler_kwargs)

        early_stopping = (
            early_stopping and (validation_split is not None) and (epochs > 1)
        )
        best_loss, best_model, best_epoch = None, None, None
        if early_stopping and early_stopping_patience is None:
            print("Using early stopping with infinite patience")
        elif early_stopping:
            print(
                "Using early stopping with patience %s", early_stopping_patience
            )
        else:
            print("No early stopping")

        n_losses = len(loss_labels)
        loss_weights = [1.0] * n_losses if loss_weights is None else loss_weights

        n_epochs_verbose = self._set_verbosity(epochs, verbose)

        print("Beginning main training loop")
        losses_train, losses_val = [], []

        # Resuming training
        if initial_epoch is None:
            initial_epoch = 0
        else:
            print("Resuming with epoch %s", initial_epoch + 1)
            for _ in range(initial_epoch):
                sched.step()  # Hacky, but last_epoch doesn't work when not saving the optimizer state

        # Initial callbacks
        if callbacks is not None:
            for callback in callbacks:
                callback(-1, self.model, 0.0, 0.0, last_batch=self.last_batch)

        # Loop over epochs
        for i_epoch in range(initial_epoch, epochs):
            print("Training epoch %s / %s", i_epoch + 1, epochs)

            # LR schedule
            if sched is not None:
                print("Learning rate: %s", sched.get_last_lr())

            try:
                (
                    loss_train,
                    loss_val,
                    loss_contributions_train,
                    loss_contributions_val,
                ) = self.epoch(
                    i_epoch,
                    train_loader,
                    val_loader,
                    opt,
                    loss_functions,
                    loss_weights,
                    clip_gradient,
                    parameters,
                    forward_kwargs=forward_kwargs,
                    custom_kwargs=custom_kwargs,
                    compute_loss_variance=compute_loss_variance,
                )
                losses_train.append(loss_train)
                losses_val.append(loss_val)
            except NanException:
                print(
                    "Ending training during epoch %s because NaNs appeared", i_epoch + 1
                )
                raise

            if early_stopping:
                try:
                    best_loss, best_model, best_epoch = self.check_early_stopping(
                        best_loss,
                        best_model,
                        best_epoch,
                        loss_val,
                        i_epoch,
                        early_stopping_patience,
                    )
                except EarlyStoppingException:
                    print(
                        "Early stopping: ending training after %s epochs", i_epoch + 1
                    )
                    break

            verbose_epoch = (i_epoch + 1) % n_epochs_verbose == 0
            self.report_epoch(
                i_epoch,
                loss_labels,
                loss_train,
                loss_val,
                loss_contributions_train,
                loss_contributions_val,
                verbose=verbose_epoch,
            )

            # Callbacks
            if callbacks is not None:
                for callback in callbacks:
                    callback(
                        i_epoch,
                        self.model,
                        loss_train,
                        loss_val,
                        last_batch=self.last_batch,
                    )

            # LR scheduler
            if sched is not None:
                sched.step()
                if (
                    restart_scheduler is not None
                    and (i_epoch + 1) % restart_scheduler == 0
                ):
                    try:
                        sched = scheduler(
                            optimizer=opt,
                            T_max=epochs_per_scheduler,
                            **scheduler_kwargs
                        )
                    except:
                        sched = scheduler(optimizer=opt, **scheduler_kwargs)

        if early_stopping and len(losses_val) > 0:
            self.wrap_up_early_stopping(
                best_model, losses_val[-1], best_loss, best_epoch
            )

        print("Training finished")

        return np.array(losses_train), np.array(losses_val)

    def epoch(
        self,
        i_epoch,
        train_loader,
        val_loader,
        optimizer,
        loss_functions,
        loss_weights,
        clip_gradient,
        parameters,
        forward_kwargs=None,
        custom_kwargs=None,
        compute_loss_variance=False,
    ):
        n_losses = len(loss_weights)

        self.model.train()
        loss_contributions_train = np.zeros(n_losses)
        loss_train = [] if compute_loss_variance else 0.0

        for i_batch, batch_data in enumerate(train_loader):
            if i_batch == 0 and i_epoch == 0:
                self.first_batch(batch_data)
            batch_loss, batch_loss_contributions = self.batch_train(
                batch_data,
                loss_functions,
                loss_weights,
                optimizer,
                clip_gradient,
                parameters,
                forward_kwargs=forward_kwargs,
                custom_kwargs=custom_kwargs,
            )
            if compute_loss_variance:
                loss_train.append(batch_loss)
            else:
                loss_train += batch_loss
            for i, batch_loss_contribution in enumerate(
                batch_loss_contributions[:n_losses]
            ):
                loss_contributions_train[i] += batch_loss_contribution

            self.report_batch(i_epoch, i_batch, True, batch_data, batch_loss)

        loss_contributions_train /= len(train_loader)
        if compute_loss_variance:
            loss_train = np.array([np.mean(loss_train), np.std(loss_train)])
        else:
            loss_train /= len(train_loader)

        if val_loader is not None:
            self.model.eval()
            loss_contributions_val = np.zeros(n_losses)
            loss_val = [] if compute_loss_variance else 0.0

            for i_batch, batch_data in enumerate(val_loader):
                batch_loss, batch_loss_contributions = self.batch_val(
                    batch_data,
                    loss_functions,
                    loss_weights,
                    forward_kwargs=forward_kwargs,
                    custom_kwargs=custom_kwargs,
                )
                if compute_loss_variance:
                    loss_val.append(batch_loss)
                else:
                    loss_val += batch_loss
                for i, batch_loss_contribution in enumerate(
                    batch_loss_contributions[:n_losses]
                ):
                    loss_contributions_val[i] += batch_loss_contribution

                self.report_batch(i_epoch, i_batch, False, batch_data, batch_loss)

            loss_contributions_val /= len(val_loader)
            if compute_loss_variance:
                loss_val = np.array([np.mean(loss_val), np.std(loss_val)])
            else:
                loss_val /= len(val_loader)

        else:
            loss_contributions_val = None
            loss_val = None

        return loss_train, loss_val, loss_contributions_train, loss_contributions_val

    def partial_epoch(
        self,
        i_epoch,
        train_loader,
        val_loader,
        optimizer,
        loss_functions,
        loss_weights,
        parameters,
        clip_gradient=None,
        i_batch_start_train=0,
        i_batch_start_val=0,
        forward_kwargs=None,
        custom_kwargs=None,
        compute_loss_variance=False,
    ):
        if compute_loss_variance:
            raise NotImplementedError

        n_losses = len(loss_weights)
        assert (
            len(loss_functions) == n_losses
        ), "{} loss functions, but {} weights".format(len(loss_functions), n_losses)

        self.model.train()
        loss_contributions_train = np.zeros(n_losses)
        loss_train = [] if compute_loss_variance else 0.0

        i_batch = i_batch_start_train

        for batch_data in train_loader:
            if i_batch == 0 and i_epoch == 0:
                self.first_batch(batch_data)
            batch_loss, batch_loss_contributions = self.batch_train(
                batch_data,
                loss_functions,
                loss_weights,
                optimizer,
                clip_gradient,
                parameters,
                forward_kwargs=forward_kwargs,
                custom_kwargs=custom_kwargs,
            )
            if compute_loss_variance:
                loss_train.append(batch_loss)
            else:
                loss_train += batch_loss
            for i, batch_loss_contribution in enumerate(
                batch_loss_contributions[:n_losses]
            ):
                loss_contributions_train[i] += batch_loss_contribution

            self.report_batch(i_epoch, i_batch, True, batch_data, batch_loss)

        i_batch += 1

        loss_contributions_train /= len(train_loader)
        if compute_loss_variance:
            loss_train = np.array([np.mean(loss_train), np.std(loss_train)])
        else:
            loss_train /= len(train_loader)

        if val_loader is not None:
            self.model.eval()
            loss_contributions_val = np.zeros(n_losses)
            loss_val = [] if compute_loss_variance else 0.0

            i_batch = i_batch_start_val

            for batch_data in val_loader:
                batch_loss, batch_loss_contributions = self.batch_val(
                    batch_data,
                    loss_functions,
                    loss_weights,
                    forward_kwargs=forward_kwargs,
                    custom_kwargs=custom_kwargs,
                )
                if compute_loss_variance:
                    loss_val.append(batch_loss)
                else:
                    loss_val += batch_loss
                for i, batch_loss_contribution in enumerate(
                    batch_loss_contributions[:n_losses]
                ):
                    loss_contributions_val[i] += batch_loss_contribution

                self.report_batch(i_epoch, i_batch, False, batch_data, batch_loss)

            i_batch += 1

            loss_contributions_val /= len(val_loader)
            if compute_loss_variance:
                loss_val = np.array([np.mean(loss_val), np.std(loss_val)])
            else:
                loss_val /= len(val_loader)

        else:
            loss_contributions_val = None
            loss_val = None

        return loss_train, loss_val, loss_contributions_train, loss_contributions_val

    def first_batch(self, batch_data):
        pass

    def batch_train(
        self,
        batch_data,
        loss_functions,
        loss_weights,
        optimizer,
        clip_gradient,
        parameters,
        forward_kwargs=None,
        custom_kwargs=None,
    ):
        loss_contributions = self.forward_pass(
            batch_data,
            loss_functions,
            forward_kwargs=forward_kwargs,
            custom_kwargs=custom_kwargs,
        )
        loss = self.sum_losses(loss_contributions, loss_weights)

        self.optimizer_step(optimizer, loss, clip_gradient, parameters)

        loss = loss.item()
        loss_contributions = [contrib.item() for contrib in loss_contributions]
        return loss, loss_contributions

    def batch_val(
        self,
        batch_data,
        loss_functions,
        loss_weights,
        forward_kwargs=None,
        custom_kwargs=None,
    ):
        loss_contributions = self.forward_pass(
            batch_data,
            loss_functions,
            forward_kwargs=forward_kwargs,
            custom_kwargs=custom_kwargs,
        )
        loss = self.sum_losses(loss_contributions, loss_weights)

        loss = loss.item()
        loss_contributions = [contrib.item() for contrib in loss_contributions]
        return loss, loss_contributions

    def forward_pass(
        self, batch_data, loss_functions, forward_kwargs=None, custom_kwargs=None
    ):
        """
        Forward pass of the model. Needs to be implemented by any subclass.

        Parameters
        ----------
        batch_data : OrderedDict with str keys and Tensor values
            The data of the minibatch.

        loss_functions : list of function
            Loss functions.

        Returns
        -------
        losses : list of Tensor
            Losses as scalar pyTorch tensors.

        """
        raise NotImplementedError

    def report_batch(self, i_epoch, i_batch, train, batch_data, batch_loss):
        pass

    def __init__(self, model, run_on_gpu=True, multi_gpu=True, double_precision=False):
        self.model = model

        self.run_on_gpu = run_on_gpu and torch.cuda.is_available()
        self.multi_gpu = self.run_on_gpu and multi_gpu and torch.cuda.device_count() > 1

        self.device = torch.device("cuda" if self.run_on_gpu else "cpu")
        self.dtype = torch.double if double_precision else torch.float
        if self.run_on_gpu and double_precision:
            torch.set_default_tensor_type("torch.cuda.DoubleTensor")
        elif self.run_on_gpu:
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        elif double_precision:
            torch.set_default_tensor_type("torch.DoubleTensor")
        else:
            torch.set_default_tensor_type("torch.FloatTensor")

        self.model = self.model.to(self.device, self.dtype)
        self.last_batch = None

        print(
            "Training on %s with %s precision",
            "{} GPUS".format(torch.cuda.device_count())
            if self.multi_gpu
            else "GPU"
            if self.run_on_gpu
            else "CPU",
            "double" if double_precision else "single",
        )

    def check_early_stopping(
        self,
        best_loss,
        best_model,
        best_epoch,
        loss,
        i_epoch,
        early_stopping_patience=None,
    ):
        try:
            loss_ = loss[0]
        except:
            loss_ = loss

        if best_loss is None or loss_ < best_loss:
            best_loss = loss_
            best_model = self.model.state_dict()
            best_epoch = i_epoch

        if (
            early_stopping_patience is not None
            and i_epoch - best_epoch > early_stopping_patience >= 0
        ):
            raise EarlyStoppingException

        return best_loss, best_model, best_epoch

    def wrap_up_early_stopping(self, best_model, currrent_loss, best_loss, best_epoch):
        try:
            loss_ = currrent_loss[0]
        except:
            loss_ = currrent_loss

        if loss_ is None or best_loss is None:
            print("Loss is None, cannot wrap up early stopping")
        elif best_loss < loss_:
            print(
                "Early stopping after epoch %s, with loss %8.5f compared to final loss %8.5f",
                best_epoch + 1,
                best_loss,
                loss_,
            )
            self.model.load_state_dict(best_model)
        else:
            print("Early stopping did not improve performance")

    @staticmethod
    def report_epoch(
        i_epoch,
        loss_labels,
        loss_train,
        loss_val,
        loss_contributions_train,
        loss_contributions_val,
        verbose=False,
    ):
        logging_fn = print if verbose else print

        def contribution_summary(labels, contributions):
            summary = ""
            for i, (label, value) in enumerate(zip(labels, contributions)):
                if i > 0:
                    summary += ", "
                summary += "{}: {:>6.3f}".format(label, value)
            return summary

        try:
            train_report = "Epoch {:>3d}: train loss {:>8.5f} +/- {:>8.5f} ({})".format(
                i_epoch + 1,
                loss_train[0],
                loss_train[1],
                contribution_summary(loss_labels, loss_contributions_train),
            )
        except:
            train_report = "Epoch {:>3d}: train loss {:>8.5f} ({})".format(
                i_epoch + 1,
                loss_train,
                contribution_summary(loss_labels, loss_contributions_train),
            )
        logging_fn(train_report)

        if loss_val is not None:
            try:
                val_report = "           val. loss  {:>8.5f} +/- {:>8.5f} ({})".format(
                    loss_val[0],
                    loss_val[1],
                    contribution_summary(loss_labels, loss_contributions_val),
                )
            except:
                val_report = "           val. loss  {:>8.5f} ({})".format(
                    loss_val, contribution_summary(loss_labels, loss_contributions_val)
                )
            logging_fn(val_report)

    @staticmethod
    def _check_for_nans(label, *tensors, fix_until=None, replace=0.0):
        for tensor in tensors:
            if tensor is None:
                continue

            if torch.isnan(tensor).any():
                n_nans = torch.sum(torch.isnan(tensor)).item()
                if fix_until is not None:
                    if n_nans <= fix_until:
                        print(
                            "%s contains %s NaNs, setting them to zero", label, n_nans
                        )
                        tensor[torch.isnan(tensor)] = replace
                        return

                print("%s contains %s NaNs, aborting training!", label, n_nans)
                raise NanException

    @staticmethod
    def make_dataloader(dataset, validation_split, batch_size, n_workers=4):
        print("Setting up dataloaders with %s workers", n_workers)

        if validation_split is None or validation_split <= 0.0:
            train_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                # pin_memory=self.run_on_gpu,
                num_workers=n_workers,
            )
            val_loader = None

        else:
            assert 0.0 < validation_split < 1.0, "Wrong validation split: {}".format(
                validation_split
            )

            n_samples = len(dataset)
            indices = list(range(n_samples))
            split = int(np.floor(validation_split * n_samples))
            np.random.shuffle(indices)
            train_idx, valid_idx = indices[split:], indices[:split]

            print("Training partition indices: %s...", train_idx[:10])
            print("Validation partition indices: %s...", valid_idx[:10])

            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(valid_idx)

            train_loader = DataLoader(
                dataset,
                sampler=train_sampler,
                batch_size=batch_size,
                # pin_memory=self.run_on_gpu,
                num_workers=n_workers,
            )
            val_loader = DataLoader(
                dataset,
                sampler=val_sampler,
                batch_size=batch_size,
                # pin_memory=self.run_on_gpu,
                num_workers=n_workers,
            )

        return train_loader, val_loader

    @staticmethod
    def sum_losses(contributions, weights):
        loss = weights[0] * contributions[0]
        for _w, _l in zip(weights[1:], contributions[1:]):
            loss = loss + _w * _l
        return loss

    def optimizer_step(self, optimizer, loss, clip_gradient, parameters):
        optimizer.zero_grad()
        loss.backward()
        if clip_gradient is not None:
            clip_grad_norm_(parameters, clip_gradient)
            # grad_norm = clip_grad_norm_(parameters, clip_gradient)
            # print("  Gradient norm (clipping at %s): %s", clip_gradient, grad_norm)
        optimizer.step()

    @staticmethod
    def _set_verbosity(epochs, verbose):
        # Verbosity
        if verbose == "all":  # Print output after every epoch
            n_epochs_verbose = 1
        elif verbose == "many":  # Print output after 2%, 4%, ..., 100% progress
            n_epochs_verbose = max(int(round(epochs / 50, 0)), 1)
        elif verbose == "some":  # Print output after 10%, 20%, ..., 100% progress
            n_epochs_verbose = max(int(round(epochs / 20, 0)), 1)
        elif verbose == "few":  # Print output after 20%, 40%, ..., 100% progress
            n_epochs_verbose = max(int(round(epochs / 5, 0)), 1)
        elif verbose == "none":  # Never print output
            n_epochs_verbose = epochs + 2
        else:
            raise ValueError("Unknown value %s for keyword verbose", verbose)
        return n_epochs_verbose

    def first_batch(self, batch_data):
        if self.multi_gpu:
            x, y = batch_data
            if len(x.size()) < 2:
                x = x.view(x.size(0), -1)
            x = x.to(self.device, self.dtype)
            self.model(x[: x.shape[0] // torch.cuda.device_count(), ...])

    def forward_pass(
        self, batch_data, loss_functions, forward_kwargs=None, custom_kwargs=None
    ):
        if forward_kwargs is None:
            forward_kwargs = {}

        x = batch_data[0]
        self._check_for_nans("Training data", x)

        if len(x.size()) < 2:
            x = x.view(x.size(0), -1)
        x = x.to(self.device, self.dtype)

        if self.multi_gpu:
            results = nn.parallel.data_parallel(
                self.model, x, module_kwargs=forward_kwargs
            )
        else:
            results = self.model(x, **forward_kwargs)
        if len(results) == 4:
            x_reco, log_prob, u, hidden = results
        else:
            x_reco, log_prob, u = results
            hidden = None

        self._check_for_nans("Reconstructed data", x_reco, fix_until=5)
        if log_prob is not None:
            self._check_for_nans("Log likelihood", log_prob, fix_until=5)
        if x.size(0) >= 15:
            self.last_batch = {
                "x": x.detach().cpu().numpy(),
                "x_reco": x_reco.detach().cpu().numpy(),
                "log_prob": None
                if log_prob is None
                else log_prob.detach().cpu().numpy(),
                "u": u.detach().cpu().numpy(),
            }

        losses = [
            loss_fn(x_reco, x, log_prob, hidden=hidden) for loss_fn in loss_functions
        ]
        self._check_for_nans("Loss", *losses)

        return losses
