"""PKR learner.

Built on the DER dynamic-expansion backbone (one feature extractor per task,
older extractors frozen, auxiliary classifier, weight alignment). Each task is
trained in two stages:

  Stage 1  DER-style training on new data + exemplars:
           L = CE(logits) + CE(aux_logits)
           then build the rehearsal memory and take a snapshot of the network.

  Stage 2  Re-train the same task, distilling from the stage-1 snapshot with
           the HCL loss. For task t > 0 the logits of the previous task's
           classes in the teacher are replaced by those of the stage-1
           snapshot of task t-1:
           task 0 : L = CE(logits) + HCL
           task t : L = 0.5 * (CE(logits) + CE(aux_logits)) + 0.5 * HCL

Weight alignment is applied after each stage when t > 0.
"""
import copy
import logging

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.base import BaseLearner
from utils.inc_net import DERNet
from utils.toolkit import count_parameters, tensor2numpy

# Training schedule (unchanged from the original implementation)
init_epoch = 200
init_lr = 0.1
init_milestones = [60, 120, 170]
init_lr_decay = 0.1
init_weight_decay = 0.0005

epochs = 170
lrate = 0.1
milestones = [80, 120, 150]
lrate_decay = 0.1
batch_size = 128
weight_decay = 2e-4
num_workers = 0


class PKR(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = DERNet(args, False)
        self._network_module_ptr = None
        # Stage-1 snapshot of the previous task (teacher for the previous-task logits).
        self._prev_stage1_network = None

    def after_task(self):
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        # Freeze the feature extractors of previous tasks.
        for i in range(self._cur_task):
            for p in self._network.convnets[i].parameters():
                p.requires_grad = False

        logging.info("All params: {}".format(count_parameters(self._network)))
        logging.info("Trainable params: {}".format(count_parameters(self._network, True)))

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        # Stage 1
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        # Stage 2: distil from the stage-1 snapshot of the current task.
        stage1_network = copy.deepcopy(self._network)
        self._train(self.train_loader, self.test_loader, stage1_network)
        self._prev_stage1_network = stage1_network

    def _set_train_mode(self):
        self._network.train()
        if len(self._multiple_gpus) > 1:
            self._network_module_ptr = self._network.module
        else:
            self._network_module_ptr = self._network
        self._network_module_ptr.convnets[-1].train()
        for i in range(self._cur_task):
            self._network_module_ptr.convnets[i].eval()

    def _train(self, train_loader, test_loader, old_network=None):
        self._network.to(self._device)
        if self._cur_task == 0:
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                momentum=0.9,
                lr=init_lr,
                weight_decay=init_weight_decay,
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=init_milestones, gamma=init_lr_decay
            )
            self._init_train(train_loader, test_loader, optimizer, scheduler, old_network)
        else:
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=lrate,
                momentum=0.9,
                weight_decay=weight_decay,
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=milestones, gamma=lrate_decay
            )
            self._update_representation(train_loader, test_loader, optimizer, scheduler, old_network)
            if len(self._multiple_gpus) > 1:
                self._network.module.weight_align(self._total_classes - self._known_classes)
            else:
                self._network.weight_align(self._total_classes - self._known_classes)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler, old_network):
        prog_bar = tqdm(range(init_epoch))
        for _, epoch in enumerate(prog_bar):
            self._set_train_mode()
            losses, losses_hcl = 0.0, 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]
                loss = F.cross_entropy(logits, targets.long())

                if old_network is not None:
                    with torch.no_grad():
                        old_logits = old_network.get_logits_for_distillation(inputs)
                    hcl_loss = self.hcl(logits, old_logits)
                    loss += hcl_loss
                    losses_hcl += hcl_loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_hcl {:.3f}, Train_accy {:.2f}".format(
                self._cur_task, epoch + 1, init_epoch,
                losses / len(train_loader), losses_hcl / len(train_loader), train_acc,
            )
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info += ", Test_accy {:.2f}".format(test_acc)
            prog_bar.set_description(info)

        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler, old_network):
        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            self._set_train_mode()
            losses, losses_clf, losses_aux, losses_hcl = 0.0, 0.0, 0.0, 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outputs = self._network(inputs)
                logits, aux_logits = outputs["logits"], outputs["aux_logits"]
                loss_clf = F.cross_entropy(logits, targets.long())

                # Auxiliary targets: 0 for old classes, 1..task_size for new classes.
                aux_targets = targets.clone()
                aux_targets = torch.where(
                    aux_targets - self._known_classes + 1 > 0,
                    aux_targets - self._known_classes + 1,
                    0,
                )
                loss_aux = F.cross_entropy(aux_logits, aux_targets.long())
                loss = loss_clf + loss_aux

                if old_network is not None:
                    with torch.no_grad():
                        old_logits = old_network.get_logits_for_distillation(inputs)
                        prev_logits = self._prev_stage1_network.get_logits_for_distillation(inputs)
                        task_size = self._total_classes - self._known_classes
                        previous_task_start = self._known_classes - task_size
                        old_logits[:, previous_task_start:self._known_classes] = prev_logits[:, -task_size:]
                    hcl_loss = self.hcl(logits, old_logits)
                    loss = loss * 0.5 + hcl_loss * 0.5
                    losses_hcl += hcl_loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                losses_aux += loss_aux.item()
                losses_clf += loss_clf.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            info = (
                "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_aux {:.3f}, "
                "Loss_hcl {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task, epoch + 1, epochs,
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_aux / len(train_loader),
                    losses_hcl / len(train_loader),
                    train_acc,
                )
            )
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info += ", Test_accy {:.2f}".format(test_acc)
            prog_bar.set_description(info)

        logging.info(info)

    @staticmethod
    def hcl(student_output, teacher_output):
        """HCL distillation: KL(teacher || student) on sigmoid-normalised logits."""
        teacher_output = teacher_output.detach()
        teacher_output = torch.sigmoid(teacher_output)
        teacher_output = teacher_output / torch.sum(teacher_output, dim=1, keepdim=True)
        teacher_output = torch.clamp(teacher_output, min=1e-4, max=1.0)

        student_output = torch.sigmoid(student_output)
        student_output = student_output / torch.sum(student_output, dim=1, keepdim=True)
        student_output = torch.clamp(student_output, min=1e-4, max=1.0)

        return torch.mean(torch.sum(teacher_output * torch.log(teacher_output / student_output), dim=1))
