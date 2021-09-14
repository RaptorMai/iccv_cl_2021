import torch
import torch.nn as nn
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale
from avalanche.benchmarks.utils import AvalancheConcatDataset
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.training.plugins import ExperienceBalancedStoragePolicy
from avalanche.training.plugins.strategy_plugin import StrategyPlugin

from sup_con_loss import SupConLoss

"""
A strategy pulgin can change the strategy parameters before and after 
every important fucntion. The structure is shown belown. For instance, self.before_train_forward() and
self.after_train_forward() will be called before and after the main forward loop of the strategy.

The training loop is organized as follows::
        **Training loop**
            train
                train_exp  # for each experience
                    adapt_train_dataset
                    train_dataset_adaptation
                    make_train_dataloader
                    train_epoch  # for each epoch
                        train_iteration # for each minibatch
                            forward
                            backward
                            model update

        **Evaluation loop**
        The evaluation loop is organized as follows::
            eval
                eval_exp  # for each experience
                    adapt_eval_dataset
                    eval_dataset_adaptation
                    make_eval_dataloader
                    eval_epoch  # for each epoch
                        eval_iteration # for each minibatch
                            forward
                            backward
                            model update
"""


class SupContrastPlugin(StrategyPlugin):
    def __init__(self, input_size: tuple = (3, 64, 64)):
        super(SupContrastPlugin).__init__()
        self.transform = nn.Sequential(
            RandomResizedCrop(size=(input_size[1], input_size[2]),
                              scale=(0.2, 1.)),
            RandomHorizontalFlip(),
            ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
            RandomGrayscale(p=0.2)
        )

    def before_training(self, strategy: 'BaseStrategy', **kwargs):
        strategy._criterion = SupConLoss()

    def before_training_exp(self, strategy: 'BaseStrategy', num_workers: int = 0, shuffle: bool = True, **kwargs):
        pass

    def before_train_dataset_adaptation(self, strategy: 'BaseStrategy',
                                        **kwargs):
        pass

    def after_train_dataset_adaptation(self, strategy: 'BaseStrategy',
                                       **kwargs):
        pass

    def before_training_epoch(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def before_training_iteration(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def before_forward(self, strategy: 'BaseStrategy', **kwargs):
        combined_batch_aug = self.transform(strategy.mb_x)
        strategy.mbatch[0] = torch.cat([strategy.mb_x, combined_batch_aug])

    def after_forward(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def before_backward(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def after_backward(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def after_training_iteration(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def before_update(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def after_update(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def after_training_epoch(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def after_training_exp(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def after_training(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def before_eval(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def before_eval_dataset_adaptation(self, strategy: 'BaseStrategy',
                                       **kwargs):
        pass

    def after_eval_dataset_adaptation(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def before_eval_exp(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def after_eval_exp(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def after_eval(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def before_eval_iteration(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def before_eval_forward(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def after_eval_forward(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def after_eval_iteration(self, strategy: 'BaseStrategy', **kwargs):
        pass

class NCMPlugin(StrategyPlugin):
    def __init__(self, mem_size=200):
        super(NCMPlugin).__init__()
        self.ext_mem = {}  # a Dict<task_id, Dataset>
        self.storage_policy = ExperienceBalancedStoragePolicy(
            ext_mem=self.ext_mem,
            mem_size=mem_size,
            adaptive_size=True)

    def before_eval(self, strategy: 'BaseStrategy', **kwargs):
        strategy._criterion = nn.CrossEntropyLoss()

    def before_training_exp(self, strategy: "BaseStrategy",
                            num_workers: int = 0, shuffle: bool = True,
                            **kwargs):
        """
        Dataloader to build batches containing examples from both memories and
        the training dataset
        """
        if len(self.ext_mem) == 0:
            return
        strategy.dataloader = ReplayDataLoader(
            strategy.adapted_dataset,
            AvalancheConcatDataset(self.ext_mem.values()),
            oversample_small_tasks=True,
            num_workers=num_workers,
            batch_size=strategy.train_mb_size,
            shuffle=shuffle)

    def after_training_exp(self, strategy: "BaseStrategy", **kwargs):
        self.storage_policy(strategy, **kwargs)

    def after_eval_iteration(self, strategy: 'BaseStrategy', **kwargs):
        exemplar_means = {}
        cls_exemplar = {cls: [] for cls in strategy.experience.classes_seen_so_far}
        for k in self.ext_mem.keys():
            dataset, indices = self.ext_mem[k].dataset, self.ext_mem[k].indices
            xs, ys, ts = dataset[indices]
            for x, y in zip(xs, ys):
                cls_exemplar[y.item()].append(x)
        for cls, exemplar in cls_exemplar.items():
            features = []
            # Extract feature for each exemplar in p_y
            for ex in exemplar:
                feature = strategy.model.features(ex.unsqueeze(0).to(strategy.device)).detach().clone()
                feature = feature.squeeze()
                feature.data = feature.data / feature.data.norm()  # Normalize
                features.append(feature)
            if len(features) == 0:
                mu_y = torch.normal(0, 1, size=tuple(strategy.model.features(x.unsqueeze(0)).detach().size()))
                mu_y = mu_y.squeeze()
            else:
                features = torch.stack(features)
                mu_y = features.mean(0).squeeze()
            mu_y.data = mu_y.data / mu_y.data.norm()  # Normalize
            exemplar_means[cls] = mu_y

        feature = strategy.model.features(strategy.mb_x)  # (batch_size, feature_size)
        for j in range(feature.size(0)):  # Normalize
            feature.data[j] = feature.data[j] / feature.data[j].norm()
        feature = feature.unsqueeze(2)  # (batch_size, feature_size, 1)
        means = torch.stack([exemplar_means[cls] for cls in strategy.experience.classes_seen_so_far])  # (n_classes, feature_size)

        # old ncm
        means = torch.stack([means] * strategy.mb_x.size(0))  # (batch_size, n_classes, feature_size)
        means = means.transpose(1, 2)
        feature = feature.expand_as(means)  # (batch_size, feature_size, n_classes)
        dists = (feature - means).pow(2).sum(1).squeeze()  # (batch_size, n_classes)
        strategy.mb_output = dists