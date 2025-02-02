import os
from itertools import chain
from typing import Any, List, Dict, Literal, Optional, Sequence
from functools import partial

from tqdm.auto import trange, tqdm

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import WeightedRandomSampler, DataLoader

import torchmetrics.functional as metrics
from torchmetrics.functional.classification import matthews_corrcoef
from torchmetrics.functional.clustering import (
    normalized_mutual_info_score,
    adjusted_rand_score
)

import wandb
import mlflow

import loss
from preprocessing import EventTPTokenizer
from nn import MLPProfileEncoder, EventTransformerEncoder, MLPSelector, MLPPredictor
from metric import predictive_scores, purity_score


class EventTPCEncoder(nn.Module):
    def __init__(self, 
                 profile_encoder: MLPProfileEncoder, 
                 event_encoder: EventTransformerEncoder, 
                 is_spherical: bool = True):
        super(EventTPCEncoder, self).__init__()

        self.profile_encoder = profile_encoder
        self.event_encoder = event_encoder

        self.d_model = event_encoder.d_model
        self.is_spherical = is_spherical

        self.projector = nn.Linear(
            in_features=self.d_model,
            out_features=self.d_model
        )

    def forward(self, profile_input, time_input, event_input, causal_mask, padding_mask):
        z_profile = self.profile_encoder(profile_input)  # (N, E)
        z_event = self.event_encoder(time_input, event_input, causal_mask, padding_mask)  # (N, L, E)
        z = self.projector(z_event + z_profile.unsqueeze(1))  # (N, L, E)
        if self.is_spherical:
            z = F.normalize(z, dim=-1)

        return z


class EventTransformerTPC:
    def __init__(self, 
                 tokenizer: EventTPTokenizer, 
                 encoder: EventTPCEncoder, 
                 selector: MLPSelector, 
                 predictor: MLPPredictor):
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.selector = selector
        self.predictor = predictor

        self.C = self.predictor.output_dim
        self.K = self.selector.K
        self.d_model = self.encoder.d_model

        self.cluster_embedding = nn.Embedding(self.K, self.d_model)  # initialized by std. normal

    def predict_z(self, profile_input, time_input, event_input, causal_mask, padding_mask) -> torch.Tensor:
        return self.encoder(profile_input, time_input, event_input, causal_mask, padding_mask)
    
    def predict_pi(self, profile_input, time_input, event_input, causal_mask, padding_mask, output_probs=False) -> torch.Tensor:
        z = self.predict_z(profile_input, time_input, event_input, causal_mask, padding_mask)   # (N, L, E)
        pi = self.selector(z, output_probs=output_probs)  # (n_pred, K)

        return pi
    
    def predict_s(self, profile_input, time_input, event_input, causal_mask, padding_mask, sample=True) -> torch.Tensor:
        pi = self.predict_pi(profile_input, time_input, event_input, causal_mask, padding_mask, output_probs=True)  # (n_pred, K)
        if sample:
            cat = torch.distributions.Categorical(probs=pi)
            s = cat.sample()  # (N, L)
        else:
            s = pi.argmax(dim=-1)

        return s

    def predict_y_bar(self, profile_input, time_input, event_input, causal_mask, padding_mask, sample=True) -> torch.Tensor:
        s = self.predict_s(profile_input, time_input, event_input, causal_mask, padding_mask, sample)  # (n_pred,)
        e = self.cluster_embedding(s)  # (n_pred, E)
        return self.predictor(e)
    
    def predict_y_hat(self, profile_input, time_input, event_input, causal_mask, padding_mask) -> torch.Tensor:
        z = self.predict_z(profile_input, time_input, event_input, causal_mask, padding_mask)
        y_hat = self.predictor(z)

        return y_hat
    
    def predict_clusters(self, *inputs, **kwinputs):
        return self.predict_s(*inputs, **kwinputs, sample=False)
    
    def predict_outcomes(self, *inputs, **kwinputs):
        return self.predict_y_bar(*inputs, **kwinputs, sample=False)
    
    def forward(self, *inputs, **kwinputs):
        return self.predict_s(*inputs, **kwinputs, sample=False)


class Trainer:
    """A trainer for Event-TPC training including stages 1-3.
    """
    def __init__(self, 
                 model: EventTransformerTPC, 
                 train_dataset: Sequence, 
                 eval_dataset: Sequence,
                 predict_task: Literal["multiclass", "multilabel"],
                 gamma: float, alpha: float, beta: float,
                 optim_config: Dict[str, Dict[str, Any]],
                 class_weight_type: Literal["none", "vanilla", "cb"] = "none",
                 predict_threshold: float = 0.5,
                 weighted_sampling: bool = False,
                 class_weight_scale: float = 0,
                 device: str = "auto", 
                 logger: Literal["mlflow", "wandb", "none"] = "mlflow",
                 **kwargs):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.predict_task = predict_task
        self.predict_threshold = predict_threshold
        self.class_weight_type = class_weight_type
        self.class_weight_scale = class_weight_scale
        self.weighted_sampling = weighted_sampling
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        
        # prepare all the necessary optimizers
        self.stage_1_optimizer: torch.optim.Optimizer = optim_config['stage_1'].pop('optim_cls')(
            params=chain(self.model.encoder.parameters(), self.model.predictor.parameters()),
            **optim_config['stage_1']
        )
        self.stage_2_optimizer: torch.optim.Optimizer = optim_config['stage_2'].pop("optim_cls")(
            params=self.model.selector.parameters(),
            **optim_config['stage_2']
        )
        self.actor_optimizer: torch.optim.Optimizer = optim_config['actor'].pop("optim_cls")(
            params=chain(self.model.encoder.parameters(), self.model.selector.parameters()),
            **optim_config['actor']
        )
        self.critic_optimizer: torch.optim.Optimizer = optim_config['critic'].pop("optim_cls")(
            params=self.model.predictor.parameters(),
            **optim_config['critic']
        )
        self.embedding_optimizer = optim_config['embedding'].pop("optim_cls")(
            params=self.model.cluster_embedding.parameters(),
            **optim_config['embedding']
        )
        
        self.device = device
        self.logger = logger

        if self.predict_task == "multiclass":
            self.to_probs = partial(F.softmax, dim=-1)
            self.mle_loss_fn = F.cross_entropy
            self.loss_2_fn = F.cross_entropy
            self.loss_3_fn = loss.cross_entropy
        elif self.predict_task == "multilabel":
            self.to_probs = F.sigmoid
            self.mle_loss_fn = F.binary_cross_entropy_with_logits
            self.loss_1_fn = F.binary_cross_entropy_with_logits
            self.loss_2_fn = F.cross_entropy
            self.loss_3_fn = loss.cross_entropy
        else:
            raise ValueError(f"Invalid predict task: {self.predict_task}")
        
    def log_metrics(self, metrics, step: int):
        if self.logger == "mlflow":
            mlflow.log_metrics(metrics=metrics, step=step)
        elif self.logger == "wandb":
            wandb.log(metrics)
        else:
            pass
    
    def _pred_masking(self, y, masking_rate: float = 0):
        """Randomly mask out some prediction points (in-place) to avoid heavy class imbalance.
        """
        if masking_rate == 0:
            return y
        
        N = y.size(0)
        for i in range(N):
            pred_idx = torch.where(y[i, :] != -100)[0][:-1]
            mask_cnt = int(len(pred_idx) * masking_rate)
            mask_idx = torch.randperm(len(pred_idx))[:mask_cnt]  # randomly choose some points in a rate
            y[i, pred_idx[mask_idx]] = -100

        return y
    
    def _compute_class_weights(self, y: torch.Tensor, scale: float = 1.):
        if self.class_weight_type == "none":
            return None

        C = self.model.C
        y = y.flatten(end_dim=1)  # (N*L,) or (N*L, C)
        
        if self.predict_task == "multiclass":
            class_counts = y[y != -100].bincount(minlength=C)
        else:
            class_counts = y[(y != -100).all(dim=-1)].sum(dim=0)

        if self.class_weight_type == "vanilla":
            assert scale > 0, "Vanilla class weights cannot be used with scale <= 0."

            class_weights = (class_counts.float().mean() / class_counts) * scale
        elif self.class_weight_type == "cb":
            assert scale < 1, "CB class weights cannot be used with scale >= 1"

            class_weights = loss.compute_cb_weights(class_counts, beta=scale)

        class_weights = class_weights.to(self.device)

        return class_weights

    def _collate_fn(self, data):
        inputs, targets = zip(*data)
        profile_inputs, time_inputs, event_inputs = zip(*inputs)

        profile_input = torch.tensor(profile_inputs, dtype=torch.float32)

        pad_id = self.model.tokenizer.pad_token_id
        event_input = pad_sequence(event_inputs, batch_first=True, padding_value=pad_id)
        time_input = pad_sequence(time_inputs, batch_first=True, padding_value=-1)
        target = pad_sequence(targets, batch_first=True, padding_value=-100)

        seq_length = event_input.size(1)
        causal_mask = torch.triu(torch.ones(seq_length, seq_length, dtype=torch.bool), diagonal=1)
        padding_mask =  (event_input == pad_id)

        return (profile_input, time_input, event_input, causal_mask, padding_mask), target

    def get_train_loader(self, batch_size: int = 1, **kwargs) -> DataLoader:
        if self.predict_task == "multiclass" and self.weighted_sampling:
            targets = np.array([target[target != -100][-1] for _, target in self.train_dataset])
            class_cnt = np.bincount(targets)
            class_weights = 1 / class_cnt
            sample_weights = np.array([class_weights[t] for t in targets])

            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(targets))
            
            return DataLoader(self.train_dataset, batch_size=batch_size, collate_fn=self._collate_fn, sampler=sampler, **kwargs)
        
        return DataLoader(self.train_dataset, batch_size=batch_size, collate_fn=self._collate_fn, shuffle=True, **kwargs)
    
    def get_eval_loader(self, batch_size: int = 1, **kwargs) -> DataLoader:
        return DataLoader(self.eval_dataset, batch_size=batch_size, collate_fn=self._collate_fn, **kwargs)

    def train_prediction(self, 
                         epochs: int, 
                         batch_size: int = 1, 
                         eval_epoch: int = 0, 
                         eval_batch_size: int = 1, 
                         pred_mask_rate: float = 0,
                         monitor_metric: Optional[str] = None,
                         greater_is_better: Optional[bool] = None):
        C = self.model.C
        multiclass = self.predict_task == "multiclass"
        pred_token_id = self.model.tokenizer.pred_token_id

        # estimate the class weights
        _, train_Y = self._collate_fn(self.train_dataset)
        if pred_mask_rate > 0 and multiclass:
            train_Y = self._pred_masking(train_Y, pred_mask_rate)
        class_weights = self._compute_class_weights(train_Y, self.class_weight_scale)

        train_loader = self.get_train_loader(batch_size)
        eval_loader = self.get_eval_loader(eval_batch_size)

        if monitor_metric is not None:
            best_ckpt = {"state_dict": None, monitor_metric: None}
        
        lr_schedular = torch.optim.lr_scheduler.CosineAnnealingLR(self.stage_1_optimizer, epochs * len(train_loader))
        for epoch in range(epochs):
            self.model.train()
            running_rec = {"loss": []}
            pbar = tqdm(train_loader, desc=f"Epoch {(epoch+1):03}")
            for data in pbar:
                x, y = data
                profile_input, time_input, event_input, causal_mask, padding_mask = [_x.to(self.device) for _x in x]
                y = y.to(self.device)
                if multiclass:
                    y = self._pred_masking(y, masking_rate=pred_mask_rate)

                self.stage_1_optimizer.zero_grad()
                y_hat = self.model.predict_y_hat(profile_input, time_input, event_input, causal_mask, padding_mask)
                y_hat = y_hat.flatten(end_dim=1)
                y = y.flatten(end_dim=1)
                pred_mask = (event_input == pred_token_id).flatten(end_dim=1)  # (N*L,)
                loss = self.mle_loss_fn(y_hat[pred_mask], y[pred_mask], weight=class_weights)
                loss.backward()
                self.stage_1_optimizer.step()
                lr_schedular.step()

                loss_val = loss.item()
                running_rec["loss"].append(loss_val)

                pbar.set_postfix({"loss": loss_val})

            # log the current learning rate
            running_rec['lr'] = lr_schedular.get_last_lr()[-1]
            
            if eval_epoch and ((epoch+1) % eval_epoch == 0):
                self.model.eval()
                running_rec["eval_loss"] = []
                with torch.no_grad():
                    eval_ys = []
                    eval_y_hats = []
                    for eval_data in eval_loader:
                        eval_x, eval_y = eval_data
                        profile_input, time_input, event_input, causal_mask, padding_mask = [_x.to(self.device) for _x in eval_x]
                        eval_y: torch.Tensor = eval_y.to(self.device)

                        eval_y_hat = self.model.predict_y_hat(profile_input, time_input, event_input, causal_mask, padding_mask)  # (N, L, C)
                        
                        eval_y_hat = eval_y_hat.flatten(end_dim=1)  # (N*L, C)
                        eval_y = eval_y.flatten(end_dim=1)  # (N*L,) or (N*L, C)
                        
                        pred_mask = (event_input == pred_token_id).flatten(end_dim=1)  # (N*L,)
                        eval_ys.append(eval_y[pred_mask])
                        eval_y_hats.append(self.to_probs(eval_y_hat)[pred_mask])
                        
                        eval_loss: torch.Tensor = self.mle_loss_fn(eval_y_hat[pred_mask], eval_y[pred_mask])

                        running_rec["eval_loss"].append(eval_loss.item())
                    
                    eval_y_pred = torch.concat(eval_y_hats, dim=0)
                    eval_y_true = torch.concat(eval_ys, dim=0).to(dtype=torch.int64)

                    print("# Evaluation")
                    print("-------------")

                    # calculate MCC
                    mcc_args = dict(
                        preds=eval_y_pred, 
                        target=eval_y_true,
                        task=self.predict_task,
                        num_classes=C,
                        num_labels=C
                    )
                    if multiclass:
                        mcc = matthews_corrcoef(**mcc_args).numpy(force=True)
                        running_rec.update({"MCC": mcc})
                        print(f"MCC={mcc:.2f}")
                    else:
                        running_rec.update({
                            f"MCC_{t}": matthews_corrcoef(threshold=t, **mcc_args).numpy(force=True) \
                            for t in [0.1, 0.3, 0.5]
                        })

                    # calculate AUROC, AUPRC
                    auc_family_args = dict(
                        task=self.predict_task, 
                        num_classes=C, 
                        num_labels=C
                    )
                    auroc = metrics.auroc(eval_y_pred, eval_y_true, **auc_family_args, average="none").numpy(force=True)
                    auprc = metrics.average_precision(eval_y_pred, eval_y_true, **auc_family_args, average="none").numpy(force=True)
                    print(f"AUROC={np.round(auroc, 2)}\nAUPRC={np.round(auprc, 2)}")
                    running_rec.update({
                        "AUROC": auroc,
                        "AUPRC": auprc
                    })

                    # calculate f1, precision, recall
                    f1_family_args = dict(
                        task=self.predict_task, 
                        threshold=self.predict_threshold,
                        num_classes=C, 
                        num_labels=C
                    )

                    f1   = metrics.f1_score(eval_y_pred, eval_y_true, **f1_family_args, average="none").numpy(force=True)
                    prec = metrics.precision(eval_y_pred, eval_y_true, **f1_family_args, average="none").numpy(force=True)
                    rec  = metrics.recall(eval_y_pred, eval_y_true, **f1_family_args, average="none").numpy(force=True)
                    
                    print(f"F1={np.round(f1, 2)}\nPrec.={np.round(prec, 2)}\nRec.={np.round(rec, 2)}")
                    
                    running_rec.update({
                        "F1-score": metrics.f1_score(eval_y_pred, eval_y_true, **f1_family_args, average="macro").numpy(force=True),
                        "Precision": metrics.precision(eval_y_pred, eval_y_true, **f1_family_args, average="macro").numpy(force=True),
                        "Recall": metrics.recall(eval_y_pred, eval_y_true, **f1_family_args, average="macro").numpy(force=True)
                    })

                    print("-------------")
                
                # update records
                for k in running_rec:
                    running_rec[k] = np.mean(running_rec[k])

                self.log_metrics(running_rec, epoch)

                # check save best
                if monitor_metric is not None:
                    assert greater_is_better is not None, "Need to set `greater_is_better`"

                    compare_op = np.greater if greater_is_better else np.less
                    if best_ckpt['state_dict'] is None or compare_op(running_rec[monitor_metric], best_ckpt[monitor_metric]):
                        best_ckpt['state_dict'] = self.model.state_dict()
                        best_ckpt[monitor_metric] = running_rec[monitor_metric]
                
                print()  # print a new line to separate the next epoch

        if monitor_metric is not None:
            self.model.load_state_dict(best_ckpt["state_dict"])

    def train_selector(self, latent_state: torch.Tensor, assignments: torch.Tensor, epochs: int, batch_size: int, eval_batch_size: int):
        """Train the selector on the given latent state z and the cluster assignments. 
        Note that for stage 2 training, it's recommended to directly use `train_clustering()` instead.

        Args:
            latent_state (torch.Tensor): the latent state output from the encoder.
            assignments (torch.Tensor): the cluster assignments, generally taken from the KMeans prediction on `y_hat`.
            epochs (int): the total training iteration.
            batch_size (int): the size of a batch.
        """
        Z, S = latent_state, assignments
        train_loader = DataLoader(dataset=list(zip(Z, S)), batch_size=batch_size, shuffle=True)
        pbar = trange(epochs, desc="Training Stage 2")
        for epoch in pbar:
            self.model.train()
            running_rec = dict(loss=[])
            for z, s in train_loader:
                z = z.to(self.device)
                s = s.to(self.device)
                
                self.stage_2_optimizer.zero_grad()
                
                pi_logits = self.model.selector(z, output_probs=False)   # (n_pred, K)
                loss: torch.Tensor = F.cross_entropy(pi_logits, s)
                loss.backward()
                
                self.stage_2_optimizer.step()
                
                running_rec['loss'].append(loss.item())
            
            # evaluate metrics
            auroc, auprc, f1, prec, rec, class_mcc, mcc, nmi, ari = self.eval_metrics(eval_batch_size)
            print(f"AUROC={np.round(auroc, 2)}\nAUPRC={np.round(auprc, 2)}")
            print(f"F1={np.round(f1, 2)}\nPrec.={np.round(prec, 2)}\nRec.={np.round(rec, 2)}")
            print(f"MCC={np.round(class_mcc, 2)}")
            
            running_rec.update({
                "AUROC": auroc, "AUPRC": auprc,
                "MCC": mcc, "F1-score": f1, "Precision": prec, "Recall": rec,
                "NMI": nmi, "ARI": ari
            })
            running_rec = {k: np.mean(v) for k, v in running_rec.items()}
            
            pbar.set_postfix_str(f"loss={running_rec['loss']:.4f}")
            self.log_metrics(running_rec, epoch)

    def train_clustering(self, epochs: int, batch_size: int, eval_batch_size: int, seed=None, verbose=True):
        """Perform stage-2 pre-training (the embedding & selector)

        Args:
            epochs (int): the total iteration for training selector
            batch_size (int): the batch size for training selector
        """
        pred_token_id = self.model.tokenizer.pred_token_id
        train_loader = self.get_train_loader(batch_size=eval_batch_size)
        Z = []
        Y_hat = []
        with torch.no_grad():
            for x, y in train_loader:
                profile_input, time_input, event_input, causal_mask, padding_mask = [_x.to(self.device) for _x in x]
                y = y.to(self.device)
                z: torch.Tensor = self.model.predict_z(profile_input, time_input, event_input, causal_mask, padding_mask)  # (N, L, E)
                y_hat: torch.Tensor = self.model.predictor(z)
                pred_mask = (event_input == pred_token_id).flatten()
                Z.append(z.flatten(end_dim=1)[pred_mask].numpy(force=True))
                Y_hat.append(self.to_probs(y_hat).flatten(end_dim=1)[pred_mask].numpy(force=True))
        
        Z = np.concatenate(Z, axis=0)  # (n_pred, E)
        Y_hat = np.concatenate(Y_hat, axis=0)  # (n_pred, C)

        if verbose:
            print("Running KMeans ...")
        
        assigner = KMeans(n_clusters=self.model.K, init="k-means++", tol=1e-8, verbose=verbose, random_state=seed)
        assigner = assigner.fit(Y_hat)
        Ey = assigner.cluster_centers_
        S = assigner.predict(Y_hat).astype(np.int64)

        E = np.zeros((self.model.K, self.model.d_model))
        for k in range(self.model.K):
            E[k, :] = Z[np.argmin(np.sum(np.abs(Y_hat - Ey[k, :]), axis=-1)), :]
        
        self.model.cluster_embedding.weight = nn.Parameter(data=torch.tensor(E, dtype=torch.float32, device=self.device))
        self.train_selector(Z, S, epochs, batch_size, eval_batch_size)

    def compute_loss_3(self):
        K = self.model.K
        _normalize = F.softmax if self.predict_task == "multiclass" else F.sigmoid
        E_y = _normalize(self.model.predictor(self.model.cluster_embedding.weight.data), dim=-1)  # (K, C)
        E_y_ex = torch.vstack([torch.prod(E_y[torch.arange(K) > k], dim=0) for k in range(K)])  # (K, C)

        return self.loss_3_fn(E_y_ex, E_y, from_logits=False)

    def _train_actor(self, observation, target, class_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        pred_token_id = self.model.tokenizer.pred_token_id

        self.actor_optimizer.zero_grad()

        z = self.model.predict_z(*observation)           # (N, L, E)
        pi_logits = self.model.selector(z)  # (N, L, K)
        pi = F.softmax(pi_logits, dim=-1)   # (N, L, K)
        with torch.no_grad():
            cat = torch.distributions.Categorical(probs=pi)
            s = cat.sample()  # (N, L)
            e = self.model.cluster_embedding(s)  # (N, L, E)
            y_bar = self.model.predictor(e)  # (N, L, C)
            y = target
        
        K = self.model.K
        event_input = observation[2]
        pred_mask = (event_input == pred_token_id).flatten(end_dim=1)  # (N*L,)
        l1: torch.Tensor = self.loss_1_fn(y_bar.flatten(end_dim=1)[pred_mask], y.flatten(end_dim=1)[pred_mask], weight=class_weights, reduction="none")
        pi_sample: torch.Tensor = (F.one_hot(s, num_classes=K) * F.log_softmax(pi_logits, dim=-1)).flatten(end_dim=1)[pred_mask].sum(dim=-1)
        l1_actor: torch.Tensor = (l1 * pi_sample).mean()
        l2: torch.Tensor = self.loss_2_fn(pi_logits.flatten(end_dim=1)[pred_mask], pi.flatten(end_dim=1)[pred_mask])
        
        loss = l1_actor + self.alpha * l2
        loss.backward()

        self.actor_optimizer.step()

        return l1_actor, l2
    
    def _train_critic(self, observation, target, class_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        pred_token_id = self.model.tokenizer.pred_token_id

        self.critic_optimizer.zero_grad()

        z = self.model.predict_z(*observation)
        pi_logits = self.model.selector(z)
        cat = torch.distributions.Categorical(logits=pi_logits)
        s = cat.sample()  # (N, L) (No grad!)
        e = self.model.cluster_embedding(s)  # (N, L, E)
        y_bar = self.model.predictor(e)  # (N, L, C)
        y = target

        # loss
        event_input = observation[2]
        pred_mask = (event_input == pred_token_id).flatten(end_dim=1)  # (N*L,)
        loss: torch.Tensor = self.loss_1_fn(y_bar.flatten(end_dim=1)[pred_mask], y.flatten(end_dim=1)[pred_mask], weight=class_weights)
        loss.backward()

        self.critic_optimizer.step()

        return loss
    
    def _train_embedding(self, observation, target, class_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        pred_token_id = self.model.tokenizer.pred_token_id

        self.embedding_optimizer.zero_grad()

        # calculate the predictive clustering loss
        s = self.model.predict_s(*observation)
        e: torch.Tensor = self.model.cluster_embedding(s)
        y_bar: torch.Tensor = self.model.predictor(e)  # (N, L, C)
        y = target

        event_input = observation[2]
        pred_mask = (event_input == pred_token_id).flatten(end_dim=1)  # (N*L,)
        l1: torch.Tensor = self.loss_1_fn(y_bar.flatten(end_dim=1)[pred_mask], y.flatten(end_dim=1)[pred_mask], weight=class_weights)

        # calculate the embedding separation loss
        l3: torch.Tensor = self.compute_loss_3()

        loss = l1 + self.beta * (-l3)  # leave loss_3 negative to be maximized
        loss.backward()

        self.embedding_optimizer.step()

        return l1, l3

    def train(self, 
              epochs: int, 
              batch_size: int = 1, 
              eval_epoch: int = 0, 
              eval_batch_size: int = 1,
              pred_mask_rate: float = 0, 
              verbose=False):
        self.model.train()
        multiclass = self.predict_task == "multiclass"
        C = self.model.C

        _, train_Y = self._collate_fn(self.train_dataset)
        if pred_mask_rate > 0 and multiclass:
            train_Y = self._pred_masking(train_Y, pred_mask_rate)
        class_weights = self._compute_class_weights(train_Y, self.class_weight_scale)

        train_loader = self.get_train_loader(batch_size)
        eval_loader = self.get_eval_loader(eval_batch_size)

        history = []
        pbar = trange(epochs, desc="Training Stage 3")
        for epoch in range(epochs):
            self.model.train()

            # train the actor & critic while freezing the embedding
            self.model.requires_grad_(True)
            self.model.cluster_embedding.requires_grad_(False)
            running_rec = {
                "loss_1_actor": [],
                "loss_1_critic": [],
                "loss_1_embedding": [],
                "loss_2": [],
                "loss_3": []
            }
            pbar = tqdm(train_loader, desc=f"Epoch {(epoch+1):03} (AC)")
            for data in pbar:
                observation, target = data
                observation = [x.to(self.device) for x in observation]
                target = target.to(self.device)
                if multiclass:
                    target = self._pred_masking(target, masking_rate=pred_mask_rate)
                
                l1_critic = self._train_critic(observation, target, class_weights)
                l1_actor, l2 = self._train_actor(observation, target, class_weights)

                running_rec['loss_1_critic'].append(l1_critic.item())
                running_rec['loss_1_actor'].append(l1_actor.item())
                running_rec['loss_2'].append(l2.item())

            # train the embedding while freezing the actor & critic
            self.model.requires_grad_(False)
            self.model.cluster_embedding.requires_grad_(True)
            pbar = tqdm(train_loader, desc=f"Epoch {(epoch+1):03} (Emb)")
            for data in pbar:
                observation, target = data
                observation = [x.to(self.device) for x in observation]
                target = target.to(self.device)
                if multiclass:
                    target = self._pred_masking(target, masking_rate=pred_mask_rate)

                l1_embedding, l3 = self._train_embedding(observation, target, class_weights)

                running_rec['loss_1_embedding'].append(l1_embedding.item())
                running_rec['loss_3'].append(l3.item())

            # log the training losses
            running_rec = {k: np.mean(v) for k, v in running_rec.items()}
            self.log_metrics(running_rec, epoch)

            if eval_epoch and ((epoch+1) % eval_epoch == 0):
                self.model.eval()

                eval_l1, eval_l2, eval_l3 = self.eval_loop(eval_loader)
                auroc, auprc, f1, prec, rec, class_mcc, mcc, nmi, ari= self.eval_metrics(eval_batch_size)

                if verbose:
                    pbar.set_postfix_str(
                        f"l1_a={l1_actor:.4f} - l1_c={l1_critic:.4f} - l1_e={l1_embedding:.4f} - l2={l2:.4f} - l3={l3:.4f} | eval_l1={eval_l1:.4f} - eval_l2={eval_l2:.4f} - eval_l3={eval_l3:.4f}"
                    )
                    print(f"AUROC: {np.round(auroc, 3)}")
                    print(f"AUPRC: {np.round(auprc, 3)}")
                    print(f"MCC: {mcc:.4f} / F1: {f1:.4f} / Precision: {prec:.4f} / Recall: {rec:.4f}")
                    print(f"NMI: {nmi:.4f} / ARI: {ari:.4f}")
                else:
                    pbar.set_postfix_str(f"eval_l1={eval_l1:.4f} - eval_l2={eval_l2:.4f} - eval_l3={eval_l3:.4f}")

                running_rec.update({
                    'eval_loss_1': eval_l1, 'eval_loss_2': eval_l2, 'eval_loss_3': eval_l3,
                    "AUROC": auroc, "AUPRC": auprc,
                    "MCC": mcc, "F1-score": f1, "Precision": prec, "Recall": rec,
                    "NMI": nmi, "ARI": ari
                })
                running_rec = {k: np.mean(v) for k, v in running_rec.items()}
                self.log_metrics(running_rec, epoch)

                history.append((
                    epoch,
                    running_rec['loss_1_actor'], 
                    running_rec['loss_1_critic'],
                    running_rec['loss_1_embedding'],
                    running_rec['loss_2'], 
                    running_rec['loss_3'],
                    eval_l1, eval_l2, eval_l3,
                    mcc, 
                    *f1, *prec, *rec,
                    *auroc, *auprc,
                    nmi, ari
                ))
            
        self.history_stage3 = pd.DataFrame(data=history, columns=[
            "epoch",
            "loss_1_actor", "loss_1_critic", "loss_1_embedding", "loss_2", "loss_3", 
            "eval_loss_1", "eval_loss_2", "eval_loss_3",
            "mcc", 
            *[f"f1_{i}" for i in range(C)], 
            *[f"precision_{i}" for i in range(C)], 
            *[f"recall_{i}" for i in range(C)],
            *[f"AUROC_{i}" for i in range(C)],
            *[f"AUPRC_{i}" for i in range(C)],
            "nmi", "ari",
        ])

        return self.history_stage3

    def eval_loop(self, eval_loader: DataLoader):
        pred_token_id = self.model.tokenizer.pred_token_id

        eval_losses = {"loss_1": [], "loss_2": [], "loss_3": []}
        with torch.no_grad():
            for eval_data in eval_loader:
                x, y = eval_data
                x = [_x.to(self.device) for _x in x]
                y: torch.Tensor = y.to(self.device)

                pi_logits = self.model.predict_pi(*x)  # (N, L, K)
                pi = F.softmax(pi_logits, dim=-1)     # (N, L, K)
                s = pi.argmax(dim=-1)                 # (N, L)
                e = self.model.cluster_embedding(s)
                y_bar = self.model.predictor(e)

                event_input = x[2]
                pred_mask = (event_input == pred_token_id).flatten(end_dim=1)  # (N*L,)
                eval_l1 = self.loss_1_fn(y_bar.flatten(end_dim=1)[pred_mask], y.flatten(end_dim=1)[pred_mask]).item()
                eval_l2 = self.loss_2_fn(pi_logits.flatten(end_dim=1)[pred_mask], pi.flatten(end_dim=1)[pred_mask]).item()
                eval_l3 = self.compute_loss_3().item()

                eval_losses["loss_1"].append(eval_l1)
                eval_losses["loss_2"].append(eval_l2)
                eval_losses["loss_3"].append(eval_l3)
        
        avg_eval_l1 = np.mean(eval_losses["loss_1"])
        avg_eval_l2 = np.mean(eval_losses["loss_2"])
        avg_eval_l3 = np.mean(eval_losses["loss_3"])

        return avg_eval_l1, avg_eval_l2, avg_eval_l3

    def eval_metrics(self, batch_size: int):
        pred_token_id = self.model.tokenizer.pred_token_id
        C = self.model.C
        
        eval_loader = self.get_eval_loader(batch_size=batch_size)
        with torch.no_grad():
            assignments = []
            eval_ys = []
            eval_y_bars = []
            for eval_data in eval_loader:
                eval_x, eval_y = eval_data
                observations = [_x.to(self.device) for _x in eval_x]
                event_input = observations[2]
                eval_y: torch.Tensor = eval_y.to(self.device)
                pred_mask = (event_input == pred_token_id).flatten(end_dim=1)  # (N*L,)

                eval_s = self.model.predict_s(*observations, sample=False)  # (N, L)
                eval_y_bar = self.model.predict_y_bar(*observations)  # (N, L, C)

                eval_ys.append(eval_y.flatten(end_dim=1)[pred_mask])
                assignments.append(eval_s.flatten(end_dim=1)[pred_mask])
                eval_y_bars.append(self.to_probs(eval_y_bar).flatten(end_dim=1)[pred_mask])
        
        eval_y_pred = torch.concat(eval_y_bars, dim=0)
        eval_y_true = torch.concat(eval_ys, dim=0).to(dtype=torch.int64)
        assignments = torch.concat(assignments, dim=0).to(dtype=torch.int64)

        auroc, auprc, f1, prec, rec, class_mcc = predictive_scores(
            eval_y_pred, eval_y_true, 
            task=self.predict_task, 
            num_classes=C, 
            pred_threshold=self.predict_threshold
        )

        mcc_args = dict(
            preds=eval_y_pred, 
            target=eval_y_true,
            task=self.predict_task,
            num_classes=C,
            num_labels=C,
            threshold=self.predict_threshold
        )
        mcc = matthews_corrcoef(**mcc_args).item()

        # get clustering scores
        nmi    = normalized_mutual_info_score(assignments, eval_y_true).item()
        ari    = adjusted_rand_score(assignments, eval_y_true).item()

        return auroc, auprc, f1, prec, rec, class_mcc, mcc, nmi, ari
