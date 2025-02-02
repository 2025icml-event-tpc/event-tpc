import os
import json
import random
import argparse

import numpy as np
import polars as pl

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F

import mlflow

from utils import logging, load_yaml, load_estp
from nn import MLPProfileEncoder, EventTransformerEncoder, MLPSelector, MLPPredictor
from event_tpc import EventTransformerTPC, Trainer


# === constants === #
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
CLS_CX_NONE = "cx_none"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", default="", type=str)
    parser.add_argument("--run_name", default="", type=str)
    parser.add_argument("--run_desc", default="", type=str)
    parser.add_argument("--data_dir", default="data/etp_dataset", type=str)
    parser.add_argument("--eval_ratio", default=0.2, type=float)
    parser.add_argument("--test_ratio", default=0.2, type=float)
    parser.add_argument("--presplit_run_id", default="", type=str, help="Will be overwritten by pretrained_run_id if exists.")
    parser.add_argument("--stage", default=0, choices=[0, 1, 2, 3], type=int, help="Specify the training stage. 0 denotes training all stages.")
    parser.add_argument("--model_config", type=str)
    parser.add_argument('--save_best', default="none", type=str, help="Sytax: [greater or less] [metric]. Set \"none\" to disable.")
    parser.add_argument('--save_model', action="store_true")
    parser.add_argument('--pretrained_run_id', default="", type=str, help="The pre-training run id to load the pre-trained model and other configs")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def sinusoidal_temporal_encoding(time_inputs: torch.Tensor, enc_dim: int, n=10000):
    # time_inputs: (N, L)
    N, L = time_inputs.size()
    result = torch.empty(N, L, enc_dim, device=time_inputs.device)
    i_end = enc_dim // 2
    for i in range(i_end):
        denominator = pow(n, 2*i/enc_dim)
        result[:, :, 2*i] = torch.sin(time_inputs / denominator)
        result[:, :, 2*i+1] = torch.cos(time_inputs / denominator)

    if enc_dim % 2 == 1:
        result[:, :, -1] = torch.sin(time_inputs / pow(n, 2*(i_end+1)/enc_dim))

    return result


def main():
    raw_args = parse_args()
    model_cfg: dict = load_yaml(raw_args.model_config)
    args = dict(
        **vars(raw_args),
        **model_cfg
    )

    # DEBUG
    if raw_args.verbose:
        print("============ Arguments ============")
        for k, v in raw_args._get_kwargs():
            if k != "verbose":
                print(f"{k}={v}")
        print()
        print("============ Model Config ============")
        for k, v in model_cfg.items():
            print(f"{k}={v}")
        print()

    data_dir = args['data_dir']
    run_name = args['run_name']
    run_desc = args['run_desc']
    eval_ratio = args['eval_ratio']
    test_ratio = args['test_ratio']
    presplit_run_id = args['presplit_run_id']
    K = args['num_cluster']
    C = args['num_classes']
    d_model = args['d_model']
    ffn_dim = args['ffn_dim']
    attn_layers = args['attn_layers']
    fc_dim = args['fc_dim']
    dropout = args['dropout']
    stage = args['stage']
    epochs = args['epochs']
    eval_epoch = args['eval_epoch']
    batch_size = args['batch_size']
    eval_batch_size = args['eval_batch_size']
    pred_mask_rate = args['pred_mask_rate']
    weighted_sampling = args['weighted_sampling']
    class_weight_type = args['class_weight_type']
    class_weight_scale = args['class_weight_scale']
    alpha = args['alpha']
    beta = args['beta']
    lr = args['lr']
    actor_lr = args['actor_lr']
    critic_lr = args['critic_lr']
    embedding_lr = args['embedding_lr']
    save_best = dict(zip(['how', 'metric'], args['save_best'].split())) if args['save_best'] != "none" else None
    save_model = args['save_model']
    pretrained_run_id = args['pretrained_run_id']
    
    seed = args['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # setup mlflow
    mlflow_url = os.getenv("MLFLOW_URL")
    mlflow.set_tracking_uri(mlflow_url)
    logging.info(f"MLFlow URL: {mlflow_url}")

    def log_params(**additional_params):
        default_params = args.copy()
        additional_keys = additional_params.keys()

        # update key-values in default_params
        for k in additional_keys:
            default_params[k] = additional_params.pop(k)
        
        return mlflow.log_params(default_params)


    def log_model(model: nn.Module, artifact_path: str):
        return mlflow.pytorch.log_model(pytorch_model=model, artifact_path=artifact_path)

    tokenizer, estp_data = load_estp(data_dir)
    RIDs = [rid for rid, _, _ in estp_data]

    # train-test split
    if pretrained_run_id or presplit_run_id:
        if pretrained_run_id:
            logging.info("Found an existing pre-training grouped idcodes.")
            presplit_run_id = pretrained_run_id
        else:
            logging.info("Found an existing pre-splitted idcodes.")

        grouped_idcodes = mlflow.artifacts.load_dict(f"runs:/{presplit_run_id}/grouped_idcodes.json")
        train_ids         = grouped_idcodes['train_ids']
        # dropped_train_ids = grouped_idcodes['dropped_train_ids']
        eval_ids          = grouped_idcodes['eval_ids']
        test_ids          = grouped_idcodes['test_ids']
    else:
        train_eval_ids, test_ids = train_test_split(
            RIDs, 
            test_size=test_ratio, 
            random_state=seed
        )
        train_ids, eval_ids = train_test_split(
            train_eval_ids, 
            test_size=eval_ratio, 
            random_state=seed
        )

    def construct_dataset_by_idcodes(idcodes):
        return [((profile_input, torch.tensor(time_input), torch.tensor(event_input)), torch.tensor(target)) \
                  for idcode, (profile_input, time_input, event_input), target in estp_data if idcode in idcodes]
    train_data, eval_data, test_data = (construct_dataset_by_idcodes(idcodes) for idcodes in [train_ids, eval_ids, test_ids])

    # build model
    if pretrained_run_id:
        model = mlflow.pytorch.load_model(f"runs:/{pretrained_run_id}/model")
    else:
        profile_input_example = estp_data[0][1][0]
        profile_encoder = MLPProfileEncoder(input_dim=len(profile_input_example), hidden_dim=fc_dim, output_dim=d_model, initializer=nn.init.xavier_uniform_)
        event_encoder = EventTransformerEncoder(
            n_events=len(tokenizer.tokens), 
            d_model=d_model, 
            nhead=4,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation=F.relu,
            num_layers=attn_layers,
            pos_embedding=sinusoidal_temporal_encoding
        )
        selector = MLPSelector(K=K, input_dim=d_model, hidden_dim=fc_dim, dropout=dropout, initializer=nn.init.xavier_uniform_)
        predictor = MLPPredictor(
            input_dim=d_model,
            hidden_dim=fc_dim,
            output_dim=C,
            dropout=dropout,
            initializer=nn.init.xavier_uniform_
        )
        model = EventTransformerTPC(tokenizer, profile_encoder, event_encoder, selector, predictor)

    optim_cfg = dict(
        stage_1   = dict(optim_cls=torch.optim.Adam, lr=lr[0]),
        stage_2   = dict(optim_cls=torch.optim.Adam, lr=lr[1]),
        actor     = dict(optim_cls=torch.optim.Adam, lr=actor_lr),
        critic    = dict(optim_cls=torch.optim.Adam, lr=critic_lr),
        embedding = dict(optim_cls=torch.optim.Adam, lr=embedding_lr)
    )
    trainer = Trainer(model, 
                      train_dataset=train_data, 
                      eval_dataset=eval_data, 
                      predict_task="multiclass", 
                      predict_threshold=0.5,
                      weighted_sampling=weighted_sampling,
                      class_weight_type=class_weight_type,
                      class_weight_scale=class_weight_scale,
                      alpha=alpha, beta=beta, 
                      optim_config=optim_cfg, 
                      device=DEVICE, 
                      logger="mlflow")

    # =================== Training =================== #
    if stage == 0 or stage == 1:
        experiment_name = args['experiment_name'] if args['experiment_name'] else f"TPC Training Stage 1"
        experiment = mlflow.set_experiment(experiment_name)

        logging.info("Training stage 1 ...")
        logging.info(f"MLFlow Experiment ID: {experiment.experiment_id}")
        with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=run_name, description=run_desc) as run:
            print(f"MLFlow Run ID: {run.info.run_id}")
            log_params()
            mlflow.log_dict(dict(
                train_ids=train_ids, 
                # dropped_train_ids=dropped_train_ids,
                eval_ids=eval_ids, 
                test_ids=test_ids
            ), "grouped_idcodes.json")

            save_best_args = dict(
                monitor_metric=save_best['metric'] if save_best else None,
                greater_is_better=(save_best['how'] == 'greater') if save_best else None
            )

            trainer.train_prediction(
                epochs[0], 
                batch_size=batch_size[0], 
                eval_epoch=eval_epoch[0], 
                eval_batch_size=eval_batch_size[0], 
                pred_mask_rate=pred_mask_rate,
                **save_best_args
            )

            # save the model
            if save_model:
                model_info = log_model(trainer.model, artifact_path="model")
                print(f"Saved Model URI: {model_info.model_uri}")

            # update pretrained_run_id
            args['pretrained_run_id'] = run.info.run_id

    if stage == 0 or stage == 2:
        experiment_name = args['experiment_name'] if args['experiment_name'] else f"TPC Training Stage 2"
        experiment = mlflow.set_experiment(experiment_name)

        logging.info("Training stage 2 ...")
        logging.info(f"MLFlow Experiment ID: {experiment.experiment_id}")
        with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=run_name, description=run_desc) as run:
            print(f"MLFlow Run ID: {run.info.run_id}")
            log_params()
            mlflow.log_dict(dict(
                train_ids=train_ids, 
                eval_ids=eval_ids, 
                test_ids=test_ids
            ), "grouped_idcodes.json")

            trainer.train_clustering(
                epochs[1], 
                batch_size=batch_size[1], 
                eval_batch_size=eval_batch_size[1],
                seed=seed
            )

            # save the model
            if save_model:
                model_info = log_model(trainer.model, artifact_path="model")
                print(f"Saved Model URI: {model_info.model_uri}")

            # update pretrained_run_id
            args['pretrained_run_id'] = run.info.run_id

    if stage == 0 or stage == 3:
        experiment_name = args['experiment_name'] if args['experiment_name'] else f"TPC Training Stage 3"
        experiment = mlflow.set_experiment(experiment_name)

        logging.info("Training stage 3 ...")
        logging.info(f"MLFlow Experiment ID: {experiment.experiment_id}")
        with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=run_name, description=run_desc) as run:
            print(f"MLFlow Run ID: {run.info.run_id}")
            log_params()
            mlflow.log_dict(dict(
                train_ids=train_ids, 
                # dropped_train_ids=dropped_train_ids,
                eval_ids=eval_ids, 
                test_ids=test_ids
            ), "grouped_idcodes.json")
            
            trainer.train(
                epochs[2], 
                batch_size=batch_size[2], 
                eval_epoch=eval_epoch[2], 
                eval_batch_size=batch_size[2]
            )

            # save the model
            if save_model:
                model_info = log_model(trainer.model, artifact_path="model")
                mlflow.log_table(trainer.history_stage3, "history.json")
            print(f"Saved Model URI: {model_info.model_uri}")


main()
