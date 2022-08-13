from functools import partial
from jax import random
import jax.numpy as np
from jax.scipy.linalg import block_diag
from seq_model import BatchClassificationModel, RetrievalModel
from ssm import init_S5SSM
from ssm_init import make_Normal_HiPPO
from dataloading import Datasets
from train_helpers import create_train_state, train_epoch_normal, \
    train_epoch_vocab, validate_normal, validate_vocab, \
    reduce_lr_on_plateau, linear_warmup, cosine_annealing, constant_lr
import wandb


def train(
        USE_WANDB=False,  # log with wandb
        wandb_project=None,  # wandb project name
        wandb_entity=None,  # wandb entity name
        dir_name="./data",  # data directory
        dataset="imdb-classification",
        n_layers=4,  # Number of layers of network
        d_model=64,  # Number of features, i.e. H, dimension of layer inputs/outputs
        ssm_size=64,  # SSM Latent size, i.e. P
        blocks=1,  # How many blocks to initialize A with, see Appendix discussion
        BC_init="dense",  # B, C init approach, [dense, dense_columns, factorized]
        k=-1,  # rank to use for low-rank factorization
        discretization="zoh",  # options: [zoh, bilinear]
        mode="pool",  # use mean pooling or just take last element of state sequence
        prenorm=True,  # True: use prenorm, False: use postnorm
        batchnorm=False,  # True: use batchnorm, False: no batchnorm
        bn_momentum=-1,  # batchnorm momentum if used
        bsz=50,  # batch size
        epochs=40,
        early_stop_patience=15,
        ssm_lr=0.0095,  # learning rate to use for SSM parameters
        lr_factor=3,  # learning rate of non-SSM parameters is lr_factor*ssm_lr
        cosine_anneal=False,  # True: use cosine annealing, False: no annealing
        warmup_end=0,  # Epoch to end linear warmup on; 0 means no warmup
        lr_patience=5,  # patience to use for decay learning rate on plateau
        reduce_factor=0.2,  # decay rate for decay learning rate on plateau
        p_dropout=0.12,  # dropout rate
        weight_decay=0.008,
        opt_config="noBCdecay",  # B and/or C weight decay Options: [noBdecay, BandCdecay, noBCdecay]
        jax_seed=8915814  # set JAX randomness, affects B,C,D,Delta init and dropout
        ):
    """Main function to train over a certain number of epochs"""

    if USE_WANDB:
        # Make wandb config dictionary
        config = {
                "dataset": dataset,
                "n_layers": n_layers,
                "d_model": d_model,
                "ssm_size": ssm_size,
                "blocks": blocks,
                "BC_init": BC_init,
                "k": k,
                "discretization": discretization,
                "mode": mode,
                "prenorm": prenorm,
                "batchnorm": batchnorm,
                "bn_momentum": bn_momentum,
                "bsz": bsz,
                "epochs": epochs,
                "early_stop_patience": early_stop_patience,
                "ssm_lr": ssm_lr,
                "lr_factor": lr_factor,
                "cosine_anneal":  cosine_anneal,
                "warmup_end":    warmup_end,
                "lr_patience": lr_patience,
                "reduce_factor": reduce_factor,
                "p_dropout": p_dropout,
                "weight_decay": weight_decay,
                "opt_config": opt_config,
                "jax_seed": jax_seed,
                 }

        wandb.init(project=wandb_project, job_type='model_training',
                   config=config, entity=wandb_entity)
    else:
        wandb.init(mode='offline')

    # determine the size of initial blocks
    block_size = int(ssm_size / blocks)
    wandb.log({"block_size": block_size})

    # Set non-ssm learning rate lr (e.g. encoders, etc.) as functino of ssm_lr
    lr = lr_factor * ssm_lr

    # Set randomness...
    print("[*] Setting Randomness...")
    key = random.PRNGKey(jax_seed)
    init_rng, train_rng = random.split(key, num=2)

    # Get dataset creation function
    create_dataset_fn = Datasets[dataset]

    # Have to use different train_epoch function if inputs need to be one_hotted (for now...)
    if dataset in ["imdb-classification", "listops-classification", "aan-classification"]:
        print("Using Vocab train and val steps because dataset is: " + dataset)
        train_epoch = train_epoch_vocab
        validate = validate_vocab
        padded = True
        if dataset in ["aan-classification"]:
            # Use retreival model for document matching
            retrieval = True
            print("Using retreival model for document matching")
        else:
            retrieval = False
    else:
        train_epoch = train_epoch_normal
        validate = validate_normal
        padded = False
        retrieval = False

    # For speech dataset
    if dataset in ["speech-classification"]:
        speech = True
        print("Will evaluate on both resolutions for speech task")
    else:
        speech = False

    # For pathx dataset
    if dataset in ["pathx-classification"]:
        print("initializing with smaller timescales for pathx")
        dt_min = 0.0001
        dt_max = 0.01
    else:
        dt_min = 0.001
        dt_max = 0.1

    # Create dataset...
    if speech:
        trainloader, valloader, testloader, valloader2, testloader2, n_classes, seq_len, in_dim, train_size = create_dataset_fn(
                                                                                                dir_name,
                                                                                                bsz=bsz)

    else:
        trainloader, valloader, testloader, n_classes, seq_len, in_dim, train_size = create_dataset_fn(dir_name,
                                                                                                       bsz=bsz)
    print(f"[*] Starting S5 Training on `{dataset}` =>> Initializing...")

    # Initialize state matrix A using approximation to HiPPO-LegS matrix
    Lambda, V = make_Normal_HiPPO(block_size)
    Vc = V.conj().T

    # If initializing state matrix A as block-diagonal, put HiPPO approximation
    # on each block
    Lambda = (Lambda*np.ones((blocks, block_size))).ravel()
    V = block_diag(*([V] * blocks))
    Vinv = block_diag(*([Vc] * blocks))

    ssm_init_fn = init_S5SSM(H=d_model,
                             P=ssm_size,
                             k=k,
                             Lambda_init=Lambda,
                             V=V,
                             Vinv=Vinv,
                             BC_init=BC_init,
                             discretization=discretization,
                             dt_min=dt_min,
                             dt_max=dt_max
                             )

    if retrieval:
        # Use retrieval head for AAN task
        print("Using Retrieval head for {} task".format(dataset))
        model_cls = partial(
            RetrievalModel,
            ssm=ssm_init_fn,
            d_output=n_classes,
            d_model=d_model,
            n_layers=n_layers,
            padded=padded,
            dropout=p_dropout,
            prenorm=prenorm,
            batchnorm=batchnorm,
            bn_momentum=bn_momentum,
        )

    else:
        model_cls = partial(
            BatchClassificationModel,
            ssm=ssm_init_fn,
            d_output=n_classes,
            d_model=d_model,
            n_layers=n_layers,
            padded=padded,
            dropout=p_dropout,
            mode=mode,
            prenorm=prenorm,
            batchnorm=batchnorm,
            bn_momentum=bn_momentum,
        )

    # initialize training state
    state = create_train_state(
        model_cls,
        init_rng,
        padded,
        retrieval,
        in_dim=in_dim,
        bsz=bsz,
        seq_len=seq_len,
        weight_decay=weight_decay,
        batchnorm=batchnorm,
        opt_config=opt_config,
        ssm_lr=ssm_lr,
        lr=lr
        )

    # Training Loop over epochs
    best_loss, best_acc, best_epoch = 10000, 0, 0  # Note: this best loss is val_loss
    count, best_val_loss = 0, 10000  # Note: this line is for early stopping purposes
    lr_count, opt_acc = 0, 0.0  # This line is for learning rate decay
    step = 0  # for per step learning rate decay
    steps_per_epoch = int(train_size/bsz)
    for epoch in range(epochs):
        print(f"[*] Starting Training Epoch {epoch + 1}...")

        if epoch < warmup_end:
            print("using linear warmup for epoch {}".format(epoch+1))
            decay_function = linear_warmup
            end_step = steps_per_epoch * warmup_end

        elif cosine_anneal:
            print("using cosine annealing for epoch {}".format(epoch+1))
            decay_function = cosine_annealing
            end_step = steps_per_epoch * epochs - (steps_per_epoch * warmup_end)  # for per step learning rate decay
        else:
            print("using constant lr for epoch {}".format(epoch+1))
            decay_function = constant_lr
            end_step = None

        lr_params = (decay_function, ssm_lr, lr, step, end_step, opt_config)

        train_rng, skey = random.split(train_rng)
        state, train_loss, step = train_epoch(state,
                                              skey,
                                              model_cls,
                                              trainloader,
                                              seq_len,
                                              in_dim,
                                              batchnorm,
                                              lr_params)

        if valloader is not None:
            print(f"[*] Running Epoch {epoch + 1} Validation...")
            val_loss, val_acc = validate(state,
                                         model_cls,
                                         valloader,
                                         seq_len,
                                         in_dim,
                                         batchnorm
                                         )

            print(f"[*] Running Epoch {epoch + 1} Test...")
            test_loss, test_acc = validate(state,
                                           model_cls,
                                           testloader,
                                           seq_len,
                                           in_dim,
                                           batchnorm)

            print(f"\n=>> Epoch {epoch + 1} Metrics ===")
            print(
                f"\tTrain Loss: {train_loss:.5f} -- Val Loss: {val_loss:.5f} --Test Loss: {test_loss:.5f} --"
                f" Val Accuracy: {val_acc:.4f}"
                f" Test Accuracy: {test_acc:.4f}"
            )

        else:
            # else use test set as validation set
            print(f"[*] Running Epoch {epoch + 1} Test...")
            val_loss, val_acc = validate(state,
                                         model_cls,
                                         testloader,
                                         seq_len,
                                         in_dim,
                                         batchnorm)

            print(f"\n=>> Epoch {epoch + 1} Metrics ===")
            print(
                f"\tTrain Loss: {train_loss:.5f}  --Test Loss: {val_loss:.5f} --"
                f" Test Accuracy: {val_acc:.4f}"
            )

        # For early stopping purposes
        if val_loss < best_val_loss:
            count = 0
            best_val_loss = val_loss
        else:
            count += 1

        if val_acc > best_acc:
            count = 0
            best_loss, best_acc, best_epoch = val_loss, val_acc, epoch
            if valloader is not None:
                best_test_loss, best_test_acc = test_loss, test_acc
            else:
                best_test_loss, best_test_acc = best_loss, best_acc

            if speech:
                # Evaluate on resolution 2 val and test sets
                print(f"[*] Running Epoch {epoch + 1} Res 2 Validation...")
                val2_loss, val2_acc = validate(state,
                                               model_cls,
                                               valloader2,
                                               seq_len,
                                               in_dim,
                                               batchnorm,
                                               step_scale=2.0)

                print(f"[*] Running Epoch {epoch + 1} Res 2 Test...")
                test2_loss, test2_acc = validate(
                    state, model_cls, testloader2, seq_len, in_dim, batchnorm,
                    step_scale=2.0)
                print(f"\n=>> Epoch {epoch + 1} Res 2 Metrics ===")
                print(
                    f"\tVal2 Loss: {val2_loss:.5f} --Test2 Loss: {test2_loss:.5f} --"
                    f" Val Accuracy: {val2_acc:.4f}"
                    f" Test Accuracy: {test2_acc:.4f}"
                )

        # For learning rate decay purposes:
        input = lr, ssm_lr, lr_count, val_acc, opt_acc
        lr, ssm_lr, lr_count, opt_acc = reduce_lr_on_plateau(input,
                                                             factor=reduce_factor,
                                                             patience=lr_patience)

        # Print best accuracy & loss so far...
        print(
            f"\tBest Val Loss: {best_loss:.5f} -- Best Val Accuracy:"
            f" {best_acc:.4f} at Epoch {best_epoch + 1}\n"
            f"\tBest Test Loss: {best_test_loss:.5f} -- Best Test Accuracy:"
            f" {best_test_acc:.4f} at Epoch {best_epoch + 1}\n"
        )

        if valloader is not None:
            if speech:
                wandb.log(
                    {
                        "Training Loss": train_loss,
                        "Val loss": val_loss,
                        "Val Accuracy": val_acc,
                        "Test Loss": test_loss,
                        "Test Accuracy": test_acc,
                        "Val2 loss": val2_loss,
                        "Val2 Accuracy": val2_acc,
                        "Test2 Loss": test2_loss,
                        "Test2 Accuracy": test2_acc,
                        "count": count,
                        "Learning rate count": lr_count,
                        "Opt acc": opt_acc,
                        "lr": state.opt_state.inner_states['regular'].inner_state.hyperparams['learning_rate'],
                        "ssm_lr": state.opt_state.inner_states['ssm'].inner_state.hyperparams['learning_rate']
                    }
                )
            else:
                wandb.log(
                    {
                        "Training Loss": train_loss,
                        "Val loss": val_loss,
                        "Val Accuracy": val_acc,
                        "Test Loss": test_loss,
                        "Test Accuracy": test_acc,
                        "count": count,
                        "Learning rate count": lr_count,
                        "Opt acc": opt_acc,
                        "lr": state.opt_state.inner_states['regular'].inner_state.hyperparams['learning_rate'],
                        "ssm_lr": state.opt_state.inner_states['ssm'].inner_state.hyperparams['learning_rate']
                    }
                )

        else:
            wandb.log(
                {
                    "Training Loss": train_loss,
                    "Val loss": val_loss,
                    "Val Accuracy": val_acc,
                    "count": count,
                    "Learning rate count": lr_count,
                    "Opt acc": opt_acc,
                    "lr": state.opt_state.inner_states['regular'].inner_state.hyperparams['learning_rate'],
                    "ssm_lr": state.opt_state.inner_states['ssm'].inner_state.hyperparams['learning_rate']
                }
            )
        wandb.run.summary["Best Val Loss"] = best_loss
        wandb.run.summary["Best Val Accuracy"] = best_acc
        wandb.run.summary["Best Epoch"] = best_epoch
        wandb.run.summary["Best Test Loss"] = best_test_loss
        wandb.run.summary["Best Test Accuracy"] = best_test_acc

        if count > early_stop_patience:
            break


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--USE_WANDB", type=str2bool, default=False)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument("--dir_name", type=str)

    parser.add_argument(
        "--dataset", type=str, choices=Datasets.keys(), default='imdb-classification'
    )

    # Model Parameters
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--ssm_size", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=1)
    parser.add_argument("--BC_init", type=str)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--discretization", type=str, default="zoh")
    parser.add_argument("--mode", type=str, default="pool")

    # Optimization Parameters
    parser.add_argument("--prenorm", type=str2bool, default=False)
    parser.add_argument("--batchnorm", type=str2bool, default=False)
    parser.add_argument("--bn_momentum", type=float, default=0.90)
    parser.add_argument("--bsz", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--early_stop_patience", type=int, default=20)
    parser.add_argument("--ssm_lr", type=float, default=1e-3)
    parser.add_argument("--lr_factor", type=float, default=1)
    parser.add_argument("--cosine_anneal", type=str2bool, default=False)
    parser.add_argument("--warmup_end", type=int, default=0)
    parser.add_argument("--lr_patience", type=int, default=10)
    parser.add_argument("--reduce_factor", type=float, default=0.2)
    parser.add_argument("--p_dropout", type=float, default=0.2)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--opt_config", default="standard", type=str)  # ["standard", "Bdecay", "noCdecay"]
    parser.add_argument("--jax_seed", type=int, default=-1)

    args = parser.parse_args()

    train(
        USE_WANDB=args.USE_WANDB,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        dir_name=args.dir_name,
        dataset=args.dataset,
        n_layers=args.n_layers,
        d_model=args.d_model,
        ssm_size=args.ssm_size,
        blocks=args.blocks,
        BC_init=args.BC_init,
        k=args.k,
        discretization=args.discretization,
        mode=args.mode,
        prenorm=args.prenorm,
        batchnorm=args.batchnorm,
        bn_momentum=args.bn_momentum,
        bsz=args.bsz,
        epochs=args.epochs,
        early_stop_patience=args.early_stop_patience,
        ssm_lr=args.ssm_lr,
        lr_factor=args.lr_factor,
        cosine_anneal=args.cosine_anneal,
        warmup_end=args.warmup_end,
        lr_patience=args.lr_patience,
        reduce_factor=args.reduce_factor,
        p_dropout=args.p_dropout,
        weight_decay=args.weight_decay,
        opt_config=args.opt_config,
        jax_seed=args.jax_seed
        )
