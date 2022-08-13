from functools import partial
from jax import jit, random, value_and_grad
import jax.numpy as np
from jax.nn import one_hot
from tqdm import tqdm
from flax.training import train_state
import optax
from typing import Any


# LR schedulers
def reduce_lr_on_plateau(input, factor=0.2, patience=20):
    lr, ssm_lr, count, new_acc, opt_acc = input
    if new_acc > opt_acc:
        count = 0
        opt_acc = new_acc
    else:
        count += 1

    if count > patience:
        lr = factor * lr
        ssm_lr = factor * ssm_lr
        count = 0
    return lr, ssm_lr, count, opt_acc


def linear_warmup(step, base_lr, end_step):
    return base_lr * (step + 1) / end_step


def cosine_annealing(step, base_lr, end_step, lr_min=0):
    # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
    return lr_min + 0.5 * (base_lr - lr_min) * (1+np.cos(step/end_step * np.pi))


def constant_lr(step, base_lr, end_step):
    return base_lr


def update_learning_rate_per_step(lr_params, state):
    decay_function, ssm_lr, lr, step, end_step, opt_config = lr_params

    # Get decayed value
    lr_val = decay_function(step, lr, end_step)
    ssm_lr_val = decay_function(step, ssm_lr, end_step)
    step += 1

    # Update state
    state.opt_state.inner_states['regular'].inner_state.hyperparams['learning_rate'] = np.array(lr_val,
                                                                                                dtype=np.float32)
    state.opt_state.inner_states['ssm'].inner_state.hyperparams['learning_rate'] = np.array(ssm_lr_val,
                                                                                            dtype=np.float32)
    if opt_config in ["Bdecay"]:
        state.opt_state.inner_states['none'].inner_state.hyperparams['learning_rate'] = np.array(ssm_lr_val,
                                                                                                 dtype=np.float32)

    return state, step


def map_nested_fn(fn):
    """Recursively apply `fn to the key-value pairs of a nested dict / pytree.
    We use this for some of the optax definitions below.
    """

    def map_fn(nested_dict):
        return {
            k: (map_fn(v) if hasattr(v, "keys") else fn(k, v))
            for k, v in nested_dict.items()
        }

    return map_fn


def create_train_state(model_cls,
                       rng,
                       padded,
                       retrieval,
                       in_dim=1,
                       bsz=128,
                       seq_len=784,
                       weight_decay=0.01,
                       batchnorm=False,
                       opt_config="noBCdecay",
                       ssm_lr=1e-3,
                       lr=1e-3
                       ):
    """Initializes the training state using optax"""

    if padded:
        if retrieval:
            # For retrieval tasks we have two different sets of "documents"
            dummy_input = (np.ones((2*bsz, seq_len, in_dim)), np.ones(2*bsz))
        else:
            dummy_input = (np.ones((bsz, seq_len, in_dim)), np.ones(bsz))
    else:
        dummy_input = np.ones((bsz, seq_len, in_dim))

    model = model_cls(training=True)
    init_rng, dropout_rng = random.split(rng, num=2)
    variables = model.init(
                            {"params": init_rng,
                             "dropout": dropout_rng},
                            dummy_input,
                           )
    if batchnorm:
        params = variables["params"].unfreeze()
        batch_stats = variables["batch_stats"]
    else:
        params = variables["params"].unfreeze()
        # Note: Added immediate `unfreeze()` to play well w/ Optax. See below!

    # TODO: We currently have optionality to use a different learning rate and
    #      weight decay for different parameters in the SSM (e.g. B and C).
    #      A lot of this optionality does not seem to be strictly necessary and
    #      can probably be removed/simplified in future versions
    if opt_config in ["noBdecay"]:
        """This option applies weight decay to C, but B is kept with the
            SSM parameters with no weight decay.
        """
        print("configuring optimization with weight decay on C but not B")
        ssm_fn = map_nested_fn(
            lambda k, _: "ssm"
            if k in ["Lambda", "B", "BH", "BP", "D", "log_step",  "norm"]
            else ("none" if k in [] else "regular")
        )
        tx = optax.multi_transform(
            {
                "none": optax.inject_hyperparams(optax.sgd)(learning_rate=0.0),
                "ssm": optax.inject_hyperparams(optax.adam)(learning_rate=ssm_lr),
                "regular": optax.inject_hyperparams(optax.adamw)(learning_rate=lr,
                                                                 weight_decay=weight_decay),
            },
            ssm_fn,
        )
    elif opt_config in ["BandCdecay"]:
        """This option applies weight decay to both C and B. Note we still apply the
           ssm learning rate to B.
        """
        print("configuring optimization with weight decay on B and C")
        ssm_fn = map_nested_fn(
            lambda k, _: "ssm"
            if k in ["Lambda", "D", "log_step", "norm"]
            else ("none" if k in ["BH", "BP", "B"] else "regular")
        )
        tx = optax.multi_transform(
            {
                "none": optax.inject_hyperparams(optax.adamw)(learning_rate=ssm_lr,
                                                              weight_decay=weight_decay),
                "ssm": optax.inject_hyperparams(optax.adam)(learning_rate=ssm_lr),
                "regular": optax.inject_hyperparams(optax.adamw)(learning_rate=lr,
                                                                 weight_decay=weight_decay),
            },
            ssm_fn,
        )

    elif opt_config in ["noBCdecay"]:
        """This option does not apply weight decay to B or C. C is included 
           with the SSM parameters.
        """
        print("configuring optimization with no weight decay on B or C")
        ssm_fn = map_nested_fn(
            lambda k, _: "ssm"
            if k in ["Lambda", "B", "BH", "BP", "C", "CH", "CP", "D", "log_step", "norm"]
            else ("none" if k in [] else "regular")
        )
        tx = optax.multi_transform(
            {
                "none": optax.inject_hyperparams(optax.sgd)(learning_rate=0.0),
                "ssm": optax.inject_hyperparams(optax.adam)(learning_rate=ssm_lr),
                "regular": optax.inject_hyperparams(optax.adamw)(learning_rate=lr,
                                                                 weight_decay=weight_decay),
            },
            ssm_fn,
        )

    if batchnorm:
        class TrainState(train_state.TrainState):
            batch_stats: Any

        return TrainState.create(
                                apply_fn=model.apply,
                                params=params,
                                tx=tx,
                                batch_stats=batch_stats
                                 )

    else:
        return train_state.TrainState.create(
                                    apply_fn=model.apply,
                                    params=params,
                                    tx=tx
                                             )


# Train and eval steps
@partial(np.vectorize, signature="(c),()->()")
def cross_entropy_loss(logits, label):
    one_hot_label = one_hot(label, num_classes=logits.shape[0])
    return -np.sum(one_hot_label * logits)


@partial(np.vectorize, signature="(c),()->()")
def compute_accuracy(logits, label):
    return np.argmax(logits) == label


def train_epoch_normal(state, rng, model, trainloader, seq_len, in_dim, batchnorm, lr_params):
    """Standard training function for an epoch that loops over batches. For data that does
       not require a vocabulary and does not have ragged sequence lengths"""
    # Store Metrics
    model = model(training=True)
    batch_losses = []

    decay_function, ssm_lr, lr, step, end_step, opt_config = lr_params

    for batch_idx, (inputs, labels) in enumerate(tqdm(trainloader)):
        inputs = np.array(inputs.numpy())
        labels = np.array(labels.numpy())
        rng, drop_rng = random.split(rng)
        state, loss = train_step(
            state,
            drop_rng,
            inputs,
            labels,
            model,
            batchnorm)
        batch_losses.append(loss)

        # perform per step learning rate decay
        lr_params = (decay_function, ssm_lr, lr, step, end_step, opt_config)
        state, step = update_learning_rate_per_step(lr_params, state)

    # Return average loss over batches
    return state, np.mean(np.array(batch_losses)), step


def train_epoch_vocab(state, rng, model, trainloader, seq_len, in_dim, batchnorm, lr_params):
    """
    Training function for an epoch that loops over batches when the data requires a vocabulary and
    may have ragged sequence lengths (e.g. IMDB, listops, AAN). These dataset loaders include a
    lengths vector for the original lengths of the sequence before padding. The inputs are also stored
    as integers so we have to one-hot them here.
    """
    # Store Metrics
    model = model(training=True)
    batch_losses = []

    decay_function, ssm_lr, lr, step, end_step, opt_config = lr_params

    for batch_idx, (inputs, labels, lengths) in enumerate(tqdm(trainloader)):
        # Make all batches have same sequence length
        num_pad = seq_len - inputs.shape[1]
        # Assuming vocab padding value is zero
        inputs = np.pad(np.array(inputs.numpy()), ((0, 0), (0, num_pad)),
                        'constant', constant_values=(0,))
        inputs = one_hot(inputs, in_dim)
        labels = np.array(labels.numpy())  # Not the most efficient...
        lengths = np.array(lengths.numpy())
        rng, drop_rng = random.split(rng)
        state, loss = train_step(
            state,
            drop_rng,
            (inputs, lengths),
            labels,
            model,
            batchnorm
        )
        batch_losses.append(loss)
        lr_params = (decay_function, ssm_lr, lr, step, end_step, opt_config)
        state, step = update_learning_rate_per_step(lr_params, state)

    # Return average loss over batches
    return state, np.mean(np.array(batch_losses)), step


def validate_normal(state, model, testloader, seq_len, in_dim, batchnorm, step_scale=1.0):
    """Standard validation function that loops over batches"""
    model = model(training=False, step_scale=step_scale)
    losses, accuracies = [], []
    for batch_idx, (inputs, labels) in enumerate(tqdm(testloader)):
        inputs = np.array(inputs.numpy())
        labels = np.array(labels.numpy())  # Not the most efficient...
        loss, acc = eval_step(inputs, labels, state, model, batchnorm)
        losses.append(loss)
        accuracies.append(acc)

    return np.mean(np.array(losses)), np.mean(np.array(accuracies))


def validate_vocab(state, model, testloader, seq_len, in_dim, batchnorm):
    """Validation function for data that requires building a vocabulary. Loops over batches"""
    # Compute average loss & accuracy
    model = model(training=False)
    losses, accuracies = [], []
    for batch_idx, (inputs, labels, lengths) in enumerate(tqdm(testloader)):
        # Make all batches have same sequence length
        num_pad = seq_len - inputs.shape[1]
        inputs = np.pad(np.array(inputs.numpy()), ((0, 0), (0, num_pad)),
                        'constant', constant_values=(0,))
        inputs = one_hot(inputs, in_dim)
        labels = np.array(labels.numpy())  # Not the most efficient...
        lengths = np.array(lengths.numpy())
        loss, acc = eval_step(
            (inputs, lengths), labels, state, model, batchnorm)
        losses.append(loss)
        accuracies.append(acc)

    return np.mean(np.array(losses)), np.mean(np.array(accuracies))


@partial(jit, static_argnums=(4, 5))
def train_step(state,
               rng,
               batch_inputs,
               batch_labels,
               model,
               batchnorm
               ):
    """Performs a single training step given a batch of data"""
    def loss_fn(params):
        if batchnorm:
            logits, mod_vars = model.apply(
                {"params": params, "batch_stats": state.batch_stats},
                batch_inputs,
                rngs={"dropout": rng},
                mutable=["intermediates", "batch_stats"],
            )
        else:
            logits, mod_vars = model.apply(
                {"params": params},
                batch_inputs,
                rngs={"dropout": rng},
                mutable=["intermediates"],
            )
        loss = np.mean(cross_entropy_loss(logits, batch_labels))
        return loss, (mod_vars, logits)

    grad_fn = value_and_grad(loss_fn, has_aux=True)
    aux, grads = grad_fn(state.params)
    loss = aux[0]
    mod_vars, logits = aux[1]

    if batchnorm:
        state = state.apply_gradients(grads=grads, batch_stats=mod_vars["batch_stats"])
    else:
        state = state.apply_gradients(grads=grads)
    return state, loss


@partial(jit, static_argnums=(3, 4))
def eval_step(batch_inputs, batch_labels, state, model, batchnorm):
    """Performs a single evaluation step given a batch of data"""
    if batchnorm:
        logits = model.apply({"params": state.params, "batch_stats": state.batch_stats},
                             batch_inputs)
    else:
        logits = model.apply({"params": state.params}, batch_inputs)

    loss = np.mean(cross_entropy_loss(logits, batch_labels))
    acc = np.mean(compute_accuracy(logits, batch_labels))
    return loss, acc
