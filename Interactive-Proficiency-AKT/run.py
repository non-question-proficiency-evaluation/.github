# Original code adapted from: https://github.com/jennyzhang0215/DKVMN.git
import numpy as np
import torch
import math
from sklearn import metrics
from utils import model_isPid_type
from logger import setup_logger
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Models that require data transposition for batch-first format
transpose_data_model = {'akt'}


def binaryEntropy(target: np.ndarray, pred: np.ndarray, mod: str = "avg") -> float:
    """
    Compute binary cross-entropy loss.
    
    Calculates the binary cross-entropy between target labels and predicted probabilities.
    This is used as an evaluation metric (not for training, which uses BCEWithLogitsLoss).
    
    Args:
        target (np.ndarray): True binary labels (0 or 1) of shape (n_samples,).
        pred (np.ndarray): Predicted probabilities in range [0, 1] of shape (n_samples,).
        mod (str, optional): Computation mode. 'avg' for average, 'sum' for sum.
            Defaults to "avg".
    
    Returns:
        float: Binary cross-entropy loss. Negative value (for consistency with log likelihood).
    
    Note:
        Uses small epsilon (1e-10) to prevent log(0) errors.
    """
    loss = target * np.log(np.maximum(1e-10, pred)) + \
        (1.0 - target) * np.log(np.maximum(1e-10, 1.0-pred))
    if mod == 'avg':
        return np.average(loss)*(-1.0)
    elif mod == 'sum':
        return - loss.sum()
    else:
        assert False


def compute_auc(all_target: np.ndarray, all_pred: np.ndarray) -> float:
    """
    Compute Area Under the ROC Curve (AUC) score.
    
    Calculates the AUC metric which measures the model's ability to distinguish
    between correct and incorrect answers. Higher values (closer to 1.0) indicate
    better performance.
    
    Args:
        all_target (np.ndarray): True binary labels (0 or 1) of shape (n_samples,).
        all_pred (np.ndarray): Predicted probabilities in range [0, 1] of shape (n_samples,).
    
    Returns:
        float: AUC score in range [0, 1]. 0.5 indicates random performance, 1.0 is perfect.
    
    Note:
        Uses sklearn's roc_auc_score function for computation.
    """
    #fpr, tpr, thresholds = metrics.roc_curve(all_target, all_pred, pos_label=1.0)
    return metrics.roc_auc_score(all_target, all_pred)


def compute_accuracy(all_target: np.ndarray, all_pred: np.ndarray) -> float:
    """
    Compute classification accuracy.
    
    Converts predicted probabilities to binary predictions using a 0.5 threshold
    and computes the accuracy score.
    
    Args:
        all_target (np.ndarray): True binary labels (0 or 1) of shape (n_samples,).
        all_pred (np.ndarray): Predicted probabilities in range [0, 1] of shape (n_samples,).
            Will be modified in-place to binary predictions.
    
    Returns:
        float: Accuracy score in range [0, 1], representing the fraction of correct predictions.
    
    Note:
        Modifies all_pred in-place by converting probabilities to binary predictions.
    """
    all_pred[all_pred > 0.5] = 1.0
    all_pred[all_pred <= 0.5] = 0.0
    return metrics.accuracy_score(all_target, all_pred)


def train(net: torch.nn.Module, params, optimizer: torch.optim.Optimizer,
          q_data: np.ndarray, qa_data: np.ndarray, pid_data: np.ndarray,
          label: str) -> tuple:
    """
    Train the knowledge tracing model for one epoch.
    
    This function performs a complete training epoch, including:
    - Data shuffling and batching
    - Forward pass through the model
    - Loss computation and backpropagation
    - Gradient clipping (if enabled)
    - Metric computation (loss, accuracy, AUC)
    
    Args:
        net (torch.nn.Module): The AKT model to train. Must have forward() method.
        params: Configuration object containing hyperparameters:
            - batch_size (int): Number of samples per batch
            - maxgradnorm (float): Maximum gradient norm for clipping (-1 to disable)
            - n_question (int): Number of unique questions
            - model (str): Model type identifier (e.g., 'akt_pid', 'akt_cid')
        optimizer (torch.optim.Optimizer): Optimizer for parameter updates (e.g., Adam).
        q_data (np.ndarray): Question ID sequences of shape (num_samples, seq_len).
        qa_data (np.ndarray): Question-answer combined sequences of shape (num_samples, seq_len).
        pid_data (np.ndarray): Problem ID sequences of shape (num_samples, seq_len).
            Can be None for non-PID models.
        label (str): Label for logging purposes (e.g., 'Train', 'Valid').
    
    Returns:
        tuple: A tuple containing:
            - loss (float): Average binary cross-entropy loss.
            - accuracy (float): Classification accuracy in [0, 1].
            - auc (float): Area Under ROC Curve in [0, 1].
    
    Note:
        - Data is automatically shuffled before each epoch.
        - Padding values (target < -0.9) are excluded from metric computation.
        - Gradient clipping is applied if maxgradnorm > 0.
        - Model is set to training mode (enables dropout, batch norm updates).
    """
    logger = setup_logger("akt_training")
    net.train()
    pid_flag, model_type = model_isPid_type(params.model)
    N = int(math.ceil(len(q_data) / params.batch_size))
    logger.debug(f"Training on {len(q_data)} samples, {N} batches")
    q_data = q_data.T  # Shape: (200,3633)
    qa_data = qa_data.T  # Shape: (200,3633)
    # Shuffle the data
    shuffled_ind = np.arange(q_data.shape[1])
    np.random.shuffle(shuffled_ind)
    q_data = q_data[:, shuffled_ind]
    qa_data = qa_data[:, shuffled_ind]

    if pid_flag:
        pid_data = pid_data.T
        pid_data = pid_data[:, shuffled_ind]

    pred_list = []
    target_list = []

    element_count = 0
    true_el = 0
    for idx in range(N):
        optimizer.zero_grad()

        q_one_seq = q_data[:, idx*params.batch_size:(idx+1)*params.batch_size]
        if pid_flag:
            pid_one_seq = pid_data[:, idx *
                                   params.batch_size:(idx+1) * params.batch_size]

        qa_one_seq = qa_data[:, idx *
                             params.batch_size:(idx+1) * params.batch_size]

        if model_type in transpose_data_model:
            input_q = np.transpose(q_one_seq[:, :])  # Shape (bs, seqlen)
            input_qa = np.transpose(qa_one_seq[:, :])  # Shape (bs, seqlen)
            target = np.transpose(qa_one_seq[:, :])
            if pid_flag:
                # Shape (seqlen, batch_size)
                input_pid = np.transpose(pid_one_seq[:, :])
        else:
            input_q = (q_one_seq[:, :])  # Shape (seqlen, batch_size)
            input_qa = (qa_one_seq[:, :])  # Shape (seqlen, batch_size)
            target = (qa_one_seq[:, :])
            if pid_flag:
                input_pid = (pid_one_seq[:, :])  # Shape (seqlen, batch_size)
        target = (target - 1) / params.n_question
        target_1 = np.floor(target)
        el = np.sum(target_1 >= -.9)
        element_count += el

        input_q = torch.from_numpy(input_q).long().to(device)
        input_qa = torch.from_numpy(input_qa).long().to(device)
        target = torch.from_numpy(target_1).float().to(device)
        if pid_flag:
            input_pid = torch.from_numpy(input_pid).long().to(device)

        if pid_flag:
            loss, pred, true_ct = net(input_q, input_qa, target, input_pid)
        else:
            loss, pred, true_ct = net(input_q, input_qa, target)
        pred = pred.detach().cpu().numpy()  # (seqlen * batch_size, 1)
        loss.backward()
        true_el += true_ct.cpu().numpy()

        if params.maxgradnorm > 0.:
            torch.nn.utils.clip_grad_norm_(
                net.parameters(), max_norm=params.maxgradnorm)

        optimizer.step()

        # correct: 1.0; wrong 0.0; padding -1.0
        target = target_1.reshape((-1,))

        nopadding_index = np.flatnonzero(target >= -0.9)
        nopadding_index = nopadding_index.tolist()
        pred_nopadding = pred[nopadding_index]
        target_nopadding = target[nopadding_index]

        pred_list.append(pred_nopadding)
        target_list.append(target_nopadding)

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)

    loss = binaryEntropy(all_target, all_pred)
    auc = compute_auc(all_target, all_pred)
    accuracy = compute_accuracy(all_target, all_pred)

    return loss, accuracy, auc


def test(net: torch.nn.Module, params, optimizer: torch.optim.Optimizer,
         q_data: np.ndarray, qa_data: np.ndarray, pid_data: np.ndarray,
         label: str) -> tuple:
    """
    Evaluate the knowledge tracing model on test/validation data.
    
    This function performs evaluation without gradient computation, including:
    - Data batching (no shuffling for reproducibility)
    - Forward pass through the model
    - Metric computation (loss, accuracy, AUC)
    
    Args:
        net (torch.nn.Module): The AKT model to evaluate. Must have forward() method.
        params: Configuration object containing hyperparameters:
            - batch_size (int): Number of samples per batch
            - n_question (int): Number of unique questions
            - model (str): Model type identifier (e.g., 'akt_pid', 'akt_cid')
        optimizer (torch.optim.Optimizer): Optimizer (not used during evaluation, can be None).
        q_data (np.ndarray): Question ID sequences of shape (num_samples, seq_len).
        qa_data (np.ndarray): Question-answer combined sequences of shape (num_samples, seq_len).
        pid_data (np.ndarray): Problem ID sequences of shape (num_samples, seq_len).
            Can be None for non-PID models.
        label (str): Label for logging purposes (e.g., 'Test', 'Valid').
    
    Returns:
        tuple: A tuple containing:
            - loss (float): Average binary cross-entropy loss.
            - accuracy (float): Classification accuracy in [0, 1].
            - auc (float): Area Under ROC Curve in [0, 1].
    
    Note:
        - Data is NOT shuffled to maintain evaluation consistency.
        - Padding values (target < -0.9) are excluded from metric computation.
        - Model is set to evaluation mode (disables dropout, uses batch norm statistics).
        - All computations are performed with torch.no_grad() for efficiency.
    """
    logger = setup_logger("akt_evaluation")
    # dataArray: [ array([[],[],..])] Shape: (3633, 200)
    pid_flag, model_type = model_isPid_type(params.model)
    net.eval()
    N = int(math.ceil(float(len(q_data)) / float(params.batch_size)))
    logger.debug(f"Evaluating on {len(q_data)} samples, {N} batches")
    q_data = q_data.T  # Shape: (200,3633)
    qa_data = qa_data.T  # Shape: (200,3633)
    if pid_flag:
        pid_data = pid_data.T
    seq_num = q_data.shape[1]
    pred_list = []
    target_list = []

    count = 0
    true_el = 0
    element_count = 0
    for idx in range(N):

        q_one_seq = q_data[:, idx*params.batch_size:(idx+1)*params.batch_size]
        if pid_flag:
            pid_one_seq = pid_data[:, idx *
                                   params.batch_size:(idx+1) * params.batch_size]
        input_q = q_one_seq[:, :]  # Shape (seqlen, batch_size)
        qa_one_seq = qa_data[:, idx *
                             params.batch_size:(idx+1) * params.batch_size]
        input_qa = qa_one_seq[:, :]  # Shape (seqlen, batch_size)

        # print 'seq_num', seq_num
        if model_type in transpose_data_model:
            # Shape (seqlen, batch_size)
            input_q = np.transpose(q_one_seq[:, :])
            # Shape (seqlen, batch_size)
            input_qa = np.transpose(qa_one_seq[:, :])
            target = np.transpose(qa_one_seq[:, :])
            if pid_flag:
                input_pid = np.transpose(pid_one_seq[:, :])
        else:
            input_q = (q_one_seq[:, :])  # Shape (seqlen, batch_size)
            input_qa = (qa_one_seq[:, :])  # Shape (seqlen, batch_size)
            target = (qa_one_seq[:, :])
            if pid_flag:
                input_pid = (pid_one_seq[:, :])
        target = (target - 1) / params.n_question
        target_1 = np.floor(target)
        #target = np.random.randint(0,2, size = (target.shape[0],target.shape[1]))

        input_q = torch.from_numpy(input_q).long().to(device)
        input_qa = torch.from_numpy(input_qa).long().to(device)
        target = torch.from_numpy(target_1).float().to(device)
        if pid_flag:
            input_pid = torch.from_numpy(input_pid).long().to(device)

        with torch.no_grad():
            if pid_flag:
                loss, pred, ct = net(input_q, input_qa, target, input_pid)
            else:
                loss, pred, ct = net(input_q, input_qa, target)
        pred = pred.cpu().numpy()  # (seqlen * batch_size, 1)
        true_el += ct.cpu().numpy()
        #target = target.cpu().numpy()
        if (idx + 1) * params.batch_size > seq_num:
            real_batch_size = seq_num - idx * params.batch_size
            count += real_batch_size
        else:
            count += params.batch_size

        # correct: 1.0; wrong 0.0; padding -1.0
        target = target_1.reshape((-1,))
        nopadding_index = np.flatnonzero(target >= -0.9)
        nopadding_index = nopadding_index.tolist()
        pred_nopadding = pred[nopadding_index]
        target_nopadding = target[nopadding_index]

        element_count += pred_nopadding.shape[0]
        # print avg_loss
        pred_list.append(pred_nopadding)
        target_list.append(target_nopadding)

    assert count == seq_num, "Seq not matching"

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)
    loss = binaryEntropy(all_target, all_pred)
    auc = compute_auc(all_target, all_pred)
    accuracy = compute_accuracy(all_target, all_pred)

    return loss, accuracy, auc
