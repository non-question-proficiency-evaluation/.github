import os
import torch
from akt import AKT
# from sakt import SAKT
# from dkvmn import DKVMN
# from dkt import DKT
# from dktplus import DKTPlus
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def try_makedirs(path_: str) -> None:
    """
    Create directory if it doesn't exist, handling race conditions gracefully.
    
    This function safely creates a directory and all necessary parent directories.
    It handles the case where the directory might be created by another process
    between the existence check and creation attempt.
    
    Args:
        path_ (str): Path to the directory to create. Can be absolute or relative.
    
    Note:
        - Does not raise an error if directory already exists.
        - Handles FileExistsError that may occur in multi-process environments.
        - Uses os.makedirs() which creates parent directories as needed.
    
    Example:
        >>> try_makedirs('model/akt_pid/assist2009_pid')
        >>> try_makedirs('result/akt_pid/assist2009_pid')
    """
    if not os.path.isdir(path_):
        try:
            os.makedirs(path_)
        except FileExistsError:
            pass


def get_file_name_identifier(params) -> list:
    """
    Generate file name identifier based on model hyperparameters.
    
    Creates a list of parameter-value pairs that uniquely identify a model
    configuration. This is used to generate unique filenames for model checkpoints
    and result files, ensuring different hyperparameter configurations don't
    overwrite each other.
    
    Args:
        params: Configuration object containing hyperparameters. Must have:
            - model (str): Model type (e.g., 'akt_pid', 'dkt', 'dkvmn')
            - batch_size (int): Batch size
            - maxgradnorm (float): Maximum gradient norm
            - lr (float): Learning rate
            - seed (int): Random seed
            - seqlen (int): Sequence length
            - train_set (int): Training fold number
            - Additional model-specific parameters (n_block, d_model, dropout, etc.)
    
    Returns:
        list: List of [prefix, value] pairs. Example:
            [['_b', 24], ['_nb', 1], ['_lr', 1e-5], ['_s', 224], ...]
    
    Note:
        Different model types include different parameters in the identifier:
        - AKT/SAKT: batch_size, n_block, maxgradnorm, lr, seed, seqlen, dropout,
                    d_model, train_set, kq_same, l2
        - DKT: batch_size, maxgradnorm, lr, seed, seqlen, d_model, train_set,
               hidden_dim, dropout, l2
        - DKVMN: batch_size, maxgradnorm, lr, seed, seqlen, q_embed_dim,
                 qa_embed_dim, train_set, memory_size, l2
    
    Example:
        >>> params.model = 'akt_pid'
        >>> params.batch_size = 24
        >>> params.n_block = 1
        >>> identifier = get_file_name_identifier(params)
        >>> # Returns: [['_b', 24], ['_nb', 1], ['_gn', -1], ...]
    """
    words = params.model.split('_')
    model_type = words[0]
    if model_type == 'dkt':
        file_name = [['_b', params.batch_size], ['_gn', params.maxgradnorm], ['_lr', params.lr],
                     ['_s', params.seed], ['_sl', params.seqlen], ['_dm', params.d_model], ['_ts', params.train_set],  ['_h', params.hidden_dim], ['_do', params.dropout], ['_l2', params.l2]]
    elif model_type == 'dktplus':
        file_name = [['_b', params.batch_size], ['_gn', params.maxgradnorm], ['_lr', params.lr],
                     ['_s', params.seed], ['_sl', params.seqlen], ['_dm', params.d_model], ['_ts', params.train_set],  ['_h', params.hidden_dim], ['_do', params.dropout], ['_l2', params.l2], ['_r', params.lamda_r], ['_w1', params.lamda_w1], ['_w2', params.lamda_w2]]
    elif model_type == 'dkvmn':
        file_name = [['_b', params.batch_size], ['_gn', params.maxgradnorm], ['_lr', params.lr],
                     ['_s', params.seed], ['_sl', params.seqlen], ['_q', params.q_embed_dim], ['_qa', params.qa_embed_dim], ['_ts', params.train_set], ['_m', params.memory_size], ['_l2', params.l2]]
    elif model_type in {'akt', 'sakt'}:
        file_name = [['_b', params.batch_size], ['_nb', params.n_block], ['_gn', params.maxgradnorm], ['_lr', params.lr],
                     ['_s', params.seed], ['_sl', params.seqlen], ['_do', params.dropout], ['_dm', params.d_model], ['_ts', params.train_set], ['_kq', params.kq_same], ['_l2', params.l2]]
    return file_name


def model_isPid_type(model_name: str) -> tuple:
    """
    Determine if a model uses Problem IDs (PID) and extract model type.
    
    Parses the model name to determine whether it's a PID-based model (Rasch)
    or CID-based model (NonRasch), and extracts the base model type.
    
    Args:
        model_name (str): Model name in format '<model_type>_<variant>'.
            Examples: 'akt_pid', 'akt_cid', 'dkt_pid', 'sakt_cid'
    
    Returns:
        tuple: A tuple containing:
            - is_pid (bool): True if model uses Problem IDs (Rasch model),
              False if it uses Concept IDs (NonRasch model).
            - model_type (str): Base model type (e.g., 'akt', 'dkt', 'sakt').
    
    Example:
        >>> is_pid, model_type = model_isPid_type('akt_pid')
        >>> # Returns: (True, 'akt')
        >>> is_pid, model_type = model_isPid_type('akt_cid')
        >>> # Returns: (False, 'akt')
    """
    words = model_name.split('_')
    is_pid = True if 'pid' in words else False
    return is_pid, words[0]


def load_model(params) -> torch.nn.Module:
    """
    Initialize and load the knowledge tracing model based on configuration.
    
    Creates a model instance with the specified architecture and hyperparameters.
    The model is automatically moved to the appropriate device (GPU if available).
    
    Args:
        params: Configuration object containing:
            - model (str): Model type identifier (e.g., 'akt_pid', 'akt_cid')
            - n_question (int): Number of unique questions/skills
            - n_pid (int): Number of unique problem IDs (set to -1 for CID models)
            - n_block (int): Number of transformer blocks
            - d_model (int): Model dimension (embedding size)
            - dropout (float): Dropout rate
            - kq_same (int): Whether query and key use same weights (1 or 0)
            - l2 (float): L2 regularization for difficulty parameters
    
    Returns:
        torch.nn.Module: Initialized model moved to the appropriate device (CPU or CUDA).
        Returns None if model_type is not supported.
    
    Note:
        - For CID models (NonRasch), n_pid is automatically set to -1.
        - Model is moved to GPU if CUDA is available, otherwise uses CPU.
        - Currently supports: 'akt' (AKT model)
        - Other model types (sakt, dkvmn, dkt) are commented out but can be enabled.
    
    Example:
        >>> params.model = 'akt_pid'
        >>> params.n_question = 110
        >>> params.n_pid = 16891
        >>> model = load_model(params)
        >>> print(next(model.parameters()).device)  # cuda:0 or cpu
    """
    words = params.model.split('_')
    model_type = words[0]
    is_cid = words[1] == 'cid'
    if is_cid:
        params.n_pid = -1

    if model_type in {'akt'}:
        model = AKT(n_question=params.n_question, n_pid=params.n_pid, n_blocks=params.n_block, d_model=params.d_model,
                    dropout=params.dropout, kq_same=params.kq_same, model_type=model_type, l2=params.l2).to(device)
    else:
        model = None
    return model
