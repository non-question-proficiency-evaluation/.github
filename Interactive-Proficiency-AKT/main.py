"""
Main training and evaluation script for AKT (Context-Aware Attentive Knowledge Tracing).

This module provides the entry point for training and evaluating AKT models on
various knowledge tracing datasets. It handles command-line argument parsing,
dataset configuration, model training, and evaluation.

Usage:
    python main.py --dataset assist2009_pid --model akt_pid
    python main.py --dataset assist2015 --model akt_cid --max_iter 300
"""
import os
import glob
import argparse
import numpy as np
import torch
from load_data import DATA, PID_DATA
from run import train, test
from utils import try_makedirs, load_model, get_file_name_identifier
from logger import setup_logger
from early_stopping import EarlyStopping
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# assert torch.cuda.is_available(), "No Cuda available, AssertionError"


def train_one_dataset(params, file_name: str, train_q_data: np.ndarray, 
                     train_qa_data: np.ndarray, train_pid: np.ndarray,
                     valid_q_data: np.ndarray, valid_qa_data: np.ndarray,
                     valid_pid: np.ndarray) -> int:
    """
    Train AKT model on a single dataset fold with validation.
    
    This function performs the complete training loop including:
    - Model and optimizer initialization
    - Training loop with validation at each epoch
    - Model checkpointing (saves best model based on validation AUC)
    - Early stopping if no improvement for 40 epochs
    - Logging of all metrics to result files
    
    Args:
        params: Configuration object containing all hyperparameters:
            - max_iter (int): Maximum number of training epochs
            - lr (float): Learning rate
            - model (str): Model type identifier
            - save (str): Dataset name for saving models/results
        file_name (str): Unique identifier for this training run (based on hyperparameters).
        train_q_data (np.ndarray): Training question sequences of shape (num_samples, seq_len).
        train_qa_data (np.ndarray): Training question-answer sequences of shape (num_samples, seq_len).
        train_pid (np.ndarray): Training problem IDs of shape (num_samples, seq_len).
            Can be None for non-PID models.
        valid_q_data (np.ndarray): Validation question sequences of shape (num_samples, seq_len).
        valid_qa_data (np.ndarray): Validation question-answer sequences of shape (num_samples, seq_len).
        valid_pid (np.ndarray): Validation problem IDs of shape (num_samples, seq_len).
            Can be None for non-PID models.
    
    Returns:
        int: Epoch number with the best validation AUC.
    
    Note:
        - Models are saved in: model/<model_type>/<dataset>/<file_name>_<epoch>
        - Results are saved in: result/<model_type>/<dataset>/<file_name>
        - Only the best model (highest validation AUC) is kept
        - Training stops early if no improvement for 40 consecutive epochs
    """
    # ================================== model initialization ==================================

    logger = setup_logger("akt_training")
    
    model = load_model(params)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params.lr, betas=(0.9, 0.999), eps=1e-8)
    
    # Learning rate scheduler (optional)
    use_scheduler = getattr(params, 'use_lr_scheduler', 1) == 1
    scheduler = None
    if use_scheduler:
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, 
            verbose=False, min_lr=1e-7
        )
        logger.info("Learning rate scheduler enabled (ReduceLROnPlateau)")

    logger.info("=" * 60)
    logger.info("Starting training")
    logger.info(f"Model: {params.model}, Dataset: {params.data_name}")
    logger.info("=" * 60)

    # ================================== start training ==================================
    all_train_loss = {}
    all_train_accuracy = {}
    all_train_auc = {}
    all_valid_loss = {}
    all_valid_accuracy = {}
    all_valid_auc = {}
    best_valid_auc = 0

    for idx in range(params.max_iter):
        # Train Model
        try:
            train_loss, train_accuracy, train_auc = train(
                model, params, optimizer, train_q_data, train_qa_data, train_pid,  label='Train')
        except Exception as e:
            logger.error(f"Error during training at epoch {idx + 1}: {str(e)}")
            raise
        
        # Validation step
        try:
            valid_loss, valid_accuracy, valid_auc = test(
                model,  params, optimizer, valid_q_data, valid_qa_data, valid_pid, label='Valid')
        except Exception as e:
            logger.error(f"Error during validation at epoch {idx + 1}: {str(e)}")
            raise

        logger.info(f"Epoch {idx + 1}/{params.max_iter}")
        logger.info(f"Valid AUC: {valid_auc:.4f}, Train AUC: {train_auc:.4f}")
        logger.info(f"Valid Accuracy: {valid_accuracy:.4f}, Train Accuracy: {train_accuracy:.4f}")
        logger.info(f"Valid Loss: {valid_loss:.4f}, Train Loss: {train_loss:.4f}")
        
        # Update learning rate scheduler
        if scheduler is not None:
            scheduler.step(valid_auc)
            current_lr = optimizer.param_groups[0]['lr']
            logger.debug(f"Current learning rate: {current_lr:.2e}")

        try_makedirs('model')
        try_makedirs(os.path.join('model', params.model))
        try_makedirs(os.path.join('model', params.model, params.save))

        all_valid_auc[idx + 1] = valid_auc
        all_train_auc[idx + 1] = train_auc
        all_valid_loss[idx + 1] = valid_loss
        all_train_loss[idx + 1] = train_loss
        all_valid_accuracy[idx + 1] = valid_accuracy
        all_train_accuracy[idx + 1] = train_accuracy

        # output the epoch with the best validation auc
        if valid_auc > best_valid_auc:
            path = os.path.join('model', params.model,
                                params.save,  file_name) + '_*'
            for i in glob.glob(path):
                os.remove(i)
            best_valid_auc = valid_auc
            best_epoch = idx+1
            logger.info(f"New best model at epoch {best_epoch} with validation AUC: {best_valid_auc:.4f}")
            
            # Enhanced checkpoint saving
            checkpoint_path = os.path.join('model', params.model, params.save,
                                          file_name) + '_' + str(idx+1)
            checkpoint = {
                'epoch': idx,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'best_valid_auc': best_valid_auc,
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'train_auc': train_auc,
                'valid_loss': valid_loss,
                'valid_accuracy': valid_accuracy,
                'valid_auc': valid_auc,
                'config': vars(params)  # Save all hyperparameters
            }
            torch.save(checkpoint, checkpoint_path)
            logger.debug(f"Checkpoint saved to: {checkpoint_path}")
        # Check early stopping
        if early_stopping(valid_auc):
            logger.info(f"Early stopping triggered at epoch {idx + 1}")
            logger.info(f"No improvement for {early_stopping.patience} epochs")
            logger.info(f"Best validation AUC: {best_valid_auc:.4f} at epoch {best_epoch}")
            break
        
        # Legacy early stopping (if no improvement for 40 epochs)
        if idx - best_epoch > 40:
            logger.info(f"Early stopping: No improvement for 40 epochs. Best epoch: {best_epoch}")
            break   

    try_makedirs('result')
    try_makedirs(os.path.join('result', params.model))
    try_makedirs(os.path.join('result', params.model, params.save))
    f_save_log = open(os.path.join(
        'result', params.model, params.save, file_name), 'w')
    f_save_log.write("valid_auc:\n" + str(all_valid_auc) + "\n\n")
    f_save_log.write("train_auc:\n" + str(all_train_auc) + "\n\n")
    f_save_log.write("valid_loss:\n" + str(all_valid_loss) + "\n\n")
    f_save_log.write("train_loss:\n" + str(all_train_loss) + "\n\n")
    f_save_log.write("valid_accuracy:\n" + str(all_valid_accuracy) + "\n\n")
    f_save_log.write("train_accuracy:\n" + str(all_train_accuracy) + "\n\n")
    f_save_log.close()
    return best_epoch


def test_one_dataset(params, file_name: str, test_q_data: np.ndarray,
                     test_qa_data: np.ndarray, test_pid: np.ndarray,
                     best_epoch: int) -> None:
    """
    Evaluate the trained model on test data.
    
    This function loads the best model checkpoint (based on validation AUC) and
    evaluates it on the test set. After evaluation, all model checkpoints are
    deleted to save disk space.
    
    Args:
        params: Configuration object containing:
            - model (str): Model type identifier
            - save (str): Dataset name for loading models
        file_name (str): Unique identifier for this training run (must match training).
        test_q_data (np.ndarray): Test question sequences of shape (num_samples, seq_len).
        test_qa_data (np.ndarray): Test question-answer sequences of shape (num_samples, seq_len).
        test_pid (np.ndarray): Test problem IDs of shape (num_samples, seq_len).
            Can be None for non-PID models.
        best_epoch (int): Epoch number of the best model checkpoint to load.
    
    Returns:
        None: Prints test metrics to console. Does not return values.
    
    Note:
        - Loads model from: model/<model_type>/<dataset>/<file_name>_<best_epoch>
        - Deletes all model checkpoints after evaluation
        - Prints test AUC, accuracy, and loss to console
    """
    logger = setup_logger("akt_testing")
    logger.info("=" * 60)
    logger.info(f"Starting testing with best model from epoch {best_epoch}")
    logger.info("=" * 60)
    model = load_model(params)

    checkpoint_path = os.path.join('model', params.model, params.save, file_name) + '_'+str(best_epoch)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Log checkpoint information if available
        if 'best_valid_auc' in checkpoint:
            logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', best_epoch)}")
            logger.info(f"Checkpoint validation AUC: {checkpoint.get('best_valid_auc', 'N/A'):.4f}")
    except FileNotFoundError:
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        raise
    except KeyError as e:
        logger.error(f"Error loading checkpoint: missing key {str(e)}")
        raise

    test_loss, test_accuracy, test_auc = test(
        model, params, None, test_q_data, test_qa_data, test_pid, label='Test')
    logger.info("=" * 60)
    logger.info("Test Results:")
    logger.info(f"Test AUC: {test_auc:.4f}")
    logger.info(f"Test Accuracy: {test_accuracy:.4f}")
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info("=" * 60)

    # Now Delete all the models
    path = os.path.join('model', params.model, params.save,  file_name) + '_*'
    for i in glob.glob(path):
        os.remove(i)


if __name__ == '__main__':
    """
    Main entry point for training and evaluating AKT models.
    
    This script:
    1. Parses command-line arguments for hyperparameters and dataset selection
    2. Configures dataset-specific parameters
    3. Loads training, validation, and test data
    4. Trains the model and selects best epoch based on validation AUC
    5. Evaluates the best model on test data
    
    Command-line arguments are organized into categories:
    - Basic parameters: max_iter, train_set, seed
    - Training parameters: batch_size, lr, optimizer, gradient clipping
    - Model architecture: d_model, n_blocks, n_heads, dropout, etc.
    - Dataset selection: dataset name
    
    The script automatically configures dataset-specific parameters based on
    the selected dataset (assist2009_pid, assist2017_pid, assist2015, statics).
    """
    # Parse Arguments
    parser = argparse.ArgumentParser(description='Train and evaluate AKT knowledge tracing model')
    # Basic Parameters
    parser.add_argument('--max_iter', type=int, default=300,
                        help='number of iterations')
    parser.add_argument('--train_set', type=int, default=1)
    parser.add_argument('--seed', type=int, default=224, help='default seed')

    # Common parameters
    parser.add_argument('--optim', type=str, default='adam',
        help='Default Optimizer')
    parser.add_argument('--batch_size', type=int,
        default=24, help='the batch size')
    parser.add_argument('--lr', type=float, default=1e-5,
        help='learning rate')
    parser.add_argument('--use_early_stopping', type=int, default=1,
        help='Enable early stopping (1) or disable (0)')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
        help='Number of epochs to wait before early stopping')
    parser.add_argument('--maxgradnorm', type=float,
                        default=-1, help='maximum gradient norm')
    parser.add_argument('--final_fc_dim', type=int, default=512,
                        help='hidden state dim for final fc layer')

    # AKT Specific Parameter
    parser.add_argument('--d_model', type=int, default=256,
                        help='Transformer d_model shape')
    parser.add_argument('--d_ff', type=int, default=1024,
                        help='Transformer d_ff shape')
    parser.add_argument('--dropout', type=float,
                        default=0.05, help='Dropout rate')
    parser.add_argument('--n_block', type=int, default=1,
                        help='number of blocks')
    parser.add_argument('--n_head', type=int, default=8,
                        help='number of heads in multihead attention')
    parser.add_argument('--kq_same', type=int, default=1)

    # AKT-R Specific Parameter
    parser.add_argument('--l2', type=float,
                        default=1e-5, help='l2 penalty for difficulty')

    # DKVMN Specific  Parameter
    parser.add_argument('--q_embed_dim', type=int, default=50,
                        help='question embedding dimensions')
    parser.add_argument('--qa_embed_dim', type=int, default=256,
                        help='answer and question embedding dimensions')
    parser.add_argument('--memory_size', type=int,
                        default=50, help='memory size')
    parser.add_argument('--init_std', type=float, default=0.1,
                        help='weight initialization std')
    # DKT Specific Parameter
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--lamda_r', type=float, default=0.1)
    parser.add_argument('--lamda_w1', type=float, default=0.1)
    parser.add_argument('--lamda_w2', type=float, default=0.1)

    # Datasets and Model
    parser.add_argument('--model', type=str, default='akt_pid',
                        help="combination of akt/sakt/dkvmn/dkt (mandatory), pid/cid (mandatory) separated by underscore '_'. For example tf_pid")
    parser.add_argument('--dataset', type=str, default="assist2009_pid")

    params = parser.parse_args()
    dataset = params.dataset

    if dataset in {"assist2009_pid"}:
        params.n_question = 110
        params.batch_size = 24
        params.seqlen = 200
        params.data_dir = 'data/'+dataset
        params.data_name = dataset
        params.n_pid = 16891

    if dataset in {"assist2017_pid"}:
        params.batch_size = 24
        params.seqlen = 200
        params.data_dir = 'data/'+dataset
        params.data_name = dataset
        params.n_question = 102
        params.n_pid = 3162

    if dataset in {"assist2015"}:
        params.n_question = 100
        params.batch_size = 24
        params.seqlen = 200
        params.data_dir = 'data/'+dataset
        params.data_name = dataset

    if dataset in {"statics"}:
        params.n_question = 1223
        params.batch_size = 24
        params.seqlen = 200
        params.data_dir = 'data/'+dataset
        params.data_name = dataset

    params.save = params.data_name
    params.load = params.data_name

    # Setup
    if "pid" not in params.data_name:
        dat = DATA(n_question=params.n_question,
                   seqlen=params.seqlen, separate_char=',')
    else:
        dat = PID_DATA(n_question=params.n_question,
                       seqlen=params.seqlen, separate_char=',')
    seedNum = params.seed
    np.random.seed(seedNum)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seedNum)
    np.random.seed(seedNum)
    file_name_identifier = get_file_name_identifier(params)

    # Initialize logger for main script
    logger = setup_logger("akt_main")
    
    ###Train- Test
    logger.info("Configuration Parameters:")
    d = vars(params)
    for key in d:
        logger.info(f"\t{key}: {d[key]}")
    file_name = ''
    for item_ in file_name_identifier:
        file_name = file_name+item_[0] + str(item_[1])

    train_data_path = params.data_dir + "/" + \
        params.data_name + "_train"+str(params.train_set)+".csv"
    valid_data_path = params.data_dir + "/" + \
        params.data_name + "_valid"+str(params.train_set)+".csv"

    try:
        train_q_data, train_qa_data, train_pid = dat.load_data(train_data_path)
        valid_q_data, valid_qa_data, valid_pid = dat.load_data(valid_data_path)
    except (FileNotFoundError, IOError, ValueError) as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

    logger.info("Data loaded successfully:")
    logger.info(f"Train Q data shape: {train_q_data.shape}")
    logger.info(f"Train QA data shape: {train_qa_data.shape}")
    logger.info(f"Valid Q data shape: {valid_q_data.shape}")
    logger.info(f"Valid QA data shape: {valid_qa_data.shape}")
    # Train and get the best episode
    best_epoch = train_one_dataset(
        params, file_name, train_q_data, train_qa_data, train_pid, valid_q_data, valid_qa_data, valid_pid)
    test_data_path = params.data_dir + "/" + \
        params.data_name + "_test"+str(params.train_set)+".csv"
    try:
        test_q_data, test_qa_data, test_index = dat.load_data(test_data_path)
        test_one_dataset(params, file_name, test_q_data,
                         test_qa_data, test_index, best_epoch)
    except (FileNotFoundError, IOError, ValueError) as e:
        logger.error(f"Error loading or testing on test data: {str(e)}")
        raise
