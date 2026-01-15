# Original code adapted from: https://github.com/jennyzhang0215/DKVMN.git
import numpy as np
import math


class DATA(object):
    """
    Data loader for standard knowledge tracing datasets without Problem IDs.
    
    This class handles loading and preprocessing of datasets that use a 3-line format:
    - Line 1: Student ID, timestamp
    - Line 2: Question/Skill IDs sequence
    - Line 3: Answer sequence (0=incorrect, 1=correct)
    
    The loader automatically splits long sequences into multiple sequences of fixed length
    and pads shorter sequences with zeros.
    
    Args:
        n_question (int): Number of unique questions/skills in the dataset.
            Example: 110 for ASSISTments2009, 100 for ASSISTments2015.
        seqlen (int): Maximum sequence length. Sequences longer than this will be split.
            Example: 200 for most datasets.
        separate_char (str): Character used to separate values in CSV files. Default: ','.
        name (str, optional): Name identifier for the dataset. Defaults to "data".
    
    Attributes:
        separate_char (str): Separator character for CSV parsing.
        n_question (int): Number of unique questions.
        seqlen (int): Maximum sequence length.
    
    Example:
        >>> loader = DATA(n_question=100, seqlen=200, separate_char=',')
        >>> q_data, qa_data, student_ids = loader.load_data('data/assist2015_train1.csv')
        >>> print(q_data.shape)  # (num_sequences, 200)
    """
    def __init__(self, n_question: int, seqlen: int, separate_char: str, name: str = "data"):
        # In the ASSISTments2009 dataset:
        # param: n_queation = 110
        #        seqlen = 200
        self.separate_char = separate_char
        self.n_question = n_question
        self.seqlen = seqlen

    def load_data(self, path: str) -> tuple:
        """
        Load and preprocess data from a CSV file.
        
        This method reads a CSV file in the 3-line format and converts it into
        numpy arrays suitable for training. Long sequences are automatically split,
        and all sequences are padded to the same length.
        
        Args:
            path (str): Path to the CSV file containing the dataset.
                Expected format:
                - Line 0: Student ID, timestamp
                - Line 1: Question IDs (comma-separated)
                - Line 2: Answers (comma-separated, 0 or 1)
                - Repeats for each student
        
        Returns:
            tuple: A tuple containing:
                - q_dataArray (np.ndarray): Question ID sequences of shape (num_sequences, seqlen).
                    Padded with zeros for sequences shorter than seqlen.
                - qa_dataArray (np.ndarray): Combined question-answer indices of shape 
                    (num_sequences, seqlen). Formula: question_id + answer * n_question.
                    Padded with zeros.
                - idx_data (np.ndarray): Student IDs corresponding to each sequence,
                    shape (num_sequences,).
        
        Note:
            - Sequences longer than seqlen are split into multiple sequences.
            - Empty values at the end of lines are automatically removed.
            - Question-answer index calculation: Xindex = Q[i] + A[i] * n_question
        """
        try:
            f_data = open(path, 'r')
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {path}")
        except IOError as e:
            raise IOError(f"Error reading data file {path}: {str(e)}")
        
        q_data = []
        qa_data = []
        idx_data = []
        try:
            for lineID, line in enumerate(f_data):
            line = line.strip()
            # lineID starts from 0
            if lineID % 3 == 0:
                student_id = lineID//3
            if lineID % 3 == 1:
                Q = line.split(self.separate_char)
                if len(Q[len(Q)-1]) == 0:
                    Q = Q[:-1]
                # print(len(Q))
            elif lineID % 3 == 2:
                A = line.split(self.separate_char)
                if len(A[len(A)-1]) == 0:
                    A = A[:-1]
                # print(len(A),A)

                # start split the data
                n_split = 1
                # print('len(Q):',len(Q))
                if len(Q) > self.seqlen:
                    n_split = math.floor(len(Q) / self.seqlen)
                    if len(Q) % self.seqlen:
                        n_split = n_split + 1
                # print('n_split:',n_split)
                for k in range(n_split):
                    question_sequence = []
                    answer_sequence = []
                    if k == n_split - 1:
                        endINdex = len(A)
                    else:
                        endINdex = (k+1) * self.seqlen
                    for i in range(k * self.seqlen, endINdex):
                        if len(Q[i]) > 0:
                            # int(A[i]) is in {0,1}
                            Xindex = int(Q[i]) + int(A[i]) * self.n_question
                            question_sequence.append(int(Q[i]))
                            answer_sequence.append(Xindex)
                        # Skip empty question IDs (should not occur in valid data)
                    #print('instance:-->', len(instance),instance)
                    q_data.append(question_sequence)
                    qa_data.append(answer_sequence)
                    idx_data.append(student_id)
        finally:
            f_data.close()
        
        if len(q_data) == 0:
            raise ValueError(f"No data loaded from {path}. File may be empty or incorrectly formatted.")
        
        ### data: [[],[],[],...] <-- set_max_seqlen is used
        # convert data into ndarrays for better speed during training
        q_dataArray = np.zeros((len(q_data), self.seqlen))
        for j in range(len(q_data)):
            dat = q_data[j]
            q_dataArray[j, :len(dat)] = dat

        qa_dataArray = np.zeros((len(qa_data), self.seqlen))
        for j in range(len(qa_data)):
            dat = qa_data[j]
            qa_dataArray[j, :len(dat)] = dat
        # dataArray: [ array([[],[],..])] Shape: (3633, 200)
        return q_dataArray, qa_dataArray, np.asarray(idx_data)


class PID_DATA(object):
    """
    Data loader for knowledge tracing datasets with Problem IDs (PID).
    
    This class handles loading and preprocessing of datasets that use a 4-line format:
    - Line 1: Student ID, timestamp
    - Line 2: Problem IDs sequence
    - Line 3: Question/Skill IDs sequence
    - Line 4: Answer sequence (0=incorrect, 1=correct)
    
    The loader automatically splits long sequences into multiple sequences of fixed length
    and pads shorter sequences with zeros. This format is used for AKT-Rasch models that
    learn problem-specific difficulty parameters.
    
    Args:
        n_question (int): Number of unique questions/skills in the dataset.
            Example: 110 for ASSISTments2009, 102 for ASSISTments2017.
        seqlen (int): Maximum sequence length. Sequences longer than this will be split.
            Example: 200 for most datasets.
        separate_char (str): Character used to separate values in CSV files. Default: ','.
        name (str, optional): Name identifier for the dataset. Defaults to "data".
    
    Attributes:
        separate_char (str): Separator character for CSV parsing.
        seqlen (int): Maximum sequence length.
        n_question (int): Number of unique questions.
    
    Example:
        >>> loader = PID_DATA(n_question=110, seqlen=200, separate_char=',')
        >>> q_data, qa_data, pid_data = loader.load_data('data/assist2009_pid_train1.csv')
        >>> print(q_data.shape)  # (num_sequences, 200)
        >>> print(pid_data.shape)  # (num_sequences, 200)
    """
    def __init__(self, n_question: int, seqlen: int, separate_char: str, name: str = "data"):
        # In the ASSISTments2009 dataset:
        # param: n_queation = 110
        #        seqlen = 200
        self.separate_char = separate_char
        self.seqlen = seqlen
        self.n_question = n_question

    def load_data(self, path: str) -> tuple:
        """
        Load and preprocess data from a CSV file with Problem IDs.
        
        This method reads a CSV file in the 4-line format and converts it into
        numpy arrays suitable for training. Long sequences are automatically split,
        and all sequences are padded to the same length.
        
        Args:
            path (str): Path to the CSV file containing the dataset.
                Expected format:
                - Line 0: Student ID, timestamp
                - Line 1: Problem IDs (comma-separated)
                - Line 2: Question IDs (comma-separated)
                - Line 3: Answers (comma-separated, 0 or 1)
                - Repeats for each student
        
        Returns:
            tuple: A tuple containing:
                - q_dataArray (np.ndarray): Question ID sequences of shape (num_sequences, seqlen).
                    Padded with zeros for sequences shorter than seqlen.
                - qa_dataArray (np.ndarray): Combined question-answer indices of shape 
                    (num_sequences, seqlen). Formula: question_id + answer * n_question.
                    Padded with zeros.
                - p_dataArray (np.ndarray): Problem ID sequences of shape (num_sequences, seqlen).
                    Padded with zeros.
        
        Note:
            - Sequences longer than seqlen are split into multiple sequences.
            - Empty values at the end of lines are automatically removed.
            - Question-answer index calculation: Xindex = Q[i] + A[i] * n_question
            - Problem IDs are used in AKT-Rasch models for difficulty modeling.
        """
        try:
            f_data = open(path, 'r')
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {path}")
        except IOError as e:
            raise IOError(f"Error reading data file {path}: {str(e)}")
        
        q_data = []
        qa_data = []
        p_data = []
        try:
            for lineID, line in enumerate(f_data):
            line = line.strip()
            # lineID starts from 0
            if lineID % 4 == 0:
                student_id = lineID//4
            if lineID % 4 == 2:
                Q = line.split(self.separate_char)
                if len(Q[len(Q)-1]) == 0:
                    Q = Q[:-1]
                # print(len(Q))
            if lineID % 4 == 1:
                P = line.split(self.separate_char)
                if len(P[len(P) - 1]) == 0:
                    P = P[:-1]

            elif lineID % 4 == 3:
                A = line.split(self.separate_char)
                if len(A[len(A)-1]) == 0:
                    A = A[:-1]
                # print(len(A),A)

                # start split the data
                n_split = 1
                # print('len(Q):',len(Q))
                if len(Q) > self.seqlen:
                    n_split = math.floor(len(Q) / self.seqlen)
                    if len(Q) % self.seqlen:
                        n_split = n_split + 1
                # print('n_split:',n_split)
                for k in range(n_split):
                    question_sequence = []
                    problem_sequence = []
                    answer_sequence = []
                    if k == n_split - 1:
                        endINdex = len(A)
                    else:
                        endINdex = (k+1) * self.seqlen
                    for i in range(k * self.seqlen, endINdex):
                        if len(Q[i]) > 0:
                            Xindex = int(Q[i]) + int(A[i]) * self.n_question
                            question_sequence.append(int(Q[i]))
                            problem_sequence.append(int(P[i]))
                            answer_sequence.append(Xindex)
                        # Skip empty question IDs (should not occur in valid data)
                    q_data.append(question_sequence)
                    qa_data.append(answer_sequence)
                    p_data.append(problem_sequence)

        f_data.close()
        ### data: [[],[],[],...] <-- set_max_seqlen is used
        # convert data into ndarrays for better speed during training
        q_dataArray = np.zeros((len(q_data), self.seqlen))
        for j in range(len(q_data)):
            dat = q_data[j]
            q_dataArray[j, :len(dat)] = dat

        qa_dataArray = np.zeros((len(qa_data), self.seqlen))
        for j in range(len(qa_data)):
            dat = qa_data[j]
            qa_dataArray[j, :len(dat)] = dat

        p_dataArray = np.zeros((len(p_data), self.seqlen))
        for j in range(len(p_data)):
            dat = p_data[j]
            p_dataArray[j, :len(dat)] = dat
        return q_dataArray, qa_dataArray, p_dataArray
