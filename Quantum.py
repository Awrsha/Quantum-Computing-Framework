# --- Core Libraries ---
import numpy as np                     # Fundamental package for numerical operations, especially on n-dimensional arrays.
import time                            # Provides time-related functions, e.g., for tracking execution time in profiling.
import random                          # Standard library for generating random numbers and sequences.

# --- Concurrency and Threading ---
import threading                       # Provides low-level support for threading, essential for parallel task management.
from queue import PriorityQueue        # A specialized Queue where elements are retrieved based on priority (min-heap).

# --- Data Visualization ---
import matplotlib.pyplot as plt        # Comprehensive library for static, animated, and interactive plotting.
from scipy.stats import norm           # Import Gaussian distribution for statistical functions (PDF, CDF) often needed in analysis.

# --- Graph Theory and Networks ---
import networkx as nx                  # Network analysis and graph theory; crucial for graph-based machine learning models.

# --- Typing and Type Hinting ---
from typing import List, Dict, Tuple, Any  # Type hints to improve code readability and IDE-assisted debugging support.

# --- Abstract Base Classes for Interfaces ---
from abc import ABC, abstractmethod    # Enables the creation of abstract base classes for a clear object-oriented design.

# --- Logging and Debugging ---
import logging                         # Standard library for tracking events that happen during execution, essential for debugging.

# --- JSON Operations ---
import json                            # Lightweight data-interchange format; often used for config files, saving model parameters.

# --- OS-level Operations ---
import os                              # Operating system interfaces for file and directory manipulation.

# --- High-level Concurrency ---
import concurrent.futures              # Higher-level API for asynchronous execution using ThreadPool and ProcessPool Executors.

# --- Data Classes for Structured Data ---
from dataclasses import dataclass, field   # Provides a decorator and functions for generating boilerplate code for class instances.

# --- Enumeration ---
from enum import Enum, auto            # Enum for defining enumerated constants, ensuring clarity in variable states.

# --- Customizable Logging Configurations ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  
# Set up logging configurations: 'INFO' level for general updates, with time-stamped logs for precise event tracking.

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuantumGate(Enum):
    """Enum representing various quantum gates commonly used in quantum computing.
    
    These gates perform fundamental operations on qubits, with Hadamard and Pauli
    gates acting as single-qubit gates, CNOT and SWAP as two-qubit gates, and
    other gates for phase and rotation transformations.
    """
    HADAMARD = auto()
    CNOT = auto()
    PAULI_X = auto()
    PAULI_Y = auto()
    PAULI_Z = auto()
    PHASE = auto()
    T = auto()
    SWAP = auto()

class QuantumState:
    """Represents a quantum state vector of one or more qubits with state manipulation.
    
    This class models the initialization, gate application, entanglement, and 
    measurement of a quantum state. Each state is initialized to |0> and allows
    operations on individual qubits or on entangled qubit pairs.
    """
    
    def __init__(self, name: str, num_qubits: int = 1):
        self.name = name
        self.num_qubits = num_qubits
        self.state = self.initialize_state()

    def initialize_state(self) -> np.ndarray:
        """Initialize quantum state as a vector in the computational basis.
        
        Sets the system to |0> state, represented by a vector with the first
        element as 1 and all others as 0.
        """
        state = np.zeros(2**self.num_qubits, dtype=np.complex128)
        state[0] = 1  # Initialize to |0> state
        return state

    def apply_gate(self, gate: QuantumGate, target_qubit: int, control_qubit: int = None):
        """Apply specified quantum gate to the state, either single or two-qubit.

        Args:
            gate (QuantumGate): The type of quantum gate to apply.
            target_qubit (int): The target qubit for the gate.
            control_qubit (int, optional): For two-qubit gates, specifies the control qubit.
        
        Raises:
            ValueError: If a control qubit is missing for CNOT or SWAP gate.
        """
        if gate == QuantumGate.HADAMARD:
            self._apply_hadamard(target_qubit)
        elif gate == QuantumGate.CNOT:
            if control_qubit is None:
                raise ValueError("CNOT gate requires a control qubit")
            self._apply_cnot(control_qubit, target_qubit)
        elif gate == QuantumGate.PAULI_X:
            self._apply_pauli_x(target_qubit)
        elif gate == QuantumGate.PAULI_Y:
            self._apply_pauli_y(target_qubit)
        elif gate == QuantumGate.PAULI_Z:
            self._apply_pauli_z(target_qubit)
        elif gate == QuantumGate.PHASE:
            self._apply_phase(target_qubit)
        elif gate == QuantumGate.T:
            self._apply_t(target_qubit)
        elif gate == QuantumGate.SWAP:
            if control_qubit is None:
                raise ValueError("SWAP gate requires two qubits")
            self._apply_swap(target_qubit, control_qubit)

    def _apply_hadamard(self, target_qubit: int):
        """Applies Hadamard gate to a target qubit, creating superposition."""
        h_gate = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        self._apply_single_qubit_gate(h_gate, target_qubit)

    def _apply_cnot(self, control_qubit: int, target_qubit: int):
        """Applies CNOT gate using control and target qubits.
        
        The target qubit flips only if the control qubit is in the |1> state.
        """
        cnot = np.eye(4, dtype=np.complex128)
        cnot[2:, 2:] = np.array([[0, 1], [1, 0]])
        self._apply_two_qubit_gate(cnot, control_qubit, target_qubit)

    def _apply_pauli_x(self, target_qubit: int):
        """Applies Pauli-X (NOT) gate, which flips the state of the target qubit."""
        x_gate = np.array([[0, 1], [1, 0]])
        self._apply_single_qubit_gate(x_gate, target_qubit)

    def _apply_pauli_y(self, target_qubit: int):
        """Applies Pauli-Y gate, introducing a phase shift of π for the |1> state."""
        y_gate = np.array([[0, -1j], [1j, 0]])
        self._apply_single_qubit_gate(y_gate, target_qubit)

    def _apply_pauli_z(self, target_qubit: int):
        """Applies Pauli-Z gate, introducing a π phase shift for the |1> state."""
        z_gate = np.array([[1, 0], [0, -1]])
        self._apply_single_qubit_gate(z_gate, target_qubit)

    def _apply_phase(self, target_qubit: int):
        """Applies Phase (S) gate, rotating the |1> state by π/2 radians."""
        s_gate = np.array([[1, 0], [0, 1j]])
        self._apply_single_qubit_gate(s_gate, target_qubit)

    def _apply_t(self, target_qubit: int):
        """Applies T gate, rotating the |1> state by π/4 radians."""
        t_gate = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])
        self._apply_single_qubit_gate(t_gate, target_qubit)

    def _apply_swap(self, qubit1: int, qubit2: int):
        """Applies SWAP gate to exchange the states of two qubits."""
        swap_gate = np.array([[1, 0, 0, 0],
                              [0, 0, 1, 0],
                              [0, 1, 0, 0],
                              [0, 0, 0, 1]])
        self._apply_two_qubit_gate(swap_gate, qubit1, qubit2)

    def _apply_single_qubit_gate(self, gate: np.ndarray, target_qubit: int):
        """Constructs a full system matrix for a single-qubit gate and applies it."""
        n = self.num_qubits
        gate_full = np.eye(2**n, dtype=np.complex128)
        for i in range(2**n):
            if i & (1 << (n - target_qubit - 1)):
                i_target = i
                i_other = i & ~(1 << (n - target_qubit - 1))
                gate_full[i, i_target] = gate[1, 1]
                gate_full[i, i_other] = gate[1, 0]
            else:
                i_target = i | (1 << (n - target_qubit - 1))
                i_other = i
                gate_full[i, i_target] = gate[0, 1]
                gate_full[i, i_other] = gate[0, 0]
        self.state = np.dot(gate_full, self.state)

    def _apply_two_qubit_gate(self, gate: np.ndarray, qubit1: int, qubit2: int):
        """Constructs and applies a full system matrix for a two-qubit gate."""
        n = self.num_qubits
        gate_full = np.eye(2**n, dtype=np.complex128)
        for i in range(2**n):
            i1 = (i >> (n - qubit1 - 1)) & 1
            i2 = (i >> (n - qubit2 - 1)) & 1
            for j in range(2**n):
                j1 = (j >> (n - qubit1 - 1)) & 1
                j2 = (j >> (n - qubit2 - 1)) & 1
                if (i & ~((1 << (n - qubit1 - 1)) | (1 << (n - qubit2 - 1)))) == \
                   (j & ~((1 << (n - qubit1 - 1)) | (1 << (n - qubit2 - 1)))):
                    gate_full[i, j] = gate[i1 * 2 + i2, j1 * 2 + j2]
        self.state = np.dot(gate_full, self.state)

    def measure(self) -> int:
        """Measures the quantum state, collapsing it to a classical outcome.
        
        Returns:
            int: The index of the measured state in the basis vector.
        """
        probabilities = np.abs(self.state)**2
        result = np.random.choice(len(probabilities), p=probabilities)
        self.state = np.zeros_like(self.state)
        self.state[result] = 1
        return result

    def get_probabilities(self) -> np.ndarray:
        """Returns the probability distribution of measuring each possible state."""
        return np.abs(self.state)**2

    def entangle(self, other: 'QuantumState'):
        """Combines this quantum state with another through tensor product entanglement."""
        self.state = np.kron(self.state, other.state)
        self.num_qubits += other.num_qubits

    def __str__(self) -> str:
        return f"QuantumState(name={self.name}, num_qubits={self.num_qubits})"

class QuantumCircuit:
    """Represents a quantum circuit with multiple qubits and gates."""

    def __init__(self, num_qubits: int):
        """
        Initialize a quantum circuit with the specified number of qubits.
        
        Parameters:
            num_qubits (int): Number of qubits in the circuit.
        """
        self.num_qubits = num_qubits
        self.state = QuantumState("Circuit", num_qubits)
        self.gates: List[Tuple[QuantumGate, int, int]] = []
    
    def add_gate(self, gate: QuantumGate, target_qubit: int, control_qubit: int = None):
        """
        Add a quantum gate to the circuit, specifying target and optionally control qubits.
        
        Parameters:
            gate (QuantumGate): The gate to apply (e.g., H, X, CNOT).
            target_qubit (int): Index of the qubit the gate acts upon.
            control_qubit (int, optional): For controlled gates, the index of the control qubit.
        """
        self.gates.append((gate, target_qubit, control_qubit))

    def run(self) -> QuantumState:
        """
        Execute the quantum circuit by applying each gate in sequence, returning the final state.
        
        Returns:
            QuantumState: The state after running the circuit.
        """
        for gate, target, control in self.gates:
            self.state.apply_gate(gate, target, control)
        return self.state

    def measure(self) -> int:
        """
        Perform a measurement on the quantum circuit to collapse the state to a definite outcome.
        
        Returns:
            int: The result of the measurement, in binary representation.
        """
        return self.state.measure()

    def visualize(self):
        """
        Visualize the quantum circuit using matplotlib, illustrating qubit connections and gate positions.
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_ylim(-0.5, self.num_qubits - 0.5)
        ax.set_xlim(-0.5, len(self.gates) + 0.5)
        ax.set_yticks(range(self.num_qubits))
        ax.set_yticklabels([f'Q{i}' for i in range(self.num_qubits)])
        ax.set_xticks(range(len(self.gates)))
        ax.set_xticklabels([f'Step {i}' for i in range(len(self.gates))])
        ax.grid(True)

        for i, (gate, target, control) in enumerate(self.gates):
            # Plot CNOT gates with connection between control and target qubits
            if gate == QuantumGate.CNOT:
                ax.plot([i, i], [control, target], 'k-', linewidth=2)
                ax.plot(i, control, 'ko', markersize=10)
                ax.plot(i, target, 'k^', markersize=10)
            # Plot other gates with labeled text
            else:
                ax.text(i, target, gate.name, ha='center', va='center',
                        bbox=dict(facecolor='white', edgecolor='black'))

        plt.title("Quantum Circuit Visualization")
        plt.tight_layout()
        plt.show()

class QuantumAlgorithm(ABC):
    """Abstract base class for quantum algorithms."""
    
    @abstractmethod
    def run(self) -> Any:
        """
        Run the quantum algorithm and return the computed result.
        
        Returns:
            Any: The output of the algorithm, which could vary depending on the algorithm.
        """
        pass
        
class GroverSearch(QuantumAlgorithm):
    """Implementation of Grover's search algorithm, which provides a quadratic speedup for unstructured search problems."""

    def __init__(self, n_qubits: int, target_state: int):
        """
        Initialize the Grover search algorithm.

        Parameters:
            n_qubits (int): Number of qubits representing the search space.
            target_state (int): The target state to be found in the search space, represented in binary.
        """
        self.n_qubits = n_qubits
        self.target_state = target_state
        self.circuit = QuantumCircuit(n_qubits)

    def _oracle(self):
        """Apply the oracle operator to mark the target state by flipping its phase."""
        # Apply Pauli-X gates to the target qubits to prepare for CNOT
        for i in range(self.n_qubits):
            if (self.target_state >> i) & 1:
                self.circuit.add_gate(QuantumGate.PAULI_X, i)

        # Apply CNOT gate to flip the target state (assumes the last qubit is the output qubit)
        self.circuit.add_gate(QuantumGate.CNOT, self.n_qubits - 1, self.n_qubits - 2)

        # Reapply the Pauli-X gates to return to the original state
        for i in range(self.n_qubits):
            if (self.target_state >> i) & 1:
                self.circuit.add_gate(QuantumGate.PAULI_X, i)

    def _diffusion(self):
        """Apply the diffusion operator (inversion about the mean) to amplify the marked state's probability."""
        # Apply Hadamard to all qubits to create superposition
        for i in range(self.n_qubits):
            self.circuit.add_gate(QuantumGate.HADAMARD, i)
            self.circuit.add_gate(QuantumGate.PAULI_X, i)

        # Apply CNOT to control qubit to invert the state about the mean
        self.circuit.add_gate(QuantumGate.CNOT, self.n_qubits - 1, self.n_qubits - 2)

        # Invert and return to the original state
        for i in range(self.n_qubits):
            self.circuit.add_gate(QuantumGate.PAULI_X, i)
            self.circuit.add_gate(QuantumGate.HADAMARD, i)

    def run(self) -> int:
        """Execute Grover's search algorithm and return the result of the measurement."""
        # Initialize qubits in superposition
        for i in range(self.n_qubits):
            self.circuit.add_gate(QuantumGate.HADAMARD, i)

        # Calculate the number of iterations based on the number of qubits
        n_iterations = int(np.pi / 4 * np.sqrt(2**self.n_qubits))

        # Perform the Grover iterations: oracle followed by diffusion
        for _ in range(n_iterations):
            self._oracle()
            self._diffusion()

        # Measure the final state to obtain the index of the target state
        return self.circuit.measure()


class QuantumFourierTransform(QuantumAlgorithm):
    """Implementation of Quantum Fourier Transform (QFT), which converts a quantum state into its frequency domain representation."""

    def __init__(self, n_qubits: int):
        """
        Initialize the Quantum Fourier Transform algorithm.

        Parameters:
            n_qubits (int): Number of qubits to be transformed using QFT.
        """
        self.n_qubits = n_qubits
        self.circuit = QuantumCircuit(n_qubits)

    def _swap_qubits(self):
        """Swap the qubits to reverse their order, required for QFT output."""
        for i in range(self.n_qubits // 2):
            self.circuit.add_gate(QuantumGate.SWAP, i, self.n_qubits - 1 - i)

    def run(self) -> QuantumState:
        """Execute the Quantum Fourier Transform and return the transformed quantum state."""
        # Apply Hadamard gates to each qubit to create superposition
        for i in range(self.n_qubits):
            self.circuit.add_gate(QuantumGate.HADAMARD, i)
            # Apply controlled phase rotation between qubits
            for j in range(i + 1, self.n_qubits):
                angle = 2 * np.pi / 2**(j - i + 1)  # Calculate the rotation angle
                self.circuit.add_gate(QuantumGate.CNOT, j, i)  # Control from j to i
                self.circuit.add_gate(QuantumGate.PHASE, j)     # Apply phase rotation
                self.circuit.add_gate(QuantumGate.CNOT, j, i)  # Reverse control from j to i

        # Perform the qubit order swap to complete the QFT
        self._swap_qubits()
        # Return the resulting quantum state after the QFT has been applied
        return self.circuit.run()

class QuantumTask:
    """Represents a quantum computing task with enhanced features for dependency management and execution tracking."""

    def __init__(self, task_id: int, complexity: int, priority: int, algorithm: QuantumAlgorithm):
        """
        Initializes a QuantumTask instance.

        Args:
            task_id (int): A unique identifier for the task.
            complexity (int): A numerical representation of the task's complexity, influencing execution time.
            priority (int): An integer representing the task's priority level, affecting its execution order.
            algorithm (QuantumAlgorithm): An instance of a quantum algorithm to be executed for this task.

        Attributes:
            state (QuantumState): Represents the current state of the task.
            dependencies (List[QuantumTask]): A list of other QuantumTask instances that must be completed before this task.
            result (Any): Stores the result of the task once executed.
            execution_times (List[float]): Records the execution times for tracking performance.
            start_time (float): Timestamp indicating when the task execution started.
            end_time (float): Timestamp indicating when the task execution ended.
            status (TaskStatus): Enum value representing the current status of the task.
        """
        self.task_id = task_id
        self.complexity = complexity
        self.priority = priority
        self.algorithm = algorithm
        self.state = QuantumState(f"Task-{task_id}")
        self.dependencies: List[QuantumTask] = []
        self.result: Any = None
        self.execution_times: List[float] = []
        self.start_time: float = None
        self.end_time: float = None
        self.status = TaskStatus.PENDING

    def add_dependency(self, task: 'QuantumTask'):
        """
        Adds a dependency to this task.

        Args:
            task (QuantumTask): Another QuantumTask instance that must be completed before this one.
        """
        self.dependencies.append(task)

    def execute(self) -> bool:
        """
        Executes the quantum task if all dependencies have been completed.

        The method checks if all dependencies are satisfied (i.e., completed). If so, it simulates the execution
        time based on the task's complexity, runs the specified quantum algorithm, and logs the execution time.

        Returns:
            bool: True if the task was executed successfully, False otherwise.
        """
        if self.start_time is None:
            self.start_time = time.time()

        # Check if all dependencies have been resolved
        for dep in self.dependencies:
            if dep.result is None:  # If any dependency hasn't finished, return False
                return False

        # Simulate execution time based on complexity
        execution_time = self.complexity * random.uniform(0.5, 1.5)
        time.sleep(execution_time)  # Simulate time taken for execution
        
        try:
            self.result = self.algorithm.run()  # Execute the quantum algorithm
            self.status = TaskStatus.COMPLETED  # Mark the task as completed
        except Exception as e:
            logger.error(f"Error executing task {self.task_id}: {str(e)}")  # Log any errors that occur
            self.status = TaskStatus.FAILED  # Mark the task as failed
            return False

        self.execution_times.append(execution_time)  # Record execution time
        self.end_time = time.time()  # Capture end time
        return True

    def adjust_priority(self):
        """
        Adjusts the task's priority based on its historical execution times.

        The priority is recalculated as the inverse of the average execution time, ensuring that faster tasks are prioritized.
        """
        if self.execution_times:
            average_time = sum(self.execution_times) / len(self.execution_times)
            self.priority = max(1, int(10 / average_time))  # Higher priority for quicker tasks
    
    def execution_duration(self) -> float:
        """
        Calculates the total duration of the task execution.

        Returns:
            float: The time duration from start to end of the task execution.
        """
        return self.end_time - self.start_time if self.end_time else 0

    def __lt__(self, other: 'QuantumTask') -> bool:
        """
        Compares this task with another for priority-based sorting.

        This allows QuantumTask instances to be compared using their priority.

        Args:
            other (QuantumTask): Another QuantumTask instance to compare against.

        Returns:
            bool: True if this task has a higher priority than the other task.
        """
        return self.priority > other.priority

    def __str__(self) -> str:
        """
        Provides a string representation of the QuantumTask.

        Returns:
            str: A string that includes the task ID, priority, and current status.
        """
        return f"QuantumTask(id={self.task_id}, priority={self.priority}, status={self.status})"

class TaskStatus(Enum):
    """Enumeration for tracking the status of quantum tasks."""
    PENDING = auto()   # Task has been created but not yet started
    RUNNING = auto()   # Task is currently being executed
    COMPLETED = auto() # Task has finished execution successfully
    FAILED = auto()    # Task execution failed

class QuantumTaskFactory:
    """Factory class for creating various types of quantum tasks using specific quantum algorithms."""

    @staticmethod
    def create_grover_search_task(task_id: int, complexity: int, priority: int, n_qubits: int, target_state: int) -> QuantumTask:
        """
        Creates a Grover Search task.

        This method initializes a GroverSearch algorithm with the specified number of qubits and target state,
        then creates and returns a QuantumTask instance for this algorithm.

        Args:
            task_id (int): A unique identifier for the task.
            complexity (int): A numerical representation of the task's complexity, influencing execution time.
            priority (int): An integer representing the task's priority level, affecting its execution order.
            n_qubits (int): The number of qubits to use in the Grover Search algorithm.
            target_state (int): The target state that the Grover Search algorithm will search for.

        Returns:
            QuantumTask: An instance of QuantumTask configured to execute a Grover Search.
        """
        algorithm = GroverSearch(n_qubits, target_state)  # Initialize the GroverSearch algorithm
        return QuantumTask(task_id, complexity, priority, algorithm)  # Return a configured QuantumTask instance

    @staticmethod
    def create_qft_task(task_id: int, complexity: int, priority: int, n_qubits: int) -> QuantumTask:
        """
        Creates a Quantum Fourier Transform (QFT) task.

        This method initializes a QuantumFourierTransform algorithm with the specified number of qubits,
        then creates and returns a QuantumTask instance for this algorithm.

        Args:
            task_id (int): A unique identifier for the task.
            complexity (int): A numerical representation of the task's complexity, influencing execution time.
            priority (int): An integer representing the task's priority level, affecting its execution order.
            n_qubits (int): The number of qubits to use in the Quantum Fourier Transform algorithm.

        Returns:
            QuantumTask: An instance of QuantumTask configured to execute a Quantum Fourier Transform.
        """
        algorithm = QuantumFourierTransform(n_qubits)  # Initialize the Quantum Fourier Transform algorithm
        return QuantumTask(task_id, complexity, priority, algorithm)  # Return a configured QuantumTask instance

    # Additional factory methods can be implemented here for creating other quantum algorithms as needed.
    # Each method should follow the same pattern: instantiate the algorithm, then create and return a QuantumTask.


class TaskQueue:
    """Priority queue for managing quantum tasks, enabling efficient task scheduling based on priority."""

    def __init__(self):
        """
        Initializes a TaskQueue instance.

        This constructor sets up an empty priority queue to hold quantum tasks, allowing for organized task 
        management based on their priority levels.
        """
        self.queue = PriorityQueue()  # Create a priority queue to manage tasks

    def add_task(self, task: QuantumTask):
        """
        Adds a quantum task to the queue.

        The task is inserted into the priority queue with its priority negated to ensure that higher priority
        tasks are retrieved first.

        Args:
            task (QuantumTask): The quantum task to be added to the queue.
        """
        self.queue.put((-task.priority, task))  # Negate priority to sort higher priorities first

    def get_task(self) -> QuantumTask:
        """
        Retrieves and removes the highest priority task from the queue.

        This method retrieves the next task to be executed based on its priority, ensuring efficient task management.

        Returns:
            QuantumTask: The highest priority task currently in the queue.
        """
        return self.queue.get()[1]  # Get the task associated with the highest priority

    def is_empty(self) -> bool:
        """
        Checks if the task queue is empty.

        Returns:
            bool: True if the queue contains no tasks, False otherwise.
        """
        return self.queue.empty()  # Return whether the queue is empty

    def size(self) -> int:
        """
        Returns the number of tasks currently in the queue.

        Returns:
            int: The number of tasks in the queue.
        """
        return self.queue.qsize()  # Return the size of the task queue

class Worker(threading.Thread):
    """Worker thread for executing quantum tasks from a shared task queue."""

    def __init__(self, name: str, task_queue: TaskQueue, resource_manager: 'QuantumResourceManager'):
        """
        Initializes a new Worker instance.

        This constructor sets up a worker thread with a given name, a reference to a task queue from which it will
        retrieve tasks, and a resource manager for handling quantum resources needed for task execution.

        Args:
            name (str): The name of the worker thread, useful for logging and identification.
            task_queue (TaskQueue): The queue from which the worker will fetch tasks to execute.
            resource_manager (QuantumResourceManager): A manager for allocating and deallocating resources needed by the tasks.
        """
        threading.Thread.__init__(self)  # Initialize the base threading class
        self.name = name  # Assign the worker's name
        self.task_queue = task_queue  # Assign the shared task queue
        self.resource_manager = resource_manager  # Assign the resource manager

    def run(self):
        """
        The main execution loop for the worker thread.

        This method continuously checks the task queue for tasks to execute. If a task is available, it attempts to
        allocate resources using the resource manager. If successful, the task is executed. Upon completion, the worker
        logs the result, deallocates resources, and manages task failure scenarios. If resources are unavailable, the
        task is re-queued for later execution.

        The loop continues until there are no more tasks in the queue.
        """
        while not self.task_queue.is_empty():  # Continue until the task queue is empty
            task = self.task_queue.get_task()  # Retrieve the next task from the queue
            resource_id = f"resource-{task.task_id}"  # Create a unique resource identifier for the task

            # Attempt to allocate the required resource for executing the task
            if self.resource_manager.allocate_resource(resource_id):
                logger.info(f"{self.name} started executing Task {task.task_id}")  # Log task start
                task.status = TaskStatus.RUNNING  # Update task status to RUNNING

                # Execute the task and check the result
                if task.execute():
                    logger.info(f"{self.name} completed Task {task.task_id} with result: {task.result}")  # Log successful execution
                else:
                    logger.warning(f"{self.name} failed to execute Task {task.task_id}")  # Log execution failure
                    self.task_queue.add_task(task)  # Re-add the task to the queue for retry

                # Deallocate the resource after execution
                self.resource_manager.deallocate_resource(resource_id)
            else:
                logger.info(f"{self.name} couldn't execute Task {task.task_id} due to resource unavailability")  # Log resource unavailability
                self.task_queue.add_task(task)  # Re-add the task to the queue for later execution

class QuantumResourceManager:
    """Manages quantum computing resources, ensuring efficient allocation and deallocation."""

    def __init__(self, total_resources: int):
        """
        Initializes the QuantumResourceManager.

        This constructor sets the total number of resources available for quantum tasks and maintains the state of
        allocated and available resources, ensuring thread safety through a locking mechanism.

        Args:
            total_resources (int): The total number of quantum resources available for allocation.
        """
        self.total_resources = total_resources  # Total resources available for allocation
        self.available_resources = total_resources  # Current count of available resources
        self.allocated_resources: Dict[str, bool] = {}  # Dictionary to track allocated resources by resource ID
        self.lock = threading.Lock()  # Lock for thread-safe resource management

    def allocate_resource(self, resource_id: str) -> bool:
        """
        Allocates a quantum resource to a task.

        This method attempts to allocate a resource if one is available and not already allocated. It decrements
        the count of available resources and marks the resource as allocated.

        Args:
            resource_id (str): The unique identifier for the resource being allocated.

        Returns:
            bool: True if allocation is successful, False otherwise.
        """
        with self.lock:  # Ensure thread-safe access to shared resource data
            if self.available_resources > 0 and resource_id not in self.allocated_resources:
                self.available_resources -= 1  # Decrease available resources count
                self.allocated_resources[resource_id] = True  # Mark resource as allocated
                return True  # Successful allocation
            return False  # Allocation failed (either no resources available or resource already allocated)

    def deallocate_resource(self, resource_id: str):
        """
        Deallocates a quantum resource, making it available for future tasks.

        This method checks if the resource is currently allocated. If it is, the resource is removed from the
        allocation tracking, and the available resources count is incremented.

        Args:
            resource_id (str): The unique identifier for the resource being deallocated.
        """
        with self.lock:  # Ensure thread-safe access to shared resource data
            if resource_id in self.allocated_resources:
                del self.allocated_resources[resource_id]  # Remove resource from allocation tracking
                self.available_resources += 1  # Increase available resources count

    def get_resource_usage(self) -> Dict[str, int]:
        """
        Retrieves current resource usage statistics.

        This method returns a dictionary containing total, used, and available resources, allowing for monitoring
        of resource consumption.

        Returns:
            Dict[str, int]: A dictionary with keys 'total', 'used', and 'available' indicating resource counts.
        """
        with self.lock:  # Ensure thread-safe access to shared resource data
            return {
                "total": self.total_resources,
                "used": self.total_resources - self.available_resources,  # Calculate used resources
                "available": self.available_resources  # Current count of available resources
            }


class QuantumHyperThreading:
    """Manages quantum hyper-threading for executing multiple quantum tasks concurrently."""

    def __init__(self, num_threads: int, total_resources: int):
        """
        Initializes the QuantumHyperThreading manager.

        This constructor sets up a hyper-threading environment by initializing the number of threads, the task queue,
        a list for managing tasks, and a resource manager for handling quantum resources.

        Args:
            num_threads (int): The number of worker threads to be created for task execution.
            total_resources (int): The total number of quantum resources available for allocation.
        """
        self.num_threads = num_threads  # Number of concurrent threads for task execution
        self.task_queue = TaskQueue()  # Instantiate a task queue to manage quantum tasks
        self.tasks: List[QuantumTask] = []  # List to hold all created quantum tasks
        self.resource_manager = QuantumResourceManager(total_resources)  # Instantiate resource manager

    def create_tasks(self, num_tasks: int):
        """
        Generates a specified number of quantum tasks.

        This method creates tasks with random complexity, priority, and qubit counts. Tasks are randomly assigned
        to be either Grover's search or Quantum Fourier Transform tasks and are added to the task queue for execution.

        Args:
            num_tasks (int): The number of quantum tasks to create and add to the queue.
        """
        for i in range(num_tasks):
            complexity = random.randint(1, 5)  # Random complexity level between 1 and 5
            priority = random.randint(1, 10)  # Random priority level between 1 and 10
            n_qubits = random.randint(2, 5)  # Random number of qubits between 2 and 5
            
            # Randomly decide the type of quantum task to create
            if random.choice([True, False]):
                target_state = random.randint(0, 2**n_qubits - 1)  # Random target state for Grover's task
                task = QuantumTaskFactory.create_grover_search_task(i, complexity, priority, n_qubits, target_state)
            else:
                task = QuantumTaskFactory.create_qft_task(i, complexity, priority, n_qubits)  # Create QFT task
            
            self.tasks.append(task)  # Add the task to the local task list
            self.task_queue.add_task(task)  # Add the task to the shared task queue for execution

    def add_dependencies(self, task_id: int, dependencies: List[int]):
        """
        Adds dependencies to a specified task.

        This method links a task to other tasks that must be completed before it can execute, enabling the creation of
        complex task execution flows.

        Args:
            task_id (int): The ID of the task to which dependencies will be added.
            dependencies (List[int]): A list of task IDs representing the dependencies for the specified task.
        """
        task = self.tasks[task_id]  # Retrieve the task based on its ID
        for dep_id in dependencies:
            task.add_dependency(self.tasks[dep_id])  # Add each dependency to the task

    def execute_tasks(self):
        """
        Initiates the execution of tasks using worker threads.

        This method starts multiple worker threads, each capable of executing tasks concurrently. It logs the start
        of execution and waits for all threads to complete their assigned tasks.

        The number of worker threads created is based on the specified `num_threads` during initialization.
        """
        logger.info(f"Starting execution with {self.num_threads} quantum threads.")  # Log execution start
        workers = []  # List to hold all worker threads
        for i in range(self.num_threads):
            worker = Worker(name=f"Worker-{i+1}", task_queue=self.task_queue, resource_manager=self.resource_manager)
            workers.append(worker)  # Add the worker to the list
            worker.start()  # Start the worker thread

        for worker in workers:
            worker.join()  # Wait for all workers to finish execution

    def adjust_task_priorities(self):
        """
        Adjusts the priorities of all tasks based on their execution metrics.

        This method recalculates the priorities of tasks to optimize execution order. Higher-priority tasks are executed
        first, and priority adjustments are logged for monitoring.

        The method sorts the tasks based on their current priority and updates the priorities so that no task has a
        priority less than 1.
        """
        for task in self.tasks:
            task.adjust_priority()  # Adjust each task's priority based on its metrics
        self.tasks.sort()  # Sort tasks based on the updated priorities
        for i, task in enumerate(self.tasks):
            task.priority = max(1, task.priority - i)  # Ensure priority is at least 1

    def monitor_performance(self):
        """
        Monitors and logs the performance of all tasks.

        This method calculates and logs the average execution times for each task, providing insights into performance
        metrics that can inform future optimizations.

        It logs execution times and computes average times to help identify any bottlenecks in the task execution process.
        """
        for task in self.tasks:
            avg_time = sum(task.execution_times) / len(task.execution_times) if task.execution_times else 0  # Calculate average execution time
            logger.info(f"Task {task.task_id}: Execution times: {task.execution_times}, Average time: {avg_time:.2f}, Duration: {task.execution_duration():.2f} seconds")

class AdaptiveScheduler:
    """Implements adaptive scheduling for quantum tasks to optimize execution based on historical performance and task dynamics."""

    def __init__(self, tasks: List[QuantumTask]):
        """
        Initializes the AdaptiveScheduler with a list of quantum tasks.
        
        Args:
            tasks (List[QuantumTask]): A list of tasks to be managed by the scheduler.
        """
        self.tasks = tasks

    def prioritize_based_on_history(self):
        """
        Adjusts the priority of tasks based on their historical performance metrics.
        
        This method iterates over each task, allowing it to modify its priority according to predefined criteria
        (e.g., execution time, failure rates). After adjustments, tasks are sorted based on their new priority
        levels, ensuring that the most critical tasks are executed first.
        """
        for task in self.tasks:
            task.adjust_priority()  # Calls the method in QuantumTask to modify priority based on historical data
        self.tasks.sort()  # Sort tasks based on adjusted priorities

    def dynamic_rescheduling(self):
        """
        Performs dynamic rescheduling of tasks based on stochastic factors.
        
        This method introduces an element of randomness in task complexity and priority adjustment. Each task has
        a 20% chance of being rescheduled, allowing the system to adapt to changing workloads or priorities 
        dynamically. This approach can help balance the load and optimize resource usage in scenarios where
        tasks exhibit varying levels of complexity and urgency.
        """
        for task in self.tasks:
            if random.random() < 0.2:  # 20% probability for rescheduling
                task.complexity = random.randint(1, 5)  # Randomly adjust the task complexity
                task.adjust_priority()  # Reassess priority based on the new complexity

class PerformanceLogger:
    """Logs performance metrics for quantum tasks to facilitate monitoring and analysis."""

    def __init__(self):
        """
        Initializes the PerformanceLogger with an empty list for logs.
        
        This list will store log messages with timestamps, allowing for structured record-keeping of task
        execution metrics and overall system performance.
        """
        self.logs: List[str] = []

    def log(self, message: str):
        """
        Records a log message with a timestamp.
        
        Args:
            message (str): The message to log, typically containing performance metrics or status updates.
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())  # Generate a formatted timestamp
        self.logs.append(f"{timestamp} - {message}")  # Append the timestamped message to the logs

    def display_logs(self):
        """Outputs all logged messages to the console for immediate review."""
        for log in self.logs:
            print(log)  # Print each log entry

    def save_logs(self, filename: str):
        """
        Saves all logged messages to a specified file.
        
        Args:
            filename (str): The name of the file to which logs will be saved.
        
        This method ensures that log data can be persisted for future reference or analysis, facilitating
        post-execution reviews and audits of quantum task performance.
        """
        with open(filename, 'w') as f:
            for log in self.logs:
                f.write(f"{log}\n")  # Write each log entry to the file

class TaskMonitor:
    """Monitors the status of quantum tasks to provide real-time tracking and management capabilities."""

    def __init__(self):
        """
        Initializes the TaskMonitor with a dictionary to track the states of tasks.
        
        The task_states dictionary maps each task's unique identifier to its current status, 
        enabling efficient state management and retrieval for ongoing monitoring of quantum task execution.
        """
        self.task_states: Dict[int, TaskStatus] = {}

    def update_task_state(self, task_id: int, state: TaskStatus):
        """
        Updates the state of a specific task.

        Args:
            task_id (int): The unique identifier of the task whose state is being updated.
            state (TaskStatus): The new state of the task, represented by an instance of TaskStatus.

        This method ensures that the monitor accurately reflects the current status of each task,
        which is crucial for understanding the system's operational state.
        """
        self.task_states[task_id] = state  # Update the state of the specified task

    def get_task_state(self, task_id: int) -> TaskStatus:
        """
        Retrieves the current state of a specific task.

        Args:
            task_id (int): The unique identifier of the task for which to retrieve the state.

        Returns:
            TaskStatus: The current state of the task, or PENDING if the task ID is not found.

        This method facilitates querying task states, allowing other components of the system
        to respond dynamically based on the current execution status.
        """
        return self.task_states.get(task_id, TaskStatus.PENDING)  # Default to PENDING if task_id not found

    def display_all_states(self):
        """Outputs the states of all monitored tasks to the console for easy visibility."""
        for task_id, state in self.task_states.items():
            print(f"Task {task_id}: {state.name}")  # Print each task's ID along with its current state

    def get_task_summary(self) -> Dict[TaskStatus, int]:
        """
        Generates a summary of the current states of all tasks.

        Returns:
            Dict[TaskStatus, int]: A dictionary mapping each TaskStatus to the number of tasks in that state.

        This method provides a high-level overview of the distribution of task states,
        which can be useful for assessing system performance and bottlenecks.
        """
        summary = {status: 0 for status in TaskStatus}  # Initialize summary dictionary
        for state in self.task_states.values():
            summary[state] += 1  # Increment the count for the current state
        return summary  # Return the summary of task states

class TaskDependencyGraph:
    """Represents the dependency graph of quantum tasks to manage and visualize task interdependencies."""

    def __init__(self):
        """
        Initializes the TaskDependencyGraph as a directed graph (DiGraph).

        The directed nature of the graph allows for the representation of dependencies 
        where one task must complete before another can start, facilitating effective
        management of task execution order in quantum computing workflows.
        """
        self.graph = nx.DiGraph()  # Create a directed graph to represent task dependencies

    def add_task(self, task: QuantumTask):
        """
        Adds a task to the dependency graph along with its dependencies.

        Args:
            task (QuantumTask): The quantum task to be added to the graph.

        This method establishes the task as a node in the graph and creates directed edges
        to its dependent tasks, enabling the graph to accurately represent the execution order
        required for quantum tasks with dependencies.
        """
        self.graph.add_node(task.task_id, task=task)  # Add the task as a node
        for dep in task.dependencies:
            self.graph.add_edge(dep.task_id, task.task_id)  # Create edges for each dependency

    def display_graph(self):
        """Visualizes the task dependency graph using NetworkX and Matplotlib for clarity."""
        pos = nx.spring_layout(self.graph)  # Calculate positions for nodes in the graph
        nx.draw(self.graph, pos, with_labels=True, node_color='lightblue', node_size=500, 
                font_size=10, font_weight='bold')  # Draw the graph with specified attributes
        nx.draw_networkx_labels(self.graph, pos, {node: f"Task {node}" for node in self.graph.nodes()})  # Label the nodes
        plt.title("Task Dependency Graph")  # Set the title of the plot
        plt.axis('off')  # Turn off the axis
        plt.show()  # Display the graph

    def get_critical_path(self) -> List[int]:
        """
        Calculates and returns the critical path in the dependency graph.

        Returns:
            List[int]: A list of task IDs representing the longest path through the graph,
            which corresponds to the sequence of tasks that dictate the overall project completion time.

        Identifying the critical path is essential for optimizing task scheduling and
        ensuring that dependencies are respected during execution.
        """
        return nx.dag_longest_path(self.graph)  # Return the longest path in the directed acyclic graph

class QuantumCircuitOptimizer:
    """Optimizes quantum circuits by reducing gate count and depth for improved performance."""

    @staticmethod
    def optimize_circuit(circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Optimizes a given quantum circuit by implementing various circuit reduction techniques.

        Args:
            circuit (QuantumCircuit): The quantum circuit to be optimized.

        Returns:
            QuantumCircuit: A new optimized quantum circuit with reduced gate count and depth.

        This method applies optimization strategies that are crucial for enhancing the efficiency 
        of quantum algorithms. The optimizations aim to minimize the number of gates executed, 
        thereby reducing potential errors and improving circuit execution time.
        
        Optimization Techniques:
        1. Merge Adjacent Gates: Combines consecutive gates acting on the same qubit(s) into a single gate 
           when possible, thereby reducing overall gate count.
        2. Cancel Out Redundant Gates: Identifies and eliminates gates that have no net effect on the qubits, 
           such as consecutive NOT gates.
        3. Apply Gate Identity Rules: Utilizes known mathematical identities of quantum gates to simplify the circuit.
        """
        optimized_circuit = QuantumCircuit(circuit.num_qubits)  # Initialize an optimized circuit with the same number of qubits
        
        # Implement circuit optimization techniques here
        # Example: Merging adjacent gates, cancelling redundant gates, applying identities

        return optimized_circuit  # Return the optimized quantum circuit


class QuantumErrorCorrection:
    """Implements quantum error correction codes to protect quantum information from errors."""

    @staticmethod
    def apply_bit_flip_code(circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Applies a 3-qubit bit flip error correction code to the provided quantum circuit.

        Args:
            circuit (QuantumCircuit): The quantum circuit for which error correction is to be applied.

        Returns:
            QuantumCircuit: A new quantum circuit incorporating the bit flip error correction code.

        This method encodes each logical qubit into three physical qubits, providing redundancy 
        that allows the circuit to detect and correct single-bit errors. The 3-qubit bit flip code
        is one of the simplest forms of quantum error correction.

        Steps for Bit Flip Error Correction:
        1. **Encoding**: Each logical qubit is encoded into three physical qubits using CNOT gates.
           This process spreads the information of a single logical qubit across three qubits to provide redundancy.
        2. **Operation Application**: The original operations from the circuit are applied to the encoded qubits.
           Each gate is applied to the corresponding physical qubit representation.
        3. **Decoding and Error Correction**: The circuit applies additional CNOT gates to check for errors
           and corrects them, ensuring that the logical qubit remains intact despite potential errors in the physical qubits.
        """
        corrected_circuit = QuantumCircuit(circuit.num_qubits * 3)  # Initialize a corrected circuit with three times the qubits
        
        # Encode logical qubit into 3 physical qubits
        for i in range(0, circuit.num_qubits * 3, 3):
            corrected_circuit.add_gate(QuantumGate.CNOT, i, i+1)  # Create entanglement for redundancy
            corrected_circuit.add_gate(QuantumGate.CNOT, i, i+2)  # Create entanglement for redundancy
        
        # Apply original circuit operations on encoded qubits
        for gate, target, control in circuit.gates:
            if control is not None:
                corrected_circuit.add_gate(gate, target*3, control*3)  # Apply gate with control
            else:
                corrected_circuit.add_gate(gate, target*3)  # Apply gate without control
        
        # Decode and correct errors
        for i in range(0, circuit.num_qubits * 3, 3):
            corrected_circuit.add_gate(QuantumGate.CNOT, i, i+1)  # Perform checks to detect errors
            corrected_circuit.add_gate(QuantumGate.CNOT, i, i+2)  # Perform checks to detect errors
            corrected_circuit.add_gate(QuantumGate.CNOT, i+1, i)  # Correct errors if detected
            corrected_circuit.add_gate(QuantumGate.CNOT, i+2, i)  # Correct errors if detected
        
        return corrected_circuit  # Return the quantum circuit with applied error correction

class QuantumSimulationMetrics:
    """Collects and analyzes metrics from quantum simulations to evaluate performance."""

    def __init__(self):
        """
        Initializes the metrics collector with empty lists for execution times, 
        success rates, and resource utilization metrics.

        Attributes:
            execution_times (List[float]): Stores the execution times of quantum tasks.
            success_rates (List[float]): Stores the success rates of quantum operations.
            resource_utilization (List[float]): Stores the resource utilization percentages.
        """
        self.execution_times: List[float] = []  # List to track time taken for each quantum simulation task
        self.success_rates: List[float] = []     # List to track the success rates of completed tasks
        self.resource_utilization: List[float] = []  # List to track how efficiently resources are used

    def add_execution_time(self, time: float):
        """Adds a new execution time measurement to the metrics.

        Args:
            time (float): The time taken for a quantum task in seconds.
        """
        self.execution_times.append(time)

    def add_success_rate(self, rate: float):
        """Records the success rate of a quantum operation.

        Args:
            rate (float): The success rate as a decimal (e.g., 0.85 for 85%).
        """
        self.success_rates.append(rate)

    def add_resource_utilization(self, utilization: float):
        """Records the resource utilization percentage for a quantum task.

        Args:
            utilization (float): The percentage of resources used (0 to 100).
        """
        self.resource_utilization.append(utilization)

    def get_average_execution_time(self) -> float:
        """Calculates the average execution time from recorded metrics.

        Returns:
            float: The average execution time in seconds.
        """
        return sum(self.execution_times) / len(self.execution_times) if self.execution_times else 0

    def get_average_success_rate(self) -> float:
        """Calculates the average success rate from recorded metrics.

        Returns:
            float: The average success rate as a decimal (0 to 1).
        """
        return sum(self.success_rates) / len(self.success_rates) if self.success_rates else 0

    def get_average_resource_utilization(self) -> float:
        """Calculates the average resource utilization from recorded metrics.

        Returns:
            float: The average resource utilization as a percentage (0 to 100).
        """
        return sum(self.resource_utilization) / len(self.resource_utilization) if self.resource_utilization else 0

    def plot_metrics(self):
        """Visualizes the collected metrics using line plots for analysis."""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

        # Plot execution times
        ax1.plot(self.execution_times, marker='o', linestyle='-')
        ax1.set_title("Execution Times")
        ax1.set_xlabel("Task Index")
        ax1.set_ylabel("Time (s)")
        ax1.grid(True)

        # Plot success rates
        ax2.plot(self.success_rates, marker='o', linestyle='-', color='g')
        ax2.set_title("Success Rates")
        ax2.set_xlabel("Task Index")
        ax2.set_ylabel("Success Rate")
        ax2.grid(True)

        # Plot resource utilization
        ax3.plot(self.resource_utilization, marker='o', linestyle='-', color='orange')
        ax3.set_title("Resource Utilization")
        ax3.set_xlabel("Task Index")
        ax3.set_ylabel("Utilization (%)")
        ax3.grid(True)

        plt.tight_layout()  # Adjusts the subplots to fit into the figure area
        plt.show()  # Displays the plotted metrics


class QuantumNoiseModel:
    """Simulates various types of quantum noise to analyze the effects on quantum states."""

    @staticmethod
    def apply_depolarizing_noise(state: QuantumState, probability: float):
        """Applies depolarizing noise to the quantum state to simulate errors.

        Args:
            state (QuantumState): The quantum state to which noise is applied.
            probability (float): The probability of applying noise to each qubit (0 to 1).
        
        Depolarizing noise randomly replaces the actual state with a completely mixed state 
        based on the specified probability, simulating the impact of environmental interference.
        Each qubit has a chance of being subjected to a Pauli gate (X, Y, or Z), introducing 
        random flips or phase shifts.
        """
        for i in range(state.num_qubits):
            if random.random() < probability:  # Determine if noise should be applied
                # Randomly apply one of the Pauli gates (X, Y, or Z) to the qubit
                gate = random.choice([QuantumGate.PAULI_X, QuantumGate.PAULI_Y, QuantumGate.PAULI_Z])
                state.apply_gate(gate, i)

    @staticmethod
    def apply_amplitude_damping(state: QuantumState, gamma: float):
        """Applies amplitude damping noise to the quantum state.

        Args:
            state (QuantumState): The quantum state to which amplitude damping is applied.
            gamma (float): The probability of amplitude damping occurring (0 to 1).

        Amplitude damping models the energy loss of a quantum state, particularly relevant in 
        physical systems where qubits lose energy to the environment. This method reduces the 
        amplitude of the excited state while increasing the amplitude of the ground state, 
        simulating decoherence. The normalization step ensures that the quantum state remains 
        valid after damping.
        """
        for i in range(state.num_qubits):
            if random.random() < gamma:  # Determine if amplitude damping should be applied
                # Apply amplitude damping to the state vector
                state.state[1::2] *= np.sqrt(1 - gamma)  # Dampen the excited state
                state.state[::2] += np.sqrt(gamma) * state.state[1::2]  # Increase the ground state
                state.state /= np.linalg.norm(state.state)  # Normalize the state vector to maintain valid probabilities

    @staticmethod
    def apply_phase_flip(state: QuantumState, probability: float):
        """Applies phase flip noise to the quantum state.

        Args:
            state (QuantumState): The quantum state to which phase flip noise is applied.
            probability (float): The probability of applying phase flip to each qubit (0 to 1).

        Phase flip noise introduces a phase shift to the quantum state, simulating errors due 
        to decoherence. This can significantly impact quantum algorithms that rely on superposition 
        and interference effects. The phase flip operation is applied to each qubit with the 
        specified probability.
        """
        for i in range(state.num_qubits):
            if random.random() < probability:  # Determine if phase flip should be applied
                state.apply_gate(QuantumGate.PAULI_Z, i)  # Apply Z gate for phase flip

class QuantumCircuitVisualizer:
    """Visualizes quantum circuits using various representations."""

    @staticmethod
    def draw_circuit(circuit: QuantumCircuit):
        """
        Draw the quantum circuit using matplotlib.

        This method creates a visual representation of the quantum circuit by plotting its gates and qubits. 
        Each qubit is represented as a horizontal line, and gates are illustrated as symbols at different time steps 
        along the lines representing the qubits they act upon.

        Parameters:
            circuit (QuantumCircuit): The quantum circuit to visualize, which contains gates and qubits.
        """
        fig, ax = plt.subplots(figsize=(12, 6))  # Create a figure and axis for plotting.
        ax.set_ylim(-0.5, circuit.num_qubits - 0.5)  # Set y-axis limits to match the number of qubits.
        ax.set_xlim(-0.5, len(circuit.gates) + 0.5)  # Set x-axis limits to match the number of gates.
        ax.set_yticks(range(circuit.num_qubits))  # Set y-ticks to correspond to qubit indices.
        ax.set_yticklabels([f'Q{i}' for i in range(circuit.num_qubits)])  # Label y-ticks as qubit names.
        ax.set_xticks(range(len(circuit.gates)))  # Set x-ticks to correspond to gate indices.
        ax.set_xticklabels([f'Step {i}' for i in range(len(circuit.gates))])  # Label x-ticks as gate steps.
        ax.grid(True)  # Enable grid for better readability.

        # Iterate through the gates and plot them on the circuit.
        for i, (gate, target, control) in enumerate(circuit.gates):
            if gate == QuantumGate.CNOT:  # Special case for the CNOT gate.
                ax.plot([i, i], [control, target], 'k-', linewidth=2)  # Draw the control-target connection line.
                ax.plot(i, control, 'ko', markersize=10)  # Mark the control qubit.
                ax.plot(i, target, 'k^', markersize=10)  # Mark the target qubit with a triangle.
            else:
                ax.text(i, target, gate.name, ha='center', va='center', 
                        bbox=dict(facecolor='white', edgecolor='black'))  # Label other gates with their names.

        plt.title("Quantum Circuit Visualization")  # Set the title of the plot.
        plt.tight_layout()  # Adjust the layout for better spacing.
        plt.show()  # Display the plot.

    @staticmethod
    def print_circuit_text(circuit: QuantumCircuit):
        """
        Print a text representation of the quantum circuit.

        This method generates a textual visualization of the quantum circuit. It organizes qubit operations in rows 
        corresponding to qubits and columns corresponding to time steps, making it easier to see the sequence of gates 
        applied to each qubit. Special symbols are used to represent different gates.

        Parameters:
            circuit (QuantumCircuit): The quantum circuit to print, which contains gates and qubits.
        """
        # Create a list of strings to represent the circuit in text form, initialized with dashes for visual separation.
        lines = ['-' * (4 * len(circuit.gates) + 1) for _ in range(2 * circuit.num_qubits + 1)]
        
        # Iterate through the gates to construct the textual representation.
        for i, (gate, target, control) in enumerate(circuit.gates):
            col = 4 * i + 2  # Determine the column index for the current gate.
            if gate == QuantumGate.CNOT:  # Special handling for the CNOT gate.
                lines[2 * control + 1] = lines[2 * control + 1][:col] + '•' + lines[2 * control + 1][col + 1:]  # Mark control.
                # Draw a vertical line connecting control to target.
                for row in range(2 * min(control, target) + 2, 2 * max(control, target) + 1):
                    lines[row] = lines[row][:col] + '│' + lines[row][col + 1:]  # Indicate connection.
                lines[2 * target + 1] = lines[2 * target + 1][:col] + '⊕' + lines[2 * target + 1][col + 1:]  # Mark target.
            else:  # For other gates.
                lines[2 * target + 1] = lines[2 * target + 1][:col - 1] + f'-{gate.name[0]}-' + lines[2 * target + 1][col + 2:]  # Place the gate symbol.

        # Print the constructed text representation of the circuit line by line.
        for line in lines:
            print(line)

class QuantumStateVisualizer:
    """Visualizes quantum states using various representations."""

    @staticmethod
    def plot_probability_distribution(state: QuantumState):
        """
        Plot the probability distribution of the quantum state.

        This method generates a bar graph that illustrates the probability of each basis state in the quantum state.
        It retrieves the probabilities associated with the quantum state, which represent the likelihood of measuring 
        the system in each corresponding basis state.

        Parameters:
            state (QuantumState): The quantum state object containing the state vector, from which probabilities are derived.
        """
        probabilities = state.get_probabilities()  # Extract probabilities from the quantum state.
        fig, ax = plt.subplots()  # Create a figure and axis for plotting.
        ax.bar(range(len(probabilities)), probabilities)  # Create a bar plot for the probability distribution.
        ax.set_xlabel("Basis State")  # Label the x-axis.
        ax.set_ylabel("Probability")  # Label the y-axis.
        ax.set_title("Quantum State Probability Distribution")  # Set the title of the plot.
        plt.show()  # Display the plot.

    @staticmethod
    def plot_bloch_sphere(state: QuantumState):
        """
        Plot the Bloch sphere representation of a single-qubit state.

        The Bloch sphere is a geometrical representation of a qubit, where each point on the sphere corresponds to a 
        possible state of the qubit. This method specifically visualizes single-qubit states in three-dimensional space.

        Parameters:
            state (QuantumState): The quantum state object representing a single qubit.

        Raises:
            ValueError: If the quantum state represents more than one qubit, as the Bloch sphere representation is 
            only valid for single-qubit states.
        """
        if state.num_qubits != 1:
            raise ValueError("Bloch sphere representation is only valid for single-qubit states.")

        # Calculate the spherical coordinates (theta, phi) from the state vector.
        theta = 2 * np.arccos(np.abs(state.state[0]))  # Polar angle, derived from the amplitude of the |0> state.
        phi = np.angle(state.state[1]) - np.angle(state.state[0])  # Azimuthal angle, derived from the relative phase.

        fig = plt.figure()  # Create a new figure for the Bloch sphere.
        ax = fig.add_subplot(111, projection='3d')  # Add a 3D subplot for the Bloch sphere visualization.

        # Create a mesh grid for drawing the sphere's surface.
        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        x = np.cos(u) * np.sin(v)  # X-coordinates of the sphere.
        y = np.sin(u) * np.sin(v)  # Y-coordinates of the sphere.
        z = np.cos(v)  # Z-coordinates of the sphere.
        ax.plot_wireframe(x, y, z, color="gray", alpha=0.2)  # Draw the surface of the sphere.

        # Plot the state vector on the Bloch sphere.
        ax.quiver(0, 0, 0, np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta),
                  color='r', arrow_length_ratio=0.1)  # Arrow represents the state on the Bloch sphere.

        # Set the axes labels and title.
        ax.set_xlabel("X")  # Label for the X-axis.
        ax.set_ylabel("Y")  # Label for the Y-axis.
        ax.set_zlabel("Z")  # Label for the Z-axis.
        ax.set_title("Bloch Sphere Representation")  # Title for the plot.
        plt.show()  # Display the Bloch sphere plot.

class QuantumAlgorithmBenchmark:
    """Benchmarks different quantum algorithms."""

    def __init__(self, num_qubits: int, num_iterations: int):
        """
        Initialize the benchmark with a specified number of qubits and iterations.

        Parameters:
            num_qubits (int): The number of qubits used in the quantum algorithms being benchmarked.
            num_iterations (int): The number of times each algorithm will be executed to gather timing data.
        """
        self.num_qubits = num_qubits  # Store the number of qubits.
        self.num_iterations = num_iterations  # Store the number of iterations for the benchmark.
        self.results: Dict[str, List[float]] = {}  # Dictionary to hold the execution times for each algorithm.

    def run_benchmark(self, algorithm: QuantumAlgorithm, name: str):
        """
        Execute the specified quantum algorithm multiple times and record execution times.

        This method runs the given quantum algorithm a specified number of times, measuring the execution time for 
        each run and storing the results in the `results` dictionary. The benchmarking helps in evaluating the 
        performance of different quantum algorithms.

        Parameters:
            algorithm (QuantumAlgorithm): The quantum algorithm instance to be benchmarked.
            name (str): A descriptive name for the algorithm being executed, used as a key in the results dictionary.
        """
        execution_times = []  # List to hold the execution times for this algorithm.
        for _ in range(self.num_iterations):
            start_time = time.time()  # Record the start time of the execution.
            algorithm.run()  # Execute the quantum algorithm.
            end_time = time.time()  # Record the end time of the execution.
            execution_times.append(end_time - start_time)  # Calculate and store the execution time.
        self.results[name] = execution_times  # Store the execution times under the algorithm's name.

    def plot_results(self):
        """
        Visualize the benchmark results using a boxplot.

        This method generates a boxplot that displays the distribution of execution times for each algorithm,
        allowing for easy comparison of performance metrics. The plot helps to identify trends, outliers, and 
        the overall efficiency of different quantum algorithms.

        The boxplot provides insights into the variability and median execution times across multiple iterations.
        """
        fig, ax = plt.subplots()  # Create a new figure and axis for the plot.
        ax.boxplot(self.results.values())  # Create a boxplot of the recorded execution times.
        ax.set_xticklabels(self.results.keys())  # Set the x-tick labels to the algorithm names.
        ax.set_ylabel("Execution Time (s)")  # Label for the y-axis indicating the time in seconds.
        ax.set_title(f"Quantum Algorithm Benchmark ({self.num_qubits} qubits, {self.num_iterations} iterations)")  # Set the title of the plot.
        plt.show()  # Display the boxplot.


class QuantumCircuitCompiler:
    """Compiles quantum circuits for specific hardware architectures."""

    def __init__(self, connectivity_graph: nx.Graph):
        """
        Initialize the quantum circuit compiler with a specified connectivity graph.

        Parameters:
            connectivity_graph (nx.Graph): A graph representing the qubit connectivity for the hardware,
            where nodes represent qubits and edges represent direct interaction capabilities.
        """
        self.connectivity_graph = connectivity_graph  # Store the connectivity graph for the compiler.

    def compile(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Compile a quantum circuit to fit the constraints of a specific hardware architecture.

        This method transforms the input quantum circuit to account for the connectivity constraints of the target 
        hardware. It ensures that quantum gates are applied only between qubits that are directly connected in the 
        underlying architecture. If a gate operation cannot be performed directly due to connectivity restrictions,
        a series of SWAP gates are introduced to rearrange the qubits accordingly.

        Parameters:
            circuit (QuantumCircuit): The original quantum circuit that needs to be compiled.

        Returns:
            QuantumCircuit: A new quantum circuit that respects the connectivity constraints of the hardware.
        """
        compiled_circuit = QuantumCircuit(circuit.num_qubits)  # Initialize a new circuit with the same number of qubits.
        
        for gate, target, control in circuit.gates:  # Iterate over all gates in the original circuit.
            if gate == QuantumGate.CNOT:  # Check if the gate is a CNOT gate.
                if self.connectivity_graph.has_edge(control, target):
                    compiled_circuit.add_gate(gate, target, control)  # Add the gate directly if connected.
                else:
                    # Implement a SWAP network to bring qubits together if they are not directly connected.
                    path = nx.shortest_path(self.connectivity_graph, source=control, target=target)  # Find the shortest path.
                    for i in range(len(path) - 1):
                        compiled_circuit.add_gate(QuantumGate.SWAP, path[i], path[i + 1])  # Add SWAP gates along the path.
                    compiled_circuit.add_gate(gate, path[-1], path[-2])  # Add the CNOT gate at the final position.
                    for i in range(len(path) - 1, 0, -1):
                        compiled_circuit.add_gate(QuantumGate.SWAP, path[i], path[i - 1])  # Reverse the SWAP gates.
            else:
                compiled_circuit.add_gate(gate, target)  # For non-CNOT gates, add directly.

        return compiled_circuit  # Return the compiled circuit that fits the hardware constraints.

class QuantumCircuitDecomposer:
    """Decomposes complex quantum gates into basic gate sets."""

    @staticmethod
    def decompose_toffoli(control1: int, control2: int, target: int) -> List[Tuple[QuantumGate, int, int]]:
        """
        Decompose a Toffoli (CCNOT) gate into a sequence of CNOT and single-qubit gates.

        The Toffoli gate, which applies a NOT operation on the target qubit if both control qubits are in the |1⟩ state,
        cannot be directly implemented on some quantum hardware due to gate restrictions. This method translates the 
        Toffoli gate into a sequence of more basic gates that can be executed in a quantum circuit.

        The decomposition consists of:
        - Applying a Hadamard gate to the target qubit.
        - Using CNOT gates to create entanglement between the control qubits and the target.
        - Applying T (π/8) and its adjoint gates to facilitate the controlled flipping of the target qubit.

        Parameters:
            control1 (int): The index of the first control qubit.
            control2 (int): The index of the second control qubit.
            target (int): The index of the target qubit to be flipped.

        Returns:
            List[Tuple[QuantumGate, int, int]]: A list of tuples, where each tuple contains a quantum gate type and its associated qubit indices.
        """
        return [
            (QuantumGate.HADAMARD, target, None),  # Apply Hadamard to create superposition on the target.
            (QuantumGate.CNOT, target, control2),  # Entangle target with control2.
            (QuantumGate.T, target, None),  # Apply T gate to target.
            (QuantumGate.CNOT, target, control1),  # Entangle target with control1.
            (QuantumGate.T, control2, None),  # Apply T gate to control2.
            (QuantumGate.CNOT, control2, control1),  # Entangle control2 and control1.
            (QuantumGate.T, target, None),  # Apply T gate to target again.
            (QuantumGate.T, control1, None),  # Apply T gate to control1.
            (QuantumGate.CNOT, target, control2),  # Entangle target with control2 again.
            (QuantumGate.CNOT, control2, control1),  # Entangle control2 with control1.
            (QuantumGate.T, control2, None),  # Apply T gate to control2 once more.
            (QuantumGate.CNOT, target, control1),  # Final entanglement of target with control1.
            (QuantumGate.HADAMARD, target, None)  # Undo Hadamard to finalize the operation.
        ]

    @staticmethod
    def decompose_controlled_rotation(control: int, target: int, angle: float) -> List[Tuple[QuantumGate, int, int]]:
        """
        Decompose a controlled rotation gate into CNOT and single-qubit rotations.

        Controlled rotation gates (such as Rz) perform a rotation on the target qubit conditional on the state of the control qubit.
        This method breaks down a controlled rotation into basic gates that are more widely supported in quantum hardware, allowing 
        for greater flexibility and compatibility with different architectures.

        The decomposition consists of:
        - Applying a Phase gate (Rz(angle/2)) to the target.
        - Using a CNOT gate to transfer control from the control qubit to the target.
        - Applying another Phase gate (Rz(-angle/2)) to revert the first rotation.
        - Using a second CNOT gate to complete the controlled rotation effect.

        Parameters:
            control (int): The index of the control qubit.
            target (int): The index of the target qubit that will undergo rotation.
            angle (float): The angle of rotation for the controlled operation.

        Returns:
            List[Tuple[QuantumGate, int, int]]: A list of tuples, where each tuple contains a quantum gate type and its associated qubit indices.
        """
        return [
            (QuantumGate.PHASE, target, None),  # Apply a phase shift corresponding to half the rotation angle.
            (QuantumGate.CNOT, target, control),  # Entangle control with the target qubit.
            (QuantumGate.PHASE, target, None),  # Apply the negative phase shift to undo the first half-rotation.
            (QuantumGate.CNOT, target, control)  # Final CNOT to ensure the rotation is applied conditionally.
        ]

class QuantumCircuitSimulator:
    """Simulates the execution of quantum circuits."""

    @staticmethod
    def simulate(circuit: QuantumCircuit, num_shots: int = 1000) -> Dict[str, int]:
        """
        Simulates the execution of a quantum circuit over a specified number of shots.

        This method models the probabilistic nature of quantum mechanics by simulating the measurement outcomes 
        of a quantum circuit defined by its gates. For each shot, the state of the qubits is evolved according to 
        the specified gates, followed by a measurement to extract the quantum state into classical outcomes.

        The simulation uses the following process:
        - Initialize a quantum state representation for the specified number of qubits.
        - Apply each gate in the circuit sequentially to modify the quantum state.
        - Perform a measurement on the quantum state to collapse it into a classical binary outcome.
        - Count the frequency of each measurement outcome over the specified number of shots.

        Parameters:
            circuit (QuantumCircuit): The quantum circuit to be simulated, containing a list of gates to apply.
            num_shots (int, optional): The number of times the circuit is executed to gather statistics on measurement outcomes. Defaults to 1000.

        Returns:
            Dict[str, int]: A dictionary where the keys are binary strings representing measurement outcomes, 
            and the values are the counts of each outcome across all shots.
        """
        results = {}
        for _ in range(num_shots):
            # Initialize the quantum state for the simulation
            state = QuantumState("SimulationState", circuit.num_qubits)
            # Apply each gate in the quantum circuit to evolve the state
            for gate, target, control in circuit.gates:
                state.apply_gate(gate, target, control)
            # Measure the final state to get the outcome
            measurement = state.measure()
            # Convert the measurement result into a binary string
            binary_result = format(measurement, f'0{circuit.num_qubits}b')
            # Record the occurrence of each measurement outcome
            results[binary_result] = results.get(binary_result, 0) + 1
        return results

    @staticmethod
    def plot_simulation_results(results: Dict[str, int]):
        """
        Visualizes the results of the quantum circuit simulation.

        This method creates a bar plot to display the frequency of each measurement outcome from the simulation. 
        The x-axis represents the different measurement outcomes (in binary form), while the y-axis indicates the 
        number of occurrences of each outcome. This visualization provides insight into the probabilistic behavior 
        of the quantum circuit and helps in analyzing its performance and characteristics.

        Parameters:
            results (Dict[str, int]): A dictionary of measurement outcomes and their corresponding frequencies 
            obtained from the simulation.

        Returns:
            None: The function displays a plot and does not return any values.
        """
        fig, ax = plt.subplots()
        # Create a bar chart to represent measurement frequencies
        ax.bar(results.keys(), results.values())
        ax.set_xlabel("Measurement Outcome")  # Label for the x-axis
        ax.set_ylabel("Frequency")  # Label for the y-axis
        ax.set_title("Quantum Circuit Simulation Results")  # Title of the plot
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
        plt.tight_layout()  # Adjust layout to prevent clipping of tick-labels
        plt.show()  # Display the plot

class QuantumResourceEstimator:
    """Estimates resource requirements for quantum circuits."""

    @staticmethod
    def estimate_resources(circuit: QuantumCircuit) -> Dict[str, int]:
        """
        Estimates the resource requirements for a given quantum circuit.

        This method analyzes the structure of a quantum circuit to estimate its resource requirements,
        which are essential for evaluating the feasibility of implementing the circuit on a quantum device. 
        The estimation includes the number of qubits used, the circuit depth (the longest path from the 
        input to output qubits), and the total number of gates, categorized into single-qubit and 
        two-qubit gates.

        The process includes:
        - Counting the total number of gates in the circuit.
        - Classifying the gates into single-qubit and two-qubit gates.
        - Calculating the depth of the circuit by tracking the number of operations affecting each qubit 
          and determining the maximum depth of operations required for any qubit.

        Parameters:
            circuit (QuantumCircuit): The quantum circuit whose resources are to be estimated.

        Returns:
            Dict[str, int]: A dictionary containing estimates of various resource requirements:
            - 'num_qubits': The total number of qubits in the circuit.
            - 'circuit_depth': The maximum depth of the circuit, representing the longest sequence of 
              operations affecting any single qubit.
            - 'total_gates': The total number of gates in the circuit.
            - 'single_qubit_gates': The count of single-qubit gates in the circuit.
            - 'two_qubit_gates': The count of two-qubit gates in the circuit.
        """
        resources = {
            "num_qubits": circuit.num_qubits,  # Total number of qubits utilized in the circuit
            "circuit_depth": 0,  # Initialize circuit depth to zero
            "total_gates": len(circuit.gates),  # Total gates are the length of the gates list
            "single_qubit_gates": 0,  # Count of single-qubit gates
            "two_qubit_gates": 0  # Count of two-qubit gates
        }

        # Array to track the depth of operations affecting each qubit
        depth = [0] * circuit.num_qubits
        for gate, target, control in circuit.gates:
            if control is None:
                # If there's no control qubit, it's a single-qubit gate
                resources["single_qubit_gates"] += 1
                depth[target] += 1  # Increment depth for the target qubit
            else:
                # If there is a control qubit, it's a two-qubit gate
                resources["two_qubit_gates"] += 1
                # Update the depth based on the control qubit's current depth
                depth[target] = max(depth[target], depth[control]) + 1
                depth[control] = depth[target]  # Set control qubit depth to the new target depth

        # The overall circuit depth is the maximum depth across all qubits
        resources["circuit_depth"] = max(depth)
        return resources

class QuantumErrorMitigation:
    """Implements error mitigation techniques for quantum circuits."""

    @staticmethod
    def richardson_extrapolation(circuit: QuantumCircuit, noise_factors: List[float]) -> QuantumCircuit:
        """
        Apply Richardson extrapolation for error mitigation.

        Richardson extrapolation is a technique used to reduce the effect of noise in quantum circuits 
        by combining multiple circuits executed at varying levels of noise. This method exploits the 
        relationship between the noise factor and the expected output to construct a more accurate 
        estimation of the true result by extrapolating towards zero noise.

        The method works by stretching the gates in the circuit according to specified noise factors. 
        For each noise factor, a new circuit is created where each gate is repeated according to 
        the noise factor. These stretched circuits are then combined with appropriate coefficients 
        to mitigate the overall noise.

        Parameters:
            circuit (QuantumCircuit): The quantum circuit to mitigate noise.
            noise_factors (List[float]): A list of noise factors for extrapolation.

        Returns:
            QuantumCircuit: A new quantum circuit that incorporates error mitigation through 
            Richardson extrapolation.
        """
        mitigated_circuit = QuantumCircuit(circuit.num_qubits)  # Initialize the mitigated circuit

        for noise_factor in noise_factors:
            stretched_circuit = QuantumCircuit(circuit.num_qubits)  # Create a new circuit for each noise factor
            for gate, target, control in circuit.gates:
                # Stretch each gate by the noise factor
                for _ in range(int(noise_factor)):
                    stretched_circuit.add_gate(gate, target, control)  # Add the gate to the stretched circuit
            
            # Combine the stretched circuits with appropriate coefficients
            coefficient = 1 / (noise_factor * (noise_factor - 1))  # Calculate the coefficient for extrapolation
            mitigated_circuit.gates.extend([(gate, target, control, coefficient) for gate, target, control in stretched_circuit.gates])
        
        return mitigated_circuit  # Return the final mitigated circuit

    @staticmethod
    def zero_noise_extrapolation(circuit: QuantumCircuit, noise_levels: List[float]) -> Callable:
        """
        Apply zero-noise extrapolation for error mitigation.

        Zero-noise extrapolation is an error mitigation strategy that allows for the correction of 
        noise in quantum computations by estimating what the result would be if there were no noise. 
        This technique involves executing the quantum circuit at multiple noise levels and then 
        fitting a polynomial to the results. The coefficients of this polynomial can then be used 
        to extrapolate the results back to zero noise.

        This method creates a closure that captures the quantum circuit and noise levels, allowing 
        for execution on a specified quantum computer. The results are collected and a polynomial 
        fit is applied to extrapolate the expected outcome at zero noise.

        Parameters:
            circuit (QuantumCircuit): The quantum circuit to mitigate noise.
            noise_levels (List[float]): A list of noise levels to apply to the circuit.

        Returns:
            Callable: A function that executes the mitigated circuit on a specified quantum computer 
            and returns the extrapolated result at zero noise.
        """
        def mitigated_execution(quantum_computer):
            results = []
            for noise_level in noise_levels:
                # Apply depolarizing noise to the circuit at the specified noise level
                noisy_circuit = QuantumNoiseModel.apply_depolarizing_noise(circuit, noise_level)
                results.append(quantum_computer.run(noisy_circuit))  # Execute the noisy circuit and collect results
            
            # Perform extrapolation to zero noise using polynomial fitting
            coefficients = np.polyfit(noise_levels, results, len(noise_levels) - 1)  # Fit polynomial to results
            return np.poly1d(coefficients)(0)  # Evaluate the polynomial at zero noise
        
        return mitigated_execution  # Return the execution function for use with a quantum computer

class QuantumCircuitOptimizer:
    """Optimizes quantum circuits to reduce gate count and depth."""

    @staticmethod
    def merge_adjacent_gates(circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Merge adjacent single-qubit gates when possible.

        This method scans through the gates of the provided quantum circuit 
        and attempts to optimize the circuit by merging adjacent single-qubit 
        gates that can be combined. The primary goal of this optimization 
        is to reduce the overall gate count and depth of the circuit, which 
        can lead to improved execution times and reduced error rates in 
        quantum hardware.

        When two single-qubit gates are adjacent and act on the same target qubit,
        this method checks if they are compatible for merging. If they are, 
        the gates are combined into a single gate where possible, 
        following the specific merging rules defined in the `_merge_gates` method.

        Parameters:
            circuit (QuantumCircuit): The quantum circuit containing the gates to optimize.

        Returns:
            QuantumCircuit: A new quantum circuit with adjacent gates merged where applicable.
        """
        optimized_gates = []  # List to store optimized gates
        last_gate = None  # Keep track of the last gate for potential merging

        for gate, target, control in circuit.gates:
            if control is None:  # Check for single-qubit gates
                if last_gate and last_gate[1] == target:
                    # Attempt to merge gates if they operate on the same qubit
                    merged_gate = QuantumCircuitOptimizer._merge_gates(last_gate[0], gate)
                    if merged_gate:
                        optimized_gates.pop()  # Remove the last gate, which is now merged
                        optimized_gates.append((merged_gate, target, None))  # Add the merged gate
                    else:
                        optimized_gates.append((gate, target, None))  # No merging, keep current gate
                else:
                    optimized_gates.append((gate, target, None))  # Just append the gate
            else:
                optimized_gates.append((gate, target, control))  # Keep controlled gates unchanged
            
            last_gate = (gate, target, control)  # Update last gate for the next iteration

        optimized_circuit = QuantumCircuit(circuit.num_qubits)  # Create a new circuit for optimized gates
        optimized_circuit.gates = optimized_gates  # Assign the optimized gates to the new circuit
        return optimized_circuit  # Return the optimized circuit

    @staticmethod
    def _merge_gates(gate1: QuantumGate, gate2: QuantumGate) -> QuantumGate:
        """
        Merge compatible gates, returning a combined gate or None if not mergeable.

        This method defines specific rules for merging two quantum gates. 
        The merging process is crucial for optimizing quantum circuits, 
        as certain combinations of gates can effectively cancel out or simplify.

        For example, two consecutive Pauli-X gates (NOT operations) 
        cancel each other out, resulting in no operation (NOOP). 
        Similarly, phase gates can be combined under specific conditions 
        to reduce the complexity of the circuit.

        Parameters:
            gate1 (QuantumGate): The first gate to merge.
            gate2 (QuantumGate): The second gate to merge.

        Returns:
            QuantumGate: The resulting merged gate if the gates are compatible, 
            or None if they cannot be merged.
        """
        # Define rules for merging common gates, such as Pauli gates or phase gates
        if gate1 == gate2:  # Check if both gates are of the same type
            if gate1 == QuantumGate.PAULI_X:
                return None  # Pauli-X twice cancels out (X^2 = I)
            elif gate1 == QuantumGate.PHASE:
                return QuantumGate.PHASE  # Double phase could be simplified (PHASE + PHASE = 2*PHASE)
        return None  # Return None for gates that cannot be merged

def main():
    """
    Main function to demonstrate the execution of a quantum hyper-threading environment 
    and the benchmarking of quantum algorithms.

    This function sets up a simulated quantum environment with multiple tasks, dependencies, 
    and scheduling mechanisms to effectively manage and execute quantum computations 
    in parallel. It also benchmarks quantum algorithms and visualizes the execution process.
    """

    num_qubits = 4  # Number of qubits in the quantum circuit
    num_threads = 4  # Number of threads for task execution
    total_resources = 5  # Total resources available for task execution
    num_tasks = 10  # Total number of tasks to create
    num_iterations = 5  # Number of iterations for benchmarking quantum algorithms

    # Initialize Quantum Hyper-Threading Environment
    quantum_hyper_threading = QuantumHyperThreading(num_threads, total_resources)
    quantum_hyper_threading.create_tasks(num_tasks)  # Create a specified number of tasks for execution

    # Add dependencies (example: task 3 depends on task 1 and task 2)
    quantum_hyper_threading.add_dependencies(task_id=3, dependencies=[1, 2])  
    # This sets up a task dependency where task 3 cannot start until tasks 1 and 2 are completed.

    # Execute tasks in a multi-threaded environment
    quantum_hyper_threading.execute_tasks()  # Launch tasks concurrently across available threads

    # Monitor performance of the tasks being executed
    quantum_hyper_threading.monitor_performance()  # Track metrics like execution time and resource usage

    # Adjust task priorities based on current performance and resource availability
    quantum_hyper_threading.adjust_task_priorities()  # Optimizes task execution order for efficiency

    # Initialize Adaptive Scheduler to manage task execution based on historical performance
    adaptive_scheduler = AdaptiveScheduler(quantum_hyper_threading.tasks)
    adaptive_scheduler.prioritize_based_on_history()  # Prioritize tasks that have historically performed well
    adaptive_scheduler.dynamic_rescheduling()  # Reschedule tasks dynamically as execution progresses

    # Task Dependency Graph to visualize task relationships and dependencies
    task_graph = TaskDependencyGraph()
    for task in quantum_hyper_threading.tasks:
        task_graph.add_task(task)  # Add each task to the dependency graph for visualization and analysis
    task_graph.display_graph()  # Display the task dependency graph for better understanding of task interrelations

    # Get critical path in the task dependency graph
    critical_path = task_graph.get_critical_path()  
    # The critical path identifies the sequence of tasks that determine the minimum completion time
    print("Critical Path:", critical_path)  # Output the critical path for performance evaluation

    # Run Quantum Algorithm Benchmarking
    benchmark = QuantumAlgorithmBenchmark(num_qubits, num_iterations)
    grover_search = GroverSearch(num_qubits, target_state=1)  # Initialize Grover's search algorithm
    qft = QuantumFourierTransform(num_qubits)  # Initialize Quantum Fourier Transform algorithm
    benchmark.run_benchmark(grover_search, "Grover's Search")  # Benchmark Grover's algorithm
    benchmark.run_benchmark(qft, "Quantum Fourier Transform")  # Benchmark Quantum Fourier Transform
    benchmark.plot_results()  # Visualize benchmarking results to analyze algorithm performance

    # Visualization examples
    sample_circuit = QuantumCircuit(num_qubits)  # Create a sample quantum circuit for demonstration
    sample_circuit.add_gate(QuantumGate.HADAMARD, 0)  # Apply a Hadamard gate to the first qubit
    sample_circuit.add_gate(QuantumGate.CNOT, 1, 0)  # Apply a CNOT gate, with qubit 1 as target and qubit 0 as control
    QuantumCircuitVisualizer.draw_circuit(sample_circuit)  # Visualize the constructed quantum circuit

    # Display task states for monitoring and analysis
    task_monitor = TaskMonitor()  
    for task in quantum_hyper_threading.tasks:
        task_monitor.update_task_state(task.task_id, task.status)  # Update each task's current state in the monitor
    task_monitor.display_all_states()  # Display the state of all tasks for an overview of progress

    # Log and save performance metrics for future analysis
    logger = PerformanceLogger()
    logger.log("Performance metrics logged successfully.")  # Log performance metrics to the logger
    logger.display_logs()  # Display logged metrics for review
    logger.save_logs("performance_logs.txt")  # Save logs to a specified file for persistence

    # Simulate circuit execution and plot results for visualization
    simulation_results = QuantumCircuitSimulator.simulate(sample_circuit)  # Run simulation of the quantum circuit
    QuantumCircuitSimulator.plot_simulation_results(simulation_results)  # Plot the results of the simulation for analysis

if __name__ == "__main__":
    main()  # Execute the main function when the script is run
