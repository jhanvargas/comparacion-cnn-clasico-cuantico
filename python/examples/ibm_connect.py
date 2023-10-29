from qiskit import IBMQ, QuantumCircuit, execute

# Own libraries
from python.metadata.path import Path
from python.utils.readers import read_yaml


def run_ibm_nairobi() -> None:
    """Runs a quantum circuit on the IBM Quantum backend ibm_nairobi.
    """
    config = read_yaml(Path.ibm_config)

    # Autenticación con IBM Quantum
    IBMQ.save_account(config['ibm'], overwrite=True)
    IBMQ.load_account()

    # Conexión al backend ibm_nairobi
    provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
    backend = provider.get_backend(config['name'])

    # Creación de circuito cuántico
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])

    # Ejecución del circuito en el backend ibm_nairobi
    job = execute(qc, backend=backend, shots=1024)
    result = job.result()

    # Obtención de los resultados
    counts = result.get_counts(qc)
    print(counts)
