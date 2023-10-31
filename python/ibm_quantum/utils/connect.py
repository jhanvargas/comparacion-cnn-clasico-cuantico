# External libraries
from typing import Union
from qiskit import IBMQ
from qiskit.providers.ibmq import IBMQBackend

# Own libraries
from python.metadata.path import Path
from python.utils.readers import read_yaml


def get_ibm_quantum() -> Union[IBMQBackend, None]:
    """Conecta y devuelve una máquina cuántica de IBM Quantum.

    Returns:
        IBMQBackend: Objeto de la máquina cuántica conectada.

        Nombre: ibmq_qasm_simulator
        Status: active
        Número de qubits: 32
        Simulador: True
        --------------------
        Nombre: simulator_statevector
        Status: active
        Número de qubits: 32
        Simulador: True
        --------------------
        Nombre: simulator_mps
        Status: active
        Número de qubits: 100
        Simulador: True
        --------------------
        Nombre: simulator_extended_stabilizer
        Status: active
        Número de qubits: 63
        Simulador: True
        --------------------
        Nombre: simulator_stabilizer
        Status: active
        Número de qubits: 5000
        Simulador: True
        --------------------
        Nombre: ibm_lagos
        Status: active
        Número de qubits: 7
        Simulador: False
        --------------------
        Nombre: ibm_nairobi
        Status: active
        Número de qubits: 7
        Simulador: False
        --------------------
        Nombre: ibm_perth
        Status: active
        Número de qubits: 7
        Simulador: False
        --------------------
        Nombre: ibm_brisbane
        Status: active
        Número de qubits: 127
        Simulador: False
        --------------------

    """
    config = read_yaml(Path.ibm_config)

    try:
        IBMQ.save_account(config['ibm'], overwrite=True)
        IBMQ.load_account()

        # Conexión al backend
        provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
        backend = provider.get_backend(config['name'])

        return backend

    except Exception as e:
        print(f"Error al conectarse a IBM Quantum: {str(e)}")
        return None
