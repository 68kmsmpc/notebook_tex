import numpy as np
import matplotlib.pyplot as plt
import time, datetime
#import qiskit.quantum_info as qi



#from qiskit import Aer
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper, JordanWignerMapper, BravyiKitaevMapper
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.second_q.algorithms.ground_state_solvers import GroundStateEigensolver
from qiskit_nature import settings
from qiskit_algorithms.optimizers import  SPSA, COBYLA, L_BFGS_B
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit.circuit.library import TwoLocal
from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver, VQE
from qiskit.primitives import Estimator

from qiskit_ibm_runtime import QiskitRuntimeService,Session, Estimator, Options


service = QiskitRuntimeService(channel="ibm_quantum", token ='55ac54cb4cbfe378d8604606bacfcab38ffd165ecdb215742e64495a259fa53f0ed9e63df5b3df4489d422865a007d9cc981fcb2caffd9e76f17d94d734a8593')
options = Options(optimization_level=3, resilience_level = 2)


def reset_intermediate_info():
    global intermediate_info
    intermediate_info.clear()
    print(intermediate_info)
    intermediate_info = {
        'nfev ': [],
        'parameters ': [],
        'energy ': [],
        'stddev ': []
    }

def callback (nfev , parameters , energy , stddev ):
    intermediate_info ['nfev ']. append ( nfev )
    intermediate_info ['parameters ']. append ( parameters )
    intermediate_info ['energy ']. append ( energy )
    intermediate_info ['stddev ']. append ( stddev )

def exact_solver(qubit_op, problem):
    sol = NumPyMinimumEigensolver().compute_minimum_eigenvalue(qubit_op)
    result = problem.interpret(sol)
    return result

def fermion_to_qubit(problem, second_q_op, mapper_name,  z2symmetry_reduction=None):
  if mapper_name == "JW":
    mapper = JordanWignerMapper()
  if mapper_name == "Pa":
    mapper = ParityMapper(num_particles=problem.num_particles)
  if mapper_name == "BK":
    mapper = BravyiKitaevMapper()

  qubit_op = mapper.map(second_q_op)

  if z2symmetry_reduction != None:
    mapper = problem.get_tapered_mapper(mapper)
    qubit_op = mapper.map(second_q_op)
  else :
    qubit_op = mapper.map(second_q_op)

  return qubit_op , mapper

def run_vqe(name, problem, qubit_converter, solver, show=True):
  calc = GroundStateEigensolver(qubit_converter, solver)
  start = time.time()
  ground_state = calc.solve(problem)
  elapsed = str(datetime.timedelta(seconds = time.time()-start))
  if show:
    print("Running the VQE using the {}".format(name))
    print("Elapsed time: {} \n".format(elapsed))
    print(ground_state)
  return ground_state



def least_Energy(as_problem):
    as_num_particles = as_problem.num_particles
    as_num_spatial_orbitals = as_problem.num_spatial_orbitals
    as_fermionic_hamiltonian = as_problem.hamiltonian
    as_second_q_op = as_fermionic_hamiltonian.second_q_op()
    energy_arr = []
    ansatz_order = []
    opt_order=[]
    qubit_op, mapper = fermion_to_qubit(as_problem, as_second_q_op, "Pa", z2symmetry_reduction=None)
    as_init_state = HartreeFock(as_num_spatial_orbitals,as_num_particles,mapper)
    uccsd = UCCSD(as_num_spatial_orbitals,as_num_particles,mapper,initial_state=as_init_state, generalized=True)
    twolocal = TwoLocal(as_num_spatial_orbitals*2, ['ry', 'rz'], 'cz', initial_state=as_init_state)
    ansatzs=[uccsd,twolocal]
    iter = [250,1000]
    ansatzs_name=['UCCSD','Two local']
    
    opt_names = ['COBYLA','SPSA','L-BFGS-B'] 
    noiseless_estimator = Estimator()
    for i in range(2):
        ansatz = ansatzs[i]
        ansatz_name= ansatzs_name[i]
        MAX_ITER = iter[i]
        opt_arr = [COBYLA(maxiter=MAX_ITER),SPSA(maxiter=MAX_ITER),L_BFGS_B(maxiter=MAX_ITER)]
        for k in range(3):
            reset_intermediate_info()
            optimizer = opt_arr[k]
            opt_name = opt_names[k]

            with Session(service=service, backend="ibm_brisbane") as session:
              estimator = Estimator(session=session, options=options)
              vqe = VQE(estimator, ansatz, optimizer, callback = callback)
              result = vqe.compute_minimum_eigenvalue(qubit_op)
              vqe_result = result.eigenvalue.real #+shift
              print('result=',result)
              print('vqe_result=',vqe_result)
            # electronic_structure_result = as_problem.interpret(result)

            repulsion = as_problem.hamiltonian.constants['nuclear_repulsion_energy']
            exact_energy = exact_solver(qubit_op, as_problem).total_energies[0].real

            # convergence_energy = intermediate_info['energy '][-1]
            dimer_energy = vqe_result+repulsion
            Co_Li_energy = dimer_energy

            energy_arr.append(Co_Li_energy)
            ansatz_order.append(ansatz_name)
            opt_order.append(opt_name)
            
            plt.plot(range(len(intermediate_info['energy '])), intermediate_info['energy ']+repulsion)
            plt.axhline(exact_energy, color = 'r', linewidth = 1, label = 'exact energy')
            plt.title('{0},{1}'.format(ansatz_name , opt_name))
            plt.xlabel('VQE Iterations')
            plt.ylabel('Energy')
            plt.grid()
            plt.legend()
            plt.show()
    
    return energy_arr, ansatz_order, opt_order


intermediate_info = {
'nfev ': [],
'parameters ': [],
'energy ': [],
'stddev ': []
}


basis = 'STO-3G'
settings.use_pauli_sum_op = False


atoms = ["H", "H"]
dist = 0.735
coords = [(0,0,0), (dist,0,0)]
charge = 0
multiplicity = 1
Co_O_moleculeinfo = MoleculeInfo(atoms, coords, charge=charge, multiplicity=multiplicity)
driver = PySCFDriver.from_molecule(Co_O_moleculeinfo, basis=basis)
E_problem = driver.run() # 여기는 이후, As_transformer 로 변경. 
energy_arr, ansatz_order, opt_order = least_Energy(E_problem) 

print(energy_arr)

for k in range(len(energy_arr)):
    e=energy_arr[k]
    if e == np.min(energy_arr):
        print('optimal_calc :')
        print('anstaz: ', ansatz_order[k])
        print('optimizer: ', opt_order[k])
        print('Energy: ', e)
Co_O_2_dimer_energy=e



