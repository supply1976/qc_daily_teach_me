# Quantum Computing Daily Lessons

每日一小節量子運算 Python 練習，以繁體中文註解、Qiskit 與少量 NumPy 為主。

## 安裝

建議使用 Python virtual environment：

```bash
python -m venv .venv

# Linux / WSL
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 執行

```bash
python lessons/day01_qubit_superposition.py
```

所有範例均使用本地 statevector 模擬，不需要 IBM Quantum 帳號。

## 課程索引

| Day | 主題 | 程式 |
|---:|---|---|
| 01 | Qubit、疊加態與量測 | `lessons/day01_qubit_superposition.py` |
| 02 | 相位與量子干涉 | `lessons/day02_quantum_interference.py` |
| 03 | Bloch sphere 與單 qubit 旋轉 | `lessons/day03_bloch_sphere_rotations.py` |
| 04 | 兩個 qubit、張量積與位元順序 | `lessons/day04_tensor_product.py` |
| 05 | 量子糾纏與 Bell state | `lessons/day05_bell_entanglement.py` |
| 06 | 量測、投影與量子態塌縮 | `lessons/day06_measurement_collapse.py` |
| 07 | 密度矩陣、量子 coherence 與古典混合 | `lessons/day07_density_matrix.py` |
| 08 | 量子雜訊、Kraus operators 與 phase damping | `lessons/day08_phase_damping.py` |
| 09 | 單 qubit 量子態斷層掃描 | `lessons/day09_single_qubit_tomography.py` |
| 10 | 部分跡、約化密度矩陣與糾纏熵 | `lessons/day10_partial_trace_entanglement.py` |
| 11 | Schmidt decomposition、SVD 與糾纏強度 | `lessons/day11_schmidt_decomposition.py` |
| 12 | CHSH inequality、Bell nonlocality 與有限 shots | `lessons/day12_chsh_inequality.py` |
| 13 | 量子隱形傳態、條件分支與 fidelity | `lessons/day13_quantum_teleportation.py` |
| 14 | 量子超密編碼、Bell basis 與兩個 classical bits | `lessons/day14_superdense_coding.py` |
| 15 | Bernstein–Vazirani、phase kickback 與 hidden string | `lessons/day15_bernstein_vazirani.py` |
| 16 | 量子傅立葉轉換、相位梯度與 inverse QFT | `lessons/day16_quantum_fourier_transform.py` |
| 17 | 量子相位估計、controlled powers 與 eigenphase | `lessons/day17_quantum_phase_estimation.py` |
| 18 | QPE spectral measurement、eigenstate projection 與 entropy | `lessons/day18_qpe_spectral_measurement.py` |
| 19 | Hamiltonian time evolution、Larmor precession 與守恆量 | `lessons/day19_hamiltonian_time_evolution.py` |
| 20 | 非對易 Hamiltonian、Lie–Trotter 與精度—深度取捨 | `lessons/day20_trotterization.py` |
| 21 | 二階 Suzuki–Trotter、Ising dynamics 與糾纏熵 | `lessons/day21_suzuki_trotter_ising.py` |
| 22 | VQE、variational principle 與 parameter-shift gradient | `lessons/day22_vqe_parameter_shift.py` |
| 23 | 有限 shots、Pauli measurement grouping 與 VQE loss noise | `lessons/day23_finite_shot_vqe.py` |
| 24 | SPSA、simultaneous perturbation 與 noisy VQE optimization | `lessons/day24_spsa_noisy_vqe.py` |
| 25 | Barren plateau、global cost 與梯度方差尺度 | `lessons/day25_barren_plateau.py` |
| 26 | Quantum natural gradient、QFIM 與 Bloch-sphere 幾何 | `lessons/day26_quantum_natural_gradient.py` |
| 27 | Quantum Circuit Born Machine、KL loss 與相關分布生成 | `lessons/day27_quantum_circuit_born_machine.py` |
| 28 | Sample-based MMD、Hamming kernel 與三 qubit QCBM | `lessons/day28_mmd_qcbm.py` |
| 29 | Quantum fidelity kernel、Hamming 幾何與 SWAP test | `lessons/day29_quantum_fidelity_kernel.py` |
| 30 | Hadamard test、複數 expectation value 與 ancilla 干涉 | `lessons/day30_hadamard_test.py` |
| 31 | Hadamard-test autocorrelation、Fourier spectral function 與能譜 | `lessons/day31_spectral_function.py` |
| 32 | Quantum Krylov diagonalization、generalized eigenproblem 與 conditioning | `lessons/day32_quantum_krylov.py` |
| 33 | Krylov overlap regularization、canonical orthogonalization 與 noisy null space | `lessons/day33_krylov_regularization.py` |
| 34 | Imaginary-time ground-state filtering 與 variational QITE | `lessons/day34_imaginary_time_evolution.py` |
| 35 | Thermofield double、Gibbs thermal state 與 purification | `lessons/day35_thermofield_double.py` |
| 36 | Variational free-energy minimization、relative entropy 與 thermal-state learning | `lessons/day36_variational_free_energy.py` |
| 37 | Quantum Boltzmann machine、Ising energy model 與 correlated sampling | `lessons/day37_quantum_boltzmann_machine.py` |
| 38 | Transverse-field QBM、non-commuting Gibbs state 與 imaginary-time Trotterization | `lessons/day38_transverse_field_qbm.py` |
| 39 | Non-commuting QBM training、quantum moment matching 與 multi-basis identifiability | `lessons/day39_noncommuting_qbm_training.py` |

## 命名慣例

後續每日課程使用：

```text
lessons/dayNN_topic_name.py
```

每個檔案可獨立執行，並印出量子電路、statevector、機率或量測結果。
