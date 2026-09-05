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

## 命名慣例

後續每日課程使用：

```text
lessons/dayNN_topic_name.py
```

每個檔案可獨立執行，並印出量子電路、statevector、機率或量測結果。
