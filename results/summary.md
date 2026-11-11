| task | metric | model | mean ± std | seeds |
|---|---|---|---|---|
| henon | nrmse | Echo state network | 0.0037 ± 0.0003 | 5 |
| henon | nrmse | Linear AR (10 lags) | 0.8535 ± 0.0000 | 5 |
| henon | nrmse | LSTM | 0.0194 ± 0.0111 | 5 |
| henon | nrmse | Spiking reservoir (pulse) | 0.5864 ± 0.0070 | 5 |
| henon | nrmse | Spiking reservoir (DC) | 0.2990 ± 0.0240 | 5 |
| henon | nrmse | Spiking + STDP (pulse) | 0.6650 ± 0.0163 | 5 |
| mackey_glass | nrmse | Echo state network | 0.0021 ± 0.0002 | 5 |
| mackey_glass | nrmse | Linear AR (10 lags) | 0.1077 ± 0.0000 | 5 |
| mackey_glass | nrmse | LSTM | 0.0267 ± 0.0094 | 5 |
| mackey_glass | nrmse | Spiking reservoir (pulse) | 0.3712 ± 0.0137 | 5 |
| mackey_glass | nrmse | Spiking reservoir (DC) | 0.3219 ± 0.0133 | 5 |
| mackey_glass | nrmse | Spiking + STDP (pulse) | 0.3760 ± 0.0145 | 5 |
| memory | memory_capacity | Echo state network | 10.1370 ± 0.6307 | 5 |
| memory | memory_capacity | Spiking reservoir (pulse) | 0.5663 ± 0.0941 | 5 |
| narma10 | nrmse | Echo state network | 0.4701 ± 0.0298 | 5 |
| narma10 | nrmse | Linear AR (10 lags) | 0.5921 ± 0.0224 | 5 |
| narma10 | nrmse | LSTM | 0.2317 ± 0.0332 | 5 |
| narma10 | nrmse | Spiking reservoir (pulse) | 0.8778 ± 0.0197 | 5 |
| narma10 | nrmse | Spiking reservoir (DC) | 0.8745 ± 0.0107 | 5 |
| narma10 | nrmse | Spiking + STDP (pulse) | 0.8654 ± 0.0112 | 5 |
| vowels | accuracy | Echo state network | 0.9811 ± 0.0019 | 5 |
| vowels | accuracy | LogReg, mean+std | 0.9611 ± 0.0024 | 5 |
| vowels | accuracy | LogReg, resampled | 0.9914 ± 0.0012 | 5 |
| vowels | accuracy | LSTM | 0.9762 ± 0.0048 | 5 |
| vowels | accuracy | Spiking reservoir (pulse) | 0.9654 ± 0.0059 | 5 |
| vowels | accuracy | Spiking reservoir (DC) | 0.9735 ± 0.0132 | 5 |
| vowels | accuracy | Spiking + STDP (pulse) | 0.9703 ± 0.0097 | 5 |
