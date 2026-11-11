"""Build and execute notebooks/tour.ipynb from the cells below."""

from pathlib import Path

import nbformat
from nbclient import NotebookClient

CELLS: list[tuple[str, str]] = [
    (
        "md",
        "# A tour of the organoid reservoir\n\nBuild a small simulated organoid, stimulate it "
        "through virtual electrodes, look at the spikes, and train a linear readout on the "
        "Japanese Vowels task. Everything runs in a few seconds on a laptop.",
    ),
    (
        "code",
        "%matplotlib inline\nimport matplotlib.pyplot as plt\nimport numpy as np\n\n"
        "from organoid_reservoir import data\n"
        "from organoid_reservoir.encoding import ElectrodeEncoder, EncoderConfig, pad_sequences\n"
        "from organoid_reservoir.features import pool_sequence_states\n"
        "from organoid_reservoir.readout import accuracy, train_logistic_readout\n"
        "from organoid_reservoir.reservoir import ReservoirConfig, SpikingReservoir\n\n"
        "plt.rcParams.update({'figure.dpi': 70, 'axes.spines.top': False, "
        "'axes.spines.right': False})",
    ),
    (
        "md",
        "## 1. Data\n\nNine speakers, 12 LPC cepstrum coefficients per frame, 7 to 29 frames "
        "per utterance. The loader downloads the files from UCI on first use.",
    ),
    (
        "code",
        "ds = data.load_japanese_vowels('../data/raw')\n"
        "print(len(ds.X_train), 'train /', len(ds.X_test), 'test utterances,', ds.n_classes, "
        "'speakers')\nprint('first utterance shape:', ds.X_train[0].shape)",
    ),
    (
        "md",
        "## 2. Electrodes and reservoir\n\nEach coefficient gets a push-pull pair of electrodes "
        "on the surface of a unit sphere. The reservoir is 200 Izhikevich neurons inside the "
        "sphere, wired with a probability that decays with distance.",
    ),
    (
        "code",
        "encoder = ElectrodeEncoder(ds.n_features, EncoderConfig())\n"
        "encoder.fit(np.concatenate(ds.X_train))\n"
        "reservoir = SpikingReservoir(ReservoirConfig(n_neurons=200), encoder, seed=0)\n"
        "print(encoder.n_electrodes, 'electrodes,', reservoir.n_neurons, 'neurons,', "
        "reservoir.n_exc, 'excitatory')\n"
        "print('mean in-degree:', reservoir.mask.sum(0).mean())\n"
        "print('spectral radius of W:', round(reservoir.spectral_radius(), 2))",
    ),
    (
        "code",
        "fig = plt.figure(figsize=(4.2, 4.2))\nax = fig.add_subplot(projection='3d')\n"
        "p, e = reservoir.positions, reservoir.electrode_positions\n"
        "ax.scatter(*p[reservoir.is_exc].T, s=8, color='#2a78d6', label='excitatory')\n"
        "ax.scatter(*p[~reservoir.is_exc].T, s=8, color='#eb6834', label='inhibitory')\n"
        "ax.scatter(*e.T, s=30, marker='s', color='#0b0b0b', label='electrodes')\n"
        "ax.set_axis_off(); ax.legend(fontsize=7, frameon=False); plt.show()",
    ),
    (
        "md",
        "## 3. Stimulate and record\n\nOne utterance becomes Poisson pulse trains on the "
        "electrodes. Each frame is held for 20 ms.",
    ),
    (
        "code",
        "X, lengths = pad_sequences([ds.X_train[0]])\nstim = encoder.intensities(X)\n"
        "out = reservoir.run(stim, record_spikes=True)\nspikes = out['spikes'][0]\n"
        "t, i = np.nonzero(spikes)\nfig, ax = plt.subplots(figsize=(6.5, 2.8))\n"
        "ax.scatter(t, i, s=1, color='#2a78d6')\n"
        "ax.set_xlabel('time (ms)'); ax.set_ylabel('neuron')\n"
        "plt.show()\nprint('mean rate: %.1f Hz' % (spikes.mean() * 1000))\n"
        "print('state tensor (batch, frames, neurons):', out['states'].shape)",
    ),
    (
        "md",
        "## 4. Train a linear readout\n\nOnly the readout is trained. Per-frame filtered spike "
        "rates are pooled over the utterance (overall mean plus three temporal segments) and "
        "fed to logistic regression.",
    ),
    (
        "code",
        "Xtr, ltr = pad_sequences(ds.X_train)\nXte, lte = pad_sequences(ds.X_test)\n"
        "Ftr = pool_sequence_states(reservoir.transform(Xtr, ltr), ltr)\n"
        "Fte = pool_sequence_states(reservoir.transform(Xte, lte), lte)\n"
        "readout = train_logistic_readout(Ftr, ds.y_train)\n"
        "print('feature matrix:', Ftr.shape)\n"
        "print('test accuracy: %.3f' % accuracy(ds.y_test, readout.predict(Fte)))",
    ),
    (
        "md",
        "## 5. Unsupervised STDP exposure\n\nReplaying the training utterances with plasticity "
        "switched on changes the excitatory weights without using any labels. The readout is "
        "then retrained on the frozen network.",
    ),
    (
        "code",
        "w0 = reservoir.W.copy()\nreservoir.transform(Xtr, ltr, plastic=True, batch_size=30)\n"
        "ne = reservoir.n_exc\nchange = np.linalg.norm(reservoir.W[:ne, :ne] - w0[:ne, :ne]) / "
        "np.linalg.norm(w0[:ne, :ne])\nprint('relative E->E weight change: %.2f' % change)\n"
        "Ftr = pool_sequence_states(reservoir.transform(Xtr, ltr), ltr)\n"
        "Fte = pool_sequence_states(reservoir.transform(Xte, lte), lte)\n"
        "readout = train_logistic_readout(Ftr, ds.y_train)\n"
        "print('test accuracy after STDP: %.3f' % accuracy(ds.y_test, readout.predict(Fte)))",
    ),
    (
        "md",
        "A single seed says little here: differences of this size are within seed-to-seed "
        "noise. See `report.md` for the five-seed comparison and the baselines.",
    ),
]


def main() -> None:
    nb = nbformat.v4.new_notebook()
    for kind, src in CELLS:
        cell = (
            nbformat.v4.new_markdown_cell(src) if kind == "md" else nbformat.v4.new_code_cell(src)
        )
        nb.cells.append(cell)
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    out = Path(__file__).resolve().parent.parent / "notebooks" / "tour.ipynb"
    NotebookClient(nb, timeout=600, resources={"metadata": {"path": str(out.parent)}}).execute()
    for cell in nb.cells:
        cell.pop("id", None)
    nbformat.write(nb, out)
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
