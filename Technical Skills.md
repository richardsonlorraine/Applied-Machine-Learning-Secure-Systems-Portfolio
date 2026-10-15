Technical Skills

Deep Learning Frameworks & Architectures:

* Proficiency in PyTorch (torch>=2.0.0) for tensor manipulation, autograd, and custom model definitions.
* Design of custom 1D Convolutional Neural Networks featuring residual blocks (ResidualBlock1D), Batch Normalization, and Adaptive Average Pooling.
* Implementation of 2D Convolutional Autoencoders utilizing encoder-decoder symmetric structures with transposed convolutions and Sigmoid bounds.
* Construction of lightweight Multi-Layer Perceptrons (MLPs) paired with automated integer quantization strategies (INT8 and INT32) for hardware synthesis.

Digital Signal Processing & RF Engineering:

* Generation, manipulation, and buffering of raw In-phase/Quadrature (I/Q) baseband sample streams.
* Implementation of Short-Time Fourier Transform (STFT) pipelines via SciPy to convert time-domain signals into 2D time-frequency energy spectrograms.
* Fast Fourier Transform (FFT) and Inverse Fast Fourier Transform (IFFT) operations for multi-carrier OFDM transceivers.
* Simulation of multi-path Rayleigh fading channel impulse responses, Additive White Gaussian Noise (AWGN) injection across diverse decibel ranges, and pilot symbol insertions.

Hardware Design & Embedded Systems:

* VHDL design adhering to IEEE standards (93/2008) for synchronous state machines and datapath pipelines.
* Implementation of high-throughput AXI-Stream slave and master handshaking control signals (s_axis_tvalid, s_axis_tready, s_axis_tdata).
* Cycle-accurate pipelined Multiply-Accumulate (MAC) arithmetic units targeting low-resource FPGA fabrics such as Xilinx/AMD Zynq and Artix-7.
* Compilation and execution of hardware simulations using open-source toolchains like GHDL and ModelSim.

Software Tools & Libraries:

* Python (>=3.10) core development alongside numerical and scientific libraries including NumPy (>=1.22.0), SciPy (>=1.8.0), Matplotlib (>=3.5.0), and Seaborn (>=0.11.0).
* Configuration management and parameter handling via PyYAML (>=6.0).
* Hardware integration pipelines connecting software inference buffers with RTL-SDR and ADALM-PLUTO software-defined radios via GNU Radio or MATLAB environments.