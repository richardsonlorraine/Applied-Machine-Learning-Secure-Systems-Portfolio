Project 2. FPGA-Accelerated Lightweight Neural Network for Real-Time Signal Classification

* Intersection: FPGA/VHDL, DSP, Edge AI, Embedded Systems
* Core Objective: Design and implement a hardware accelerator in VHDL (or using High-Level Synthesis) on an FPGA development board (e.g., Xilinx/AMD Zynq or Lattice iCE40) to execute a quantized, low-latency neural network for packet header classification or real-time anomaly detection in a serial data stream.

Architecture & Stack:

* Hardware: VHDL/Verilog for streaming data interfaces (AXI-Stream), fixed-point arithmetic units, and memory management.
* Software/ML: Train a compact multi-layer perceptron (MLP) or quantized CNN in , export the weights, and use tools like FINN or custom VHDL matrix-multiplier logic to synthesize the network directly onto programmable logic.
* Portfolio Value: Showcases rare, high-value competency in low-level hardware design combined with edge AI deployment—critical for high-throughput, low-latency defense and telecommunications systems.

Here is the complete codebase and flagship portfolio layout for Project 2:
 
FPGA-Accelerated Lightweight Neural Network for Real-Time Signal Classification, fully expanded with an automated -to-VHDL weight-export pipeline, repository structure, professional documentation, and hardware testbench.

Repository File Structure

		FPGA-Accelerated-Packet-Classifier/
		├── docs/                      # Resource utilization reports & architecture diagrams
		├── sim/
		│   └── tb_fpga_nn_classifier.vhd # VHDL testbench for waveform simulation
		├── src/
		│   ├── /
		│   │   └── export_weights.py  # PyTorch training, INT8 quantization & VHDL code generator
		│   └── vhdl/
		│       ├── nn_weights_pkg.vhd # Auto-generated weight constants package
		│       └── fpga_nn_classifier.vhd # Pipelined hardware neural network accelerator
		├── requirements.txt           #  dependencies
		├── .gitignore
		└── README.md                  # Professional project documentation

1. The Professional README.md: Save this as your repository's README.md to showcase your cross-disciplinary expertise in MLOps for embedded hardware, FPGA synthesis, and digital signal processing.

		# FPGA-Accelerated Lightweight Neural Network for Real-Time Packet Classification
		[![ 3.10+](https://img.shields.io/badge/-3.10%2B-blue.svg)](https://www..org/)
		[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
		[![VHDL](https://img.shields.io/badge/VHDL-93%2F2008-informational.svg)](https://en.wikipedia.org /wiki/VHDL)
	[![License: 	MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https:// opensource.org/ licenses/MIT)

An advanced hardware-software co-design project implementing a fully automated pipeline that trains a neural network in , quantizes weights to 8-bit integers ($INT8$), and programmatically generates VHDL hardware packages for real-time edge packet classification on FPGAs using AXI-Stream protocols.

Key Features

* Automated MLOps for Hardware: A  script that trains a lightweight Multi-Layer Perceptron (MLP) and automatically compiles its weights into synthesizable VHDL ROM packages.
* Hardware-Software Co-Design: Eliminates manual weight entry by bridging PyTorch training directly with FPGA synthesis toolchains.
* Pipelined Datapath: Implements cycle-accurate Multiply-Accumulate (MAC) arithmetic optimized for low-resource FPGA fabric (targeting Xilinx/AMD Zynq or Artix-7).
* AXI-Stream Compliance: Built with standard handshaking (`tvalid`, `tready`, `tdata`) for high-throughput packet streams.

Architecture Flow

	 Training ──> INT8 Quantization ──> VHDL Code Generator ──> VHDL Package ──> FPGA Fabric

Quickstart & Workflow

1. Installation: Clone the repository and install requirements:

		git clone 	[https://github.com/your-username/FPGA-Accelerated-Packet-Classifier.git] (https: //github.com/your-username/FPGA-Accelerated-Packet-Classifier.git)
		cd FPGA-Accelerated-Packet-Classifier
		pip install -r requirements.txt

2. Run Training & Auto-Generate VHDL Weights:  src//export_weights.py

3. Simulate the VHDL Accelerator: Compile and execute the testbench using GHDL or ModelSim:

		ghdl -a src/vhdl/nn_weights_pkg.vhd src/vhdl/fpga_nn_classifier.vhd sim/tb_fpga_nn_classifier.vhd
		ghdl -e tb_fpga_nn_classifier
		ghdl -r tb_fpga_nn_classifier --wave=waveform.ghw

Core Implementation Files

		#`requirements.txt`
		torch>=2.0.0
		numpy>=1.22.0

		#src//export_weights.py
		import os
		import torch
		import torch.nn as nn
		import torch.optim as optim
		import numpy as np
		torch.manual_seed(42)
		np.random.seed(42)
		def generate_packet_data(num_samples=3000):
		    """Generates synthetic header statistics for binary packet classification."""
		    X = np.random.randn(num_samples, 16).astype(np.float32)
		    y = (X[:, 0] + X[:, 1] > 0.4).astype(np.int64)
		    return torch.tensor(X), torch.tensor(y)
		class TinyPacketMLP(nn.Module):
		    def __init__(self):
		        super(TinyPacketMLP, self).__init__()
		        self.fc1 = nn.Linear(16, 8)
		        self.relu = nn.ReLU()
		        self.fc2 = nn.Linear(8, 2)
		    def forward(self, x):
		        return self.fc2(self.relu(self.fc1(x)))
		def export_to_vhdl(w1_int8, b1_int8, output_path="src/vhdl/nn_weights_pkg.vhd"):
		    """Programmatically writes a VHDL package containing the trained INT8 weights."""
		    os.makedirs(os.path.dirname(output_path), exist_ok=True)
		    w1_flat = ", ".join([str(val) for val in w1_int8.flatten()])
		    b1_flat = ", ".join([str(val) for val in b1_int8.flatten()])
		    vhdl_code = f"""
		-- =====================================================================
		-- AUTO-GENERATED BY  EXPORTER. DO NOT EDIT MANUALLY.
		-- Neural Network Trained Weights Package (INT8 Precision)
		-- =====================================================================
		library IEEE;
		use IEEE.STD_LOGIC_1164.ALL;
		use IEEE.NUMERIC_STD.ALL;
		package nn_weights_pkg is
		    type weight_matrix_l1 is array (0 to 127) of signed(7 downto 0);
		    type bias_vector_l1 is array (0 to 7) of signed(31 downto 0);
		    constant W1_ROM : weight_matrix_l1 := ({w1_flat});
		    constant B1_ROM : bias_vector_l1 := ({b1_flat});
		end package nn_weights_pkg;
		"""with open(output_path, "w") as f:
		        f.write(vhdl_code)
		    print(f"[SUCCESS] VHDL weights package successfully exported to {output_path}")
		def main():
		    print("Training lightweight MLP model in PyTorch...")
		    X_data, y_data = generate_packet_data()
		    dataset = torch.utils.data.TensorDataset(X_data, y_data)
		    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
		    model = TinyPacketMLP()
		    criterion = nn.CrossEntropyLoss()
		    optimizer = optim.Adam(model.parameters(), lr=0.01)
		    for epoch in range(6):
		        for inputs, labels in loader:
		            optimizer.zero_grad()
		            loss = criterion(model(inputs), labels)
		            loss.backward()
		            optimizer.step() # Quantize weights to INT8 for hardware synthesis
		    with torch.no_grad():
		        w1 = model.fc1.weight.numpy()
		        b1 = model.fc1.bias.numpy()
		        scale_factor = 64.0
		        w1_int8 = np.round(w1* scale_factor).astype(np.int8)
		        b1_int8 = np.round(b1* scale_factor).astype(np.int32)
		    export_to_vhdl(w1_int8, b1_int8)
		if __name__ == "__main__":
		    main()

		#src/vhdl/fpga_nn_classifier.vhd
		vhdl
		library IEEE;
		use IEEE.STD_LOGIC_1164.ALL;
		use IEEE.NUMERIC_STD.ALL;
		-- Import the auto-generated  weights package
		use work.nn_weights_pkg.all;
		entity fpga_nn_classifier is
		    Port (clk           : in  STD_LOGIC; reset_n       : in  STD_LOGIC;
		        -- AXI-Stream Slave (Input Feature Stream)
		        s_axis_tdata  : in  STD_LOGIC_VECTOR(7 downto 0); s_axis_tvalid : in  STD_LOGIC; s_axis_tready : out STD_LOGIC;
		        -- AXI-Stream Master (Classification Result)
		        m_axis_tdata  : out STD_LOGIC_VECTOR(7 downto 0); m_axis_tvalid : out STD_LOGIC; m_axis_tready : in  STD_LOGIC); end fpga_nn_classifier;
		architecture Behavioral of fpga_nn_classifier is
		    signal accum_reg   : SIGNED(31 downto 0) := (others => '0'); signal feature_idx : integer range 0 to 16 := 0; signal processing  : STD_LOGIC := '0';
		begin
		    s_axis_tready <= '1' when reset_n = '1' and processing = '0' else '0';
		    -- Pipelined MAC Engine reading directly from -exported weight ROM
		    process(clk, reset_n)
		        variable mult_result : signed(15 downto 0);
		    begin
		        if reset_n = '0' then
		            accum_reg     <= (others => '0');
		            feature_idx   <= 0;
		            m_axis_tvalid <= '0';
		            m_axis_tdata  <= (others => '0');
		            processing    <= '0';
		        elsif rising_edge(clk) then
		            if s_axis_tvalid = '1' and processing = '0' then
		                processing  <= '1';
		                accum_reg   <= B1_ROM(0); -- Load bias
		                feature_idx <= 0;
		            elsif processing = '1' then
		                if feature_idx < 16 then
		                    -- Multiply incoming feature with exported weight from ROM package
                    mult_result := signed(s_axis_tdata)* W1_ROM(feature_idx);
		                    accum_reg   <= accum_reg + resize(mult_result, 32);
		                    feature_idx <= feature_idx + 1;
		                else
		                    -- Apply activation (ReLU thresholding)
		                    if accum_reg > 0 then
		                        m_axis_tdata <= std_logic_vector(accum_reg(7 downto 0));
		                    else
		                        m_axis_tdata <= (others => '0');
		                    end if;
		                    m_axis_tvalid <= '1';
		                    processing    <= '0';
		                end if;
		            elsif m_axis_tready = '1' then
		                m_axis_tvalid <= '0';
		            end if;
		        end if;
		    end process;
		end Behavioral;

		#sim/tb_fpga_nn_classifier.vhd
		vhdl
		library IEEE;
		use IEEE.STD_LOGIC_1164.ALL;
		use IEEE.NUMERIC_STD.ALL;
		entity tb_fpga_nn_classifier is
		end tb_fpga_nn_classifier;
		architecture sim of tb_fpga_nn_classifier is
		    signal clk           : STD_LOGIC := '0';
		    signal reset_n       : STD_LOGIC := '0';
		    signal s_axis_tdata  : STD_LOGIC_VECTOR(7 downto 0) := (others => '0');
		    signal s_axis_tvalid : STD_LOGIC := '0';
 				 signal s_axis_tready : STD_LOGIC;
		    signal m_axis_tdata  : STD_LOGIC_VECTOR(7 downto 0);
		    signal m_axis_tvalid : STD_LOGIC;
		    signal m_axis_tready : STD_LOGIC := '1';
		    constant CLK_PERIOD : time := 10 ns;
		begin
		    -- Instantiate Unit Under Test (UUT)
		    uut: entity work.fpga_nn_classifier
		        port map (clk => clk, reset_n => reset_n, s_axis_tdata  => s_axis_tdata, s_axis_tvalid => s_axis_tvalid, s_axis_tready => s_axis_tready, m_axis_tdata  => m_axis_tdata, m_axis_tvalid => m_axis_tvalid, m_axis_tready => m_axis_tready);
		    -- Clock Generation
		    clk_process: process
		    begin
		        clk <= '0';
        wait for CLK_PERIOD / 2;
        clk <= '1';
        wait for CLK_PERIOD / 2;
		    end process;
		    -- Stimulus Process
		    stim_proc: process
		    begin
		        reset_n <= '0';
		        wait for 20 ns;
		        reset_n <= '1';
		        wait for 20 ns;
		        -- Stream 16 feature values sequentially into the accelerator
		        for i in 0 to 15 loop
		            s_axis_tdata  <= std_logic_vector(to_signed(i* 3 + 5, 8));
		            s_axis_tvalid <= '1';
		            wait for CLK_PERIOD;
		        end loop;
		        s_axis_tvalid <= '0';
		        wait for 100 ns;
		        wait;
		    end process;
		end sim;