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
        port map (
            clk           => clk,
            reset_n       => reset_n,
            s_axis_tdata  => s_axis_tdata,
            s_axis_tvalid => s_axis_tvalid,
            s_axis_tready => s_axis_tready,
            m_axis_tdata  => m_axis_tdata,
            m_axis_tvalid => m_axis_tvalid,
            m_axis_tready => m_axis_tready
        );
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