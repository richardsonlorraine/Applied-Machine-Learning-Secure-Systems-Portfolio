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