-- --------------------------------------------------------------------------
-- File : sobel_adder.vhd
--
-- Entity : sobel_adder
-- Architecture : behavioral
-- Author : Carter Nesbitt
-- Created : 21 Sep 2019
-- Modified : 21 Sep 2019
--
-- VHDL '93
-- Description: 
-- --------------------------------------------------------------------------

library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity sobel_adder is
	generic (WIDTH : integer := 5)
	port ( 
		i_clk : std_logic;
		i_kernel : std_logic_vector(WIDTH*8-1 downto 0);
		o_filter : std_logic_vector(WIDTH-1 downto 0)	 
	);
end entity sobel_adder;

architecture behavioral of sobol_adder is
	
	r_sum : integer := 0;
	type convolution is array(8 downto 0) of integer;
	signal conv_x : convolution := (
		-1, 0, 1, 
		-2, 0, 2, 
		-1, 0, 1;
	)
	signal conv_y : convolution := (
		1, 2, 1, 
		0, 0, 0, 
		-1, -2, -1;
	)
begin

	sum_proc : process(i_clk) is begin
		if rising_edge(i_clk) then
			r_sum <= 0;
			sum_gen : for idx in (8 downto 0) generate
				r_sum <= r_sum + (conv_x(idx)+conv_y(idx)) *
					to_integer(unsigned(i_kernel(WIDTH*idx-1 downto WIDTH*(idx-1))));
			end generate;
		end if;
	end process sum_process;
	o_filter <= r_sum;

--	o_filter <= to_integer(unsigned(i_kernel(WIDTH*7-1 downto WIDTH*6) 
--							- to_integer(unsigned(i_kernel(WIDTH*9-1 downto WIDTH*8)))
--							+ to_integer(unsigned(i_kernel(WIDTH*4-1 downto WIDTH*3))) 
--							- to_integer(unsigned(i_kernel(WIDTH*6-1 downto WIDTH*5))) 
--							+ to_integer(unsigned(i_kernel(WIDTH*1-1 downto 0)))
--							- to_integer(unsigned(i_kernel(WIDTH*3-1 downto WIDTH*2)));
end behavioral;
