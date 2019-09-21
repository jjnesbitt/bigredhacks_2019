-- --------------------------------------------------------------------------
-- File : vga_sobel.vhd
-- Entity : vga_sobel
-- Architecture : behavioral
-- Author : Carter Nesbitt
-- Created : 21 Sep 2019
-- Modified : 21 Sep 2019
--
-- VHDL '93
-- Description: Performs rgb kernel convolution for sobel edge detection.
-- --------------------------------------------------------------------------

library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity vga_sobel is
	generic (
		IMAGE_X : integer := 1920;
		IMAGE_Y : integer := 1200;
	) port (
		i_clk : in std_logic;
		i_x : in integer;
		i_y : in integer;
		i_red : in std_logic_vector(4 downto 0);
		i_green : in std_logic_vector(4 downto 0);
		i_blue : in std_logic_vector(4 downto 0);
		o_filter : out std_logic_vector(4 downto 0)
	);
end entity vga_sobel;

architecture behavioral of vga_sobel is
	signal r_intensity : integer := 0;
	type image_data is array (IMAGE_X * 3) of std_logic_vector(4 downto 0);
	signal image_metadata : image_data := (others => (others => '0'));
	signal w_y_depth : integer := 0;
	signal r_kernel : std_logic_vector(9*5-1 downto 0);	
begin

	w_y_depth <= i_y when i_y < 3 else 3;

	filter_out : entity work.sobel_adder
		generic map( WIDTH => 5)
		port map( i_clk => i_clk, i_kernel => , o_filter => o_filter);

	intensity_calculate : process(i_red, i_blue, i_green) is begin
		r_intensity <= (to_integer(unsigned(i_red)) 
									 + to_integer(unsigned(i_red)) 
									 + to_integer(unsigned(i_red))) / 3;
	end process intensity_calculate;

	metadata_assign : process (i_clk) is begin
		if rising_edge(i_clk) then
			image_metadata(i_x + IMAGE_X*r_y_depth) <= r_intensity;
		end if;
	end process metadata_assign;

	metadata_fetch : process (i_clk, i_x, i_y) is begin
		if (rising_edge(i_clk) and i_x > 1 and i_y > 1) then
			r_kernel <= image_metadata(i_x-1 + (i_y-1)*IMAGE_X) &
									image_metadata(i_x + (i_y-1)*IMAGE_X) &
									image_metadata(i_x+1 + (i_y-1)*IMAGE_X) &
									image_metadata(i_x-1 + i_y*IMAGE_X) &
									image_metadata(i_x + i_y*IMAGE_X) &
									image_metadata(i_x+1 + i_y*IMAGE_X) &
									image_metadata(i_x-1 + (i_y+1)*IMAGE_X &
									image_metadata(i_x + (i_y+1)*IMAGE_X &
									image_metadata(i_x+1 + (i_y+1)*IMAGE_X;

		else
			r_kernel <= (others => '0');
		end if;
	end process metadata_fetch;

	block_shift : process (i_clk, i_y, w_y_depth) is begin
		if (i_y = IMAGE_Y and w_y_depth = 3) then
			for y_idx in 1 to 2 generate
				for x_idx in 0 to IMAGE_X-1 generate
					image_metadata(x_idx + y_idx*IMAGE_X) 
						<= image_metadata(x_idx + (y_idx-1)*IMAGE_X);
					end generate;
			end generate;
		end if;
	end process block_shift;

end architecture behavioral;
