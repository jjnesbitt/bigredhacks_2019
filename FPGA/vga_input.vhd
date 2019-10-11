-- --------------------------------------------------------------------------
-- File : vga_to_xy.vhd
--
-- Entity : vga_to_xy
-- Architecture : behavioral
-- Author : Carter Nesbitt
-- Created : 20 Sep 2019
-- Modified : 21 Sep 2019
--
-- VHDL '93
-- Description: Decodes a basic VGA signal input to determine the x and
--							y coordinates of each pixel.
-- --------------------------------------------------------------------------

library IEEE;
USE IEEE.std_logic_1164.all;

entity vga_to_xy is
	generic (
		h_pulse   : INTEGER := 208;     --horiztonal sync pulse width in pixels
	  h_bp    : INTEGER := 336;   --horiztonal back porch width in pixels
		h_pixels  : INTEGER := 1920;    --horiztonal display width in pixels
		h_fp    : INTEGER := 128;   --horiztonal front porch width in pixels
		h_pol   : STD_LOGIC := '0';   --horizontal sync pulse polarity (1 = positive, 0 = negative)
		v_pulse   : INTEGER := 3;     --vertical sync pulse width in rows
		v_bp    : INTEGER := 38;      --vertical back porch width in rows
		v_pixels  : INTEGER := 1200;    --vertical display width in rows
		v_fp    : INTEGER := 1;     --vertical front porch width in rows
		v_pol   : STD_LOGIC := '1');  --vertical sync pulse polarity (1 = positive, 0 = negative)
	) port (
		i_clk : in STD_LOGIC;
		i_rstn	: in STD_LOGIC;
		i_h_sync : in STD_LOGIC;
		i_v_sync : in STD_LOGIC;
		o_x	: out INTEGER;
		o_y : out INTEGER;
		o_pixel_active : out STD_LOGIC
	);
end entity vga_to_xy;

architecture behavioral of vga_to_xy is
	r_x : INTEGER := 0;
	r_y : INTEGER := 0;
begin
	process (i_clk, i_rstn) is begin
		if i_rstn = '0' then
			r_x <= 0;
			r_y <= 0;
		elsif rising_edge(i_clk) then
			if r_x >= h_pixels + h_fp + h_sync + h_bp then
				r_x <= 0;
				if r_y >= v_pixels + v_fp + v_sync + v_bp then
					r_y <= 0;
				else
					r_y <= r_y + 1;
				end if;
			else
				r_x <= r_x + 1;
			end if;
		end if;
	end process;

	o_pixel_active <= '1' when r_x < h_pixels else '0';

	o_x <= r_x;
	o_y <= r_y;
end architecture behavioral;
