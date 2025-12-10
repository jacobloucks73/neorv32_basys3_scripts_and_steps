library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library neorv32;

use neorv32.tensor_operations_basic_arithmetic.all; --import opcodes/constants and packed int8 add/sub
use neorv32.tensor_operations_pooling.all;          --import pooling opcodes & helpers (read/max/avg)
use neorv32.tensor_operations_sigmoid.all;          -- vhdl file for the sigmoid library
use neorv32.tensor_operations_flatten.all;          -- vhdl file for the flatten library
use neorv32.conv2d_package.all;                        -- NEW: Custom Conv2D opcodes and functions
use neorv32.Dense_Package.all;
entity wb_peripheral_top is
  generic (
    BASE_ADDRESS            : std_ulogic_vector(31 downto 0) := x"90000000"; --peripheral base (informational)
    TENSOR_A_BASE           : std_ulogic_vector(31 downto 0) := x"90001000"; --A window base
    TENSOR_B_BASE           : std_ulogic_vector(31 downto 0) := x"90002000"; --B window base
    TENSOR_C_BASE           : std_ulogic_vector(31 downto 0) := x"90003000"; --C window base
    TENSOR_R_BASE           : std_ulogic_vector(31 downto 0) := x"90004000"; --R window base
    CTRL_REG_ADDRESS        : std_ulogic_vector(31 downto 0) := x"90000008"; --[0]=start, [5:1]=opcode
    STATUS_REG_ADDRESS      : std_ulogic_vector(31 downto 0) := x"9000000C"; --[0]=busy, [1]=done (sticky)
    DIM_REG_ADDRESS         : std_ulogic_vector(31 downto 0) := x"90000010"; --N (LSB 8 bits)
    POOL_BASE_INDEX_ADDRESS : std_ulogic_vector(31 downto 0) := x"90000014"; --top-left idx in A
    POOL_OUT_INDEX_ADDRESS  : std_ulogic_vector(31 downto 0) := x"90000018"; --out idx in R
    WORD_INDEX_ADDRESS      : std_ulogic_vector(31 downto 0) := x"9000001C"  --word index for OP_ADD/OP_SUB
  );
  
  port (
         -- Global Signals -- 
    clk        : in  std_ulogic;                    --system clock
    reset      : in  std_ulogic;                    --synchronous reset
    
     -- Wishbone Slave Interface (for Control and Status Registers) --
    i_wb_cyc   : in  std_ulogic;                    --Wishbone: cycle valid
    i_wb_stb   : in  std_ulogic;                    --Wishbone: strobe
    i_wb_we    : in  std_ulogic;                    --Wishbone: 1=write, 0=read
    i_wb_addr  : in  std_ulogic_vector(31 downto 0);--Wishbone: address
    i_wb_data  : in  std_ulogic_vector(31 downto 0);--Wishbone: write data
    o_wb_ack   : out std_ulogic;                    --Wishbone: acknowledge
    o_wb_stall : out std_ulogic;                    --Wishbone: stall (always '0')
    o_wb_data  : out std_ulogic_vector(31 downto 0) --Wishbone: read data
  );
  
end entity;

architecture rtl of wb_peripheral_top is

    -- Generic tensor size definition (4kB memory / 1024 words)
  constant TENSOR_WORDS : integer := 1024;
  
  --Tensor memories (packed 4x int8 per 32-bit word)
  signal tensor_A         : tensor_mem_type := (others => (others => '0'));
  signal tensor_B         : tensor_mem_type := (others => (others => '0'));
  signal tensor_C         : tensor_mem_type := (others => (others => '0'));
  signal tensor_R         : tensor_mem_type := (others => (others => '0'));

  -- Control/Status Registers
  signal ctrl_reg         : std_ulogic_vector(31 downto 0) := (others => '0');
  signal status_reg       : std_ulogic_vector(31 downto 0) := (others => '0'); -- [0]=busy, [1]=error, [2]=done
  signal dim_side_len_8   : std_ulogic_vector(7 downto 0)  := (others => '0'); --N side length
  signal dim_side_len_bus : std_ulogic_vector(31 downto 0) := (others => '0'); --zero-extended N
  
  -- Internal Datapath Signals
  signal start_reg       : std_ulogic := '0';        -- latched start bit
  signal op_code_reg     : std_ulogic_vector(4 downto 0)  := (others => '0'); -- latched opcode
  signal din_reg         : unsigned(7 downto 0)           := (others => '0'); -- latched N (tensor side length)
  signal pool_base_index : std_ulogic_vector(31 downto 0) := (others => '0'); -- start index for pooling/conv2d
  signal pool_out_index  : std_ulogic_vector(31 downto 0) := (others => '0'); -- pooling output parameters
  signal word_i_reg      : unsigned(15 downto 0)          := (others => '0'); -- latched word index
  signal base_i_reg      : unsigned(15 downto 0)          := (others => '0'); --pooling base index
  signal out_i_reg       : unsigned(15 downto 0)          := (others => '0'); --pooling output index

  --Wishbone readback data and ack register
  signal data_r          : std_ulogic_vector(31 downto 0) := (others => '0');
  signal ack_r           : std_ulogic := '0';

  --Elementwise word index
  signal word_index_reg  : std_ulogic_vector(31 downto 0) := (others => '0'); --packed word index

  --Start edge detection (one-cycle pulse)
  signal start_cmd        : std_ulogic := '0';
  signal ctrl0_prev       : std_ulogic := '0';

  --Muxed write paths for DIM (allowing bus or internal updates)
  signal bus_dim_we       : std_ulogic := '0';
  signal bus_dim_data     : std_ulogic_vector(7 downto 0) := (others => '0');
  signal pool_dim_we      : std_ulogic := '0';
  signal pool_dim_data    : std_ulogic_vector(7 downto 0) := (others => '0');
  
    --Pooling datapath registers (2x2 window and result)
  signal num00_reg, num01_reg, num10_reg, num11_reg : signed(7 downto 0) := (others => '0');
  signal r8_reg           : signed(7 downto 0) := (others => '0');

  --Vector datapath registers for packed word operations
  signal a_w_reg, b_w_reg, c_w_reg, r_w_reg : std_ulogic_vector(31 downto 0)    := (others => '0');
  signal read_idx         : unsigned(1 downto 0)              := (others => '0');
  signal flat_idx         : integer range 0 to TENSOR_WORDS := 0;  -- which word we are on
  signal flat_total       : integer range 0 to TENSOR_WORDS := TENSOR_WORDS;
  
   --Latched operation parameters for the active command
  signal r_op_code        : std_ulogic_vector(4 downto 0) := (others => '0');
  signal r_N              : unsigned(7 downto 0) := (others => '0');

  --Vector datapath registers for packed word operations
  signal word_idx         : integer range 0 to TENSOR_WORDS := 0; -- temporary word index
  signal byte_sel         : integer range 0 to 3 := 0;            -- temporary byte select
  signal sel_byte         : std_ulogic_vector(7 downto 0);        -- temporary selected byte
  signal packed_word      : std_ulogic_vector(31 downto 0);       -- temporary packed word
  
  -- NEW: Conv2D Signals for 3x3, Stride 1, No Padding, Single Channel
  signal dim_height_width : std_ulogic_vector(15 downto 0) := (others => '0');
  signal kernel_idx_cnt   : unsigned(3 downto 0) := (others => '0'); -- 0..8
  signal stall_count      : unsigned(3 downto 0);
  signal mac_acc_reg      : signed(19 downto 0)           := (others => '0'); -- MAC accumulator (16-bit product + 4 bits for 9 terms)
  signal mult_result      : signed(15 downto 0)           := (others => '0'); -- I * K product
  signal input_val        : signed(7 downto 0)            := (others => '0'); -- Current input element I[x][y]
  signal kernel_val       : signed(7 downto 0)            := (others => '0'); -- Current kernel element K[i][j]


  --TODO make signed and test edge cases 
  signal in_height        : unsigned(7 downto 0);            -- H_in
  signal in_width         : unsigned(7 downto 0);            -- W_in
  signal out_height       : unsigned(7 downto 0);            -- H_out = H_in - 2
  signal out_width        : unsigned(7 downto 0);            -- W_out = W_in - 2
  
  -- Output (y,x) counters
  signal out_y_cnt        : unsigned(7 downto 0);
  signal out_x_cnt        : unsigned(7 downto 0);
  
  -- new wave conv2d signals 
  signal kernel_9         : std_ulogic_vector(71 downto 0) := (others => '0');
  signal image_9          : std_ulogic_vector(71 downto 0) := (others => '0');
  
  
  -- Dense Signals 
--  signal neuron_cnt     : unsigned(7 downto 0); -- used for multiple output neurons in one software call, depreceated
  signal elem_cnt         : unsigned(7 downto 0);
  signal x_val            : signed(7 downto 0);
  signal w_val            : signed(7 downto 0);
  signal acc_bias         : signed(19 downto 0);
  
  
  
  
  
  
  
  
  
  --Address helper: translate byte address to word offset within a tensor window
  function get_tensor_offset(addr, base: std_ulogic_vector(31 downto 0)) return natural is
    variable offset: unsigned(31 downto 0);
    
  begin
    offset := unsigned(addr) - unsigned(base);        --byte delta
    return to_integer(offset(11 downto 2));           --divide by 4 (32-bit words)
  end function;

  --Byte-lane write helper: write signed(7 downto 0) into selected byte of a 32-bit word
  procedure set_int8_into_word(signal R_tensor : inout tensor_mem_type;
                               index : in natural; val : in signed(7 downto 0)) is
                               
    variable w_index    : natural := index / 4;       --which 32-bit word
    variable byte_index : natural := index mod 4;     --which byte within the word
    variable word       : std_ulogic_vector(31 downto 0);
    
  begin
    word := R_tensor(w_index);                        --read-modify-write the word
    case byte_index is
      when 0      => word(7  downto 0)  := std_ulogic_vector(val);
      when 1      => word(15 downto 8)  := std_ulogic_vector(val);
      when 2      => word(23 downto 16) := std_ulogic_vector(val);
      when others => word(31 downto 24) := std_ulogic_vector(val);
    end case;
    
    R_tensor(w_index) <= word;                        --write back updated word
    
  end procedure;

  --Unified FSM state encoding
  type state_t is (
    S_IDLE,       -- waiting for start command
    S_OP_CODE_BRANCH, -- decode operation and jump to setup state
    
    --Pooling path
    S_P_READ, S_P_WRITE,S_P_CALC,
    --Sigmoid path
     S_SIG_READ, S_SIG_WRITE, S_SIG_CALC,
    --Flatten path
    -- S_F_SETUP,S_F_RUN,
    --idk what this is but it looks important so it stays
    S_CAPTURE,

    --elemwise add/sub path
    S_V_READA, S_V_READB, S_V_READC, S_V_CALC, S_V_WRITE,
    -- NEW: Conv2D path
    S_CONV_SETUP, S_CONV_READ_IK, S_CONV_MAC, S_CONV_WRITE, S_CONV_INIT, --S_CONV_STALL_1, S_CONV_ITER,
    
    S_DENSE_INIT, S_DENSE_READ_IN, S_DENSE_MAC, S_DENSE_BIAS, S_DENSE_WRITE,
    
    S_DONE
  );
  
  signal state : state_t := S_IDLE; 



begin
  --Simple, non-stalling slave
  o_wb_stall <= '0';
  --Zero-extend N for bus readback
  dim_side_len_bus <= (31 downto 8 => '0') & dim_side_len_8;



  --Generate a one-cycle start pulse when start=1 and not busy
  process(clk)
  
  begin
  
    if rising_edge(clk) then
      if reset = '1' then
        start_cmd <= '0';
        ctrl0_prev  <= '0';     
      else
        start_cmd <= '0';      
        if (status_reg(0) = '0' and ctrl_reg(0) = '1' and (ctrl0_prev = '0')) then
          start_cmd <= '1';
        end if;
        ctrl0_prev <= ctrl_reg(0);
      end if;
    end if;
    
  end process;
  
  
  --DIM (N) register with two write sources: pooling path or bus write
  process(clk)
  
  begin
    if rising_edge(clk) then
      if reset = '1' then
        dim_side_len_8 <= x"1C"; --default N=28
      else
        if pool_dim_we = '1' then
          dim_side_len_8 <= pool_dim_data; --internal update path (currently unused)
        elsif bus_dim_we = '1' then
          dim_side_len_8 <= bus_dim_data;  --bus write-update
        end if;
      end if;
    end if;
    
  end process;



  --Unified FSM handling pooling and vector operations
  process(clk)
    variable elem_index      : unsigned(15 downto 0);                  -- flat index into A/R
    variable word_idx        : natural;                                -- 32-bit word index
    variable byte_sel        : integer;                                -- byte lane select 0..3
    variable packed_word     : std_ulogic_vector(31 downto 0);         -- fetched 32-bit word
    variable sel_byte        : std_ulogic_vector(7 downto 0);          -- selected byte from word
    variable base            : integer;                                -- transformation objuect for signed ints
    variable input_idx_2d    : unsigned(31 downto 0);                  -- Input tensor index
    variable kernel_idx_2d   : unsigned(31 downto 0);                  -- Kernel tensor index
    variable output_idx_2d   : unsigned(31 downto 0);                  -- Output tensor index
    variable y               : integer ;
    variable x               : integer ;
    variable ky              : integer ;
    variable kx              : integer ;
    variable Nw              : integer ;
    variable inp_word_idx    : integer;
    variable inp_byte_sel    : integer;
    variable inp_word        : std_ulogic_vector(31 downto 0);
    variable inp_sel         : std_ulogic_vector(7 downto 0);
    variable ker_word_idx    : integer;
    variable ker_byte_sel    : integer;
    variable ker_word        : std_ulogic_vector(31 downto 0);
    variable ker_sel         : std_ulogic_vector(7 downto 0);
    variable prod            : signed(15 downto 0);
    variable ker_index       : integer;
    variable k_flat          : integer;
    variable q8              : signed(7 downto 0); -- i hate variables with a passion
    variable acc_tmp         : signed(mac_acc_reg'length-1 downto 0);
    
    --Dense layer vars
    variable bias_val        : signed(7 downto 0); 
    
    
  begin
    if rising_edge(clk) then
      if reset = '1' then
        state       <= S_IDLE;
        
        status_reg  <= (others => '0');
        
        start_reg   <= '0';
        op_code_reg <= (others => '0');
        din_reg     <= (others => '0');
        word_i_reg  <= (others => '0');

        r_w_reg     <= (others => '0');
        a_w_reg     <= (others => '0');
        b_w_reg     <= (others => '0');
        c_w_reg     <= (others => '0');
 
 
        base_i_reg  <= (others => '0');
        out_i_reg   <= (others => '0');
        din_reg     <= (others => '0');
        word_i_reg  <= (others => '0');
        r8_reg      <= (others => '0');
        
        -- Conv2D Reset -- 
        mac_acc_reg    <= (others => '0');
        mult_result    <= (others => '0');
--        kernel_h_cnt <= (others => '0');
--        kernel_w_cnt <= (others => '0');
        kernel_idx_cnt <= (others => '0');
        in_height      <= (others => '0');
        in_width       <= (others => '0');
        
        -- Dense Reset --
        -- neuron_cnt     <= (others => '0');
        elem_cnt       <= (others => '0');
        
        flat_idx    <= 0;
        pool_dim_we <= '0';
        
      else

        pool_dim_we <= '0';

-----------------------------------------
        -- FINITE STATE MACHINE -- 
-----------------------------------------
        case state is
          when S_IDLE =>
            status_reg(0) <= '0';                --not busy
            if start_cmd = '1' then
              status_reg(1) <= '0';             --clear done
              state <= S_CAPTURE;               --capture parameters
            end if;
            
            
        when S_CAPTURE =>
          status_reg(0) <= '1';
          op_code_reg   <= ctrl_reg(5 downto 1);                               -- opcode register, tells the FSM what to do
          din_reg       <= unsigned(dim_side_len_8);                           -- apperantly fucking N
          in_height     <= unsigned(dim_height_width(15 downto 8));
          in_width      <= unsigned(dim_height_width(7 downto 0));
--          kernel_h_cnt  <= (others => '0');
--          kernel_w_cnt  <= (others => '0');
          out_y_cnt     <= (others => '0');  
          out_x_cnt     <= (others => '0');
            
          base_i_reg    <= unsigned(pool_base_index(15 downto 0));             -- pooling base index 
          out_i_reg     <= unsigned(pool_out_index(15 downto 0));              -- pooling output index
          word_i_reg    <= unsigned(word_index_reg(15 downto 0));              -- packed word index
          read_idx      <= (others => '0');                                    -- start 2x2 sweep at top-left
          state         <= S_OP_CODE_BRANCH;                                         -- take to operation center 
 

       when S_OP_CODE_BRANCH =>                                              -- Operation Center

          if (op_code_reg = OP_MAXPOOL) or (op_code_reg = OP_AVGPOOL) then   -- Pooling 
            state <= S_P_READ;
            
          elsif(op_code_reg = OP_SIGMOID) then                               -- Sigmoid Functon
            state <= S_SIG_READ;
            
          elsif (op_code_reg = OP_ADD) or (op_code_reg = OP_SUB) then        -- Addition and subtraction
            state <= S_V_READA;
            
--          elsif (op_code_reg = OP_FLAT) then                                 -- Flatten function opcode == 01000
--            state <= S_F_SETUP;
            
          elsif (op_code_reg = OP_CONV2D) then                               -- Conv2D operation
            state <= S_CONV_INIT;  
            
          else
            status_reg(0) <= '0'; status_reg(1) <= '1'; state <= S_IDLE;
          end if;
         
         when S_P_READ =>
              --Compute flat element index for the current 2x2 position
              case read_idx is
                when "00" => elem_index := base_i_reg;                                                --(0,0)
                when "01" => elem_index := base_i_reg + 1;                                            --(0,1)
                when "10" => elem_index := base_i_reg + resize(din_reg, elem_index'length);           --(1,0)
                when others => elem_index := base_i_reg + resize(din_reg, elem_index'length) + 1;     --(1,1)
              end case;
            
              --Decode packed word and byte lane
              word_idx := to_integer(elem_index(15 downto 2));
              byte_sel := to_integer(elem_index(1 downto 0));
              packed_word := tensor_A(word_idx);
              case byte_sel is
                when 0 => sel_byte := packed_word(7  downto 0);
                when 1 => sel_byte := packed_word(15 downto 8);
                when 2 => sel_byte := packed_word(23 downto 16);
                when others => sel_byte := packed_word(31 downto 24);
              end case;
            
              --Store into the appropriate corner register
              case read_idx is
                when "00" => num00_reg <= signed(sel_byte);
                when "01" => num01_reg <= signed(sel_byte);
                when "10" => num10_reg <= signed(sel_byte);
                when others => num11_reg <= signed(sel_byte);
              end case;
              
              --Advance or move to compute
              if read_idx = "11" then
                state <= S_P_CALC;
              else
                read_idx <= read_idx + 1;
                state <= S_P_READ;
              end if;
            
          --Pooling compute: avg or max across 2x2, result in r8_reg
          when S_P_CALC =>
            if op_code_reg = OP_AVGPOOL then
              r8_reg <= avgpool4(num00_reg, num01_reg, num10_reg, num11_reg);
            else
              r8_reg <= maxpool4(num00_reg, num01_reg, num10_reg, num11_reg);
            end if;
            state <= S_P_WRITE;

          --Pooling writeback: write single int8 into R at out_i_reg
          when S_P_WRITE =>
            set_int8_into_word(tensor_R, to_integer(out_i_reg), r8_reg);
            state <= S_DONE;

          --Sigmoid states
          when S_SIG_READ =>
            a_w_reg <= tensor_A(to_integer(word_i_reg));  --read A[word] once
            state   <= S_SIG_CALC;
            
            
          when S_SIG_CALC =>
            r_w_reg <= sigmoid_packed_word(a_w_reg);      --apply sigmoid to word
            state   <= S_SIG_WRITE;
            
            
          when S_SIG_WRITE =>
            tensor_R(to_integer(word_i_reg)) <= r_w_reg;  --write result to R[word]
            state <= S_DONE;
          
          --Vector path: read packed A/B/C words at word_i_reg
          when S_V_READA =>
            a_w_reg <= tensor_A(to_integer(word_i_reg));
            state <= S_V_READB;


          when S_V_READB =>
            b_w_reg <= tensor_B(to_integer(word_i_reg));
            state <= S_V_READC;


          when S_V_READC =>
            c_w_reg <= tensor_C(to_integer(word_i_reg));
            state <= S_V_CALC;

          --Vector calc: lane-wise add or sub using imported functions
          when S_V_CALC =>
            if op_code_reg = OP_ADD then
              r_w_reg <= add_packed_int8(a_w_reg, b_w_reg, c_w_reg);
            else
              r_w_reg <= sub_packed_int8(a_w_reg, b_w_reg, c_w_reg);
            end if;
            state <= S_V_WRITE;

          --Vector writeback: write packed result word to R
          when S_V_WRITE =>
            tensor_R(to_integer(word_i_reg)) <= r_w_reg;
            state <= S_DONE;
          
          -----------------------------------------
          -- NEW: Conv2D FSM States (3x3, Stride 1)
          -----------------------------------------
          
          when S_CONV_INIT =>
            -- Initialize output pixel sweep
            -- H_out = H_in - 2, W_out = W_in - 2
              out_height <= in_height - 2;  -- assumes H_in >= 3
              out_width  <= in_width  - 2;  -- assumes W_in >= 3

              -- Start at output (0,0)
              out_y_cnt <= (others => '0');
              out_x_cnt <= (others => '0');


              --Kernel input doesnt change so no need to do it multiple times
                
                 for k in 0 to 8 loop

                    word_idx := k / 4;    -- which word?
                    byte_sel := k mod 4;  -- which byte in that word?

                    packed_word := tensor_B(word_idx);

                    case byte_sel is
                        when 0 => sel_byte := packed_word(7 downto 0);
                        when 1 => sel_byte := packed_word(15 downto 8);
                        when 2 => sel_byte := packed_word(23 downto 16);
                        when others => sel_byte := packed_word(31 downto 24);
                    end case;
                 kernel_9((k*8)+7 downto k*8) <= sel_byte; -- weird bullshit go
                
                 end loop;
                 state <= S_CONV_SETUP;

          when S_CONV_SETUP =>
            -- Initialize MAC, output counters, and kernel counters for a new pixel calculation
            
             mac_acc_reg    <= (others => '0');
             mult_result    <= (others => '0');
             kernel_idx_cnt <= (others => '0'); 
             
             -- get the first 9 inputs from tensor A in the row, everytime the y overflows
   
             for position in 0 to 8 loop

                ky := position / 3;
                kx := position mod 3;

                -------------------------------------------------------
                -- Compute input coordinate
                -------------------------------------------------------
                y  := to_integer(out_y_cnt) + ky;
                x  := to_integer(out_x_cnt) + kx;
                Nw := to_integer(in_width); -- why does this need to be instantiated again?

--                input_idx_2d := to_unsigned(y * Nw + x, 32);

                -------------------------------------------------------
                -- Read input tensor A[y,x]
                -------------------------------------------------------
                word_idx := (y * Nw + x) / 4;
                byte_sel := (y * Nw + x) mod 4;

                packed_word := tensor_A(word_idx);

                case byte_sel is
                  when 0 => sel_byte := packed_word(7  downto 0);
                  when 1 => sel_byte := packed_word(15 downto 8);
                  when 2 => sel_byte := packed_word(23 downto 16);
                  when others => sel_byte := packed_word(31 downto 24);
                end case;
                
                image_9((position*8)+7 downto position*8) <= sel_byte; -- weird bullshit go
                 
            end loop;
            
             -- at this poiint you should have the first row tensor for input, and the kernel. so you should be able to complete the first MAC.
             
             state <= S_CONV_MAC;

          when S_CONV_READ_IK =>

            image_9(7 downto 0) <= image_9(15 downto 8);
            image_9(15 downto 8) <= image_9(23 downto 16);
            
            image_9(31 downto 24) <= image_9(39 downto 32);
            image_9(39 downto 32) <= image_9(47 downto 40);
            
            image_9(55 downto 48) <= image_9(63 downto 56);
            image_9(63 downto 56) <= image_9(71 downto 64);
            
            for row in 0 to 2 loop

                -------------------------------------------------------
                -- Compute input coordinate for the last three needed inputs
                -------------------------------------------------------
                y  := to_integer(out_y_cnt) + row ;
                x  := to_integer(out_x_cnt) + 2;
                Nw := to_integer(in_width); -- why does this need to be instantiated again?

--                input_idx_2d := to_unsigned(y * Nw + x, 32);

                -------------------------------------------------------
                -- Read input tensor A[y,x]
                -------------------------------------------------------
                word_idx := (y * Nw + x) / 4;
                byte_sel := (y * Nw + x) mod 4;

                packed_word := tensor_A(word_idx);

                case byte_sel is
                  when 0 => sel_byte := packed_word(7  downto 0);
                  when 1 => sel_byte := packed_word(15 downto 8);
                  when 2 => sel_byte := packed_word(23 downto 16);
                  when others => sel_byte := packed_word(31 downto 24);
                end case;
            
            
                if (y = 0) then
                image_9(23 downto 16) <= sel_byte;
                
                elsif (y = 1) then
                image_9(47 downto 40) <= sel_byte;
                
                elsif (y = 2) then
                image_9(71 downto 64) <= sel_byte;
            
            end if;

           end loop;

            state <= S_CONV_MAC;

          when S_CONV_MAC =>
            -- Cycle 2: Multiply and Accumulate
            
            acc_tmp := (others => '0');
            
                for MAC in 0 to 8 loop
                -- Multiply: I * K (Result ready this cycle)
--                    prod := input_val * kernel_val;
                    
                    prod := signed(image_9((MAC*8)+7 downto MAC*8)) * signed(kernel_9((MAC*8)+7 downto MAC*8));
                    
                
                    -- Accumulate: Acc += I * K
                    acc_tmp := acc_tmp + resize(prod, acc_tmp'length);
                
                end loop;
                
                  mac_acc_reg <= acc_tmp;
                  
                  
                    -- All 9 MACs done
                state <= S_CONV_WRITE;
               
          when S_CONV_WRITE =>
            -- 1. Final Accumulation is done. (
            
            -- 2. Quantize and Clamp the 20-bit result to 8-bit signed.
--             r8_reg <= resize(mac_acc_reg, 8);
               q8 := resize(mac_acc_reg, 8);
            
            -- 3. Calculate Output Tensor R Index (O[y][x])
            -- Output Index = Output_H * (N-2) + Output_W
            output_idx_2d := resize(out_y_cnt, 16) * 
                             resize(out_width, 16) +
                             resize(out_x_cnt, 16);
       
            -- 4. Write result back to R
            set_int8_into_word(tensor_R, to_integer(output_idx_2d), q8);
            
            ------------------------------------------------------------------
            -- OUTER LOOP: slide output position (out_x, out_y)
            ------------------------------------------------------------------
            if out_x_cnt < (out_width-1) then
                      out_x_cnt <= out_x_cnt + 1;
                      state     <= S_CONV_READ_IK; -- just reset x

            else
                      -- End of row: reset x, maybe next row
               out_x_cnt <= (others => '0');
                      
               if out_y_cnt < (out_height-1) then 
                    out_y_cnt <= out_y_cnt + 1;
                    state     <= S_CONV_SETUP; -- reset y
                    else
                    -- All output pixels done
                        state <= S_DONE;
                end if;
            end if;
          -----------------------------------------
          -- End of NEW: Conv2D FSM States
          -----------------------------------------
         
         
          when S_DENSE_INIT =>
            -- input will be given in Tensor A, weights will be given in Tensor B, and biases in Tensor C 
            -- lets say the format for the weights entering the tensor B are 1 output neuron at a time,
            -- in other words, if the input is a 28x28 byte tensor, flattened, Tensor B will contain 28x28 * numOutputNeurons, 
            -- in the mnist set this would calculate to 7840 weights, which is just below our limit for the B tensor.
            -- for larger datasets this would not work to do all weights in the tensor at once, hence we need to limit somehow
            -- I will try to make it do one output neuron at a time  this will give us much more room and time to play with 
            -- biases are not really a factor here, there is one per output neuron, so given our ten there would be ten biases
            
            -- Inputs  in tensor_A (N int8)
            -- Weights in tensor_B (N int8 per output neuron)
            -- Bias    in tensor_C (1 int8 per output neuron)
            -- Output in tensor_R (1 int8 per output neuron)
            
            -- to call in software, set dim reg height to number of input nodes 
            -- and dim reg width to number of output nodes 
            -- then write the above to tensors
            -- set opcode, repeat for number of output nodes
            
            --TODO OPTIMIZATIONS
            
            -- keep the input on hand, this doesnt need to be loaded in everytime, same for all output neurons 
            
            
            -- TODO ::: set these to variables and signals already defined to save space and LUTs 
--            neuron_cnt      <= (others => '0');   -- which output neuron we are processing
            elem_cnt        <= (others => '0');   -- which input element i = 0..N-1 
            mac_acc_reg     <= (others => '0');   -- accumulator
            state           <= S_DENSE_READ_IN;   -- begin first input/weight read
            
            
          when S_DENSE_READ_IN =>   
             word_idx := to_integer(elem_cnt) / 4; --byte selecting
             byte_sel := to_integer(elem_cnt) mod 4;
             
             packed_word := tensor_A(word_idx);
             
             case byte_sel is
               when 0 => sel_byte := packed_word(7  downto 0);
               when 1 => sel_byte := packed_word(15 downto 8);
               when 2 => sel_byte := packed_word(23 downto 16);
               when others => sel_byte := packed_word(31 downto 24);
             end case;

             x_val <= signed(sel_byte);

            ---------------------------------------------------
            -- Read w[i] from tensor_B:
            -- Each neuron has N weights.
            -- Weight index = neuron_cnt*N + elem_cnt
            ---------------------------------------------------

            word_idx := to_integer(elem_cnt) / 4;  --byte selecting
            byte_sel := to_integer(elem_cnt) mod 4;

            packed_word := tensor_B(word_idx);

            case byte_sel is
              when 0 => sel_byte := packed_word(7  downto 0);
              when 1 => sel_byte := packed_word(15 downto 8);
              when 2 => sel_byte := packed_word(23 downto 16);
              when others => sel_byte := packed_word(31 downto 24);
            end case;

            w_val <= signed(sel_byte);

            state <= S_DENSE_MAC;

            when S_DENSE_MAC =>
            
                prod := x_val * w_val; -- actual math being done in this function

                acc_tmp := acc_tmp + resize(prod, acc_tmp'length); -- trauma from variable timing incident
                
                mac_acc_reg <= acc_tmp; -- full accumulator

                if elem_cnt < (in_height - 1) then
                        -- more elements: i++
                        elem_cnt <= elem_cnt + 1;
                        state    <= S_DENSE_READ_IN;
                    else
                        -- all N elements done: go add bias
                        state    <= S_DENSE_BIAS;
                    end if;

            when S_DENSE_BIAS =>
            
                 -- Bias is in tensor_C, first byte (index 0)
--                 word_idx := 0;
--                 byte_sel := 0;

--                 packed_word := tensor_C(word_idx);  

--                case byte_sel is
--                  when 0 => sel_byte := packed_word(7  downto 0);
--                  when 1 => sel_byte := packed_word(15 downto 8);
--                  when 2 => sel_byte := packed_word(23 downto 16);
--                  when others => sel_byte := packed_word(31 downto 24);
--                end case;

--                bias_val := signed(sel_byte); -- gathered bias from c tensor

                acc_bias <= mac_acc_reg + resize(bias_val, mac_acc_reg'length); -- accumulate MAC and Bias

                state <= S_DENSE_WRITE;


            -- Quantize + write one output neuron
            when S_DENSE_WRITE =>
            
                q8 := resize(acc_bias, 8);  -- clamp/resize to int8
            
                -- Write this neuron's output into tensor_R index 0 for this run.
                -- SW will read / UART this and then reload next neuron's weights+bias.
                set_int8_into_word(tensor_R, 0, q8);
            
                state <= S_DONE;



          --Finalize: clear busy, set done, return to IDLE
          when S_DONE =>
            status_reg(0) <= '0';
            status_reg(1) <= '1';
            state <= S_IDLE;
        end case;
      end if;
    end if;
  end process;

  --Wishbone write path: decode addresses, update regs and tensor windows
  process(clk)
    variable tensor_offset: natural; --word offset inside a tensor window
  begin
    if rising_edge(clk) then
      if reset = '1' then
        ctrl_reg <= (others => '0');
        pool_base_index <= (others => '0');
        pool_out_index  <= (others => '0');
        word_index_reg  <= (others => '0');
        bus_dim_we <= '0';
        bus_dim_data <= (others => '0');
      elsif (i_wb_cyc = '1' and i_wb_stb = '1' and i_wb_we = '1') then
        bus_dim_we <= '0'; --default, set only when DIM is written

        if (i_wb_addr = CTRL_REG_ADDRESS) then
          ctrl_reg <= i_wb_data;
        elsif (i_wb_addr = DIM_REG_ADDRESS) then
          bus_dim_we   <= '1';
          bus_dim_data <= i_wb_data(7 downto 0);
          dim_height_width  <= i_wb_data(15 downto 0); -- NEW: Conv2D H,W
        elsif (i_wb_addr = POOL_BASE_INDEX_ADDRESS) then
          pool_base_index <= i_wb_data;
        elsif (i_wb_addr = POOL_OUT_INDEX_ADDRESS) then
          pool_out_index <= i_wb_data;
        elsif (i_wb_addr = WORD_INDEX_ADDRESS) then
          word_index_reg <= i_wb_data;

        --Tensor A window write
        elsif (unsigned(i_wb_addr) >= unsigned(TENSOR_A_BASE) and
               unsigned(i_wb_addr) <  unsigned(TENSOR_A_BASE) + (TENSOR_WORDS*4)) then
          tensor_offset := get_tensor_offset(i_wb_addr, TENSOR_A_BASE);
          if tensor_offset < TENSOR_WORDS then
            tensor_A(tensor_offset) <= i_wb_data;
          end if;

        --Tensor B window write
        elsif (unsigned(i_wb_addr) >= unsigned(TENSOR_B_BASE) and
               unsigned(i_wb_addr) <  unsigned(TENSOR_B_BASE) + (TENSOR_WORDS*4)) then
          tensor_offset := get_tensor_offset(i_wb_addr, TENSOR_B_BASE);
          if tensor_offset < TENSOR_WORDS then
            tensor_B(tensor_offset) <= i_wb_data;
          end if;

        --Tensor C window write
        elsif (unsigned(i_wb_addr) >= unsigned(TENSOR_C_BASE) and
               unsigned(i_wb_addr) <  unsigned(TENSOR_C_BASE) + (TENSOR_WORDS*4)) then
          tensor_offset := get_tensor_offset(i_wb_addr, TENSOR_C_BASE);
          if tensor_offset < TENSOR_WORDS then
            tensor_C(tensor_offset) <= i_wb_data;
          end if;
        end if;
      end if;
    end if;
  end process;

  --Wishbone read path: decode and mux back regs and tensor windows
  process(clk)
    variable tensor_offset: natural; --word offset inside a tensor window
  begin
    if rising_edge(clk) then
      if reset = '1' then
        data_r <= (others => '0');
      elsif (i_wb_cyc = '1' and i_wb_stb = '1' and i_wb_we = '0') then
        if (i_wb_addr = CTRL_REG_ADDRESS) then
          data_r <= ctrl_reg;
        elsif (i_wb_addr = STATUS_REG_ADDRESS) then
          data_r <= status_reg;
        elsif (i_wb_addr = DIM_REG_ADDRESS) then
          data_r <= (31 downto 16 => '0') & dim_height_width;
          
        elsif (i_wb_addr = POOL_BASE_INDEX_ADDRESS) then
          data_r <= pool_base_index;
        elsif (i_wb_addr = POOL_OUT_INDEX_ADDRESS) then
          data_r <= pool_out_index;

        --Tensor A window read
        elsif (unsigned(i_wb_addr) >= unsigned(TENSOR_A_BASE) and
               unsigned(i_wb_addr) <  unsigned(TENSOR_A_BASE) + (TENSOR_WORDS*4)) then
          tensor_offset := get_tensor_offset(i_wb_addr, TENSOR_A_BASE);
          if tensor_offset < TENSOR_WORDS then
            data_r <= tensor_A(tensor_offset);
          else
            data_r <= (others => '0');
          end if;

        --Tensor B/C window reads (similar to Tensor A)
        elsif (unsigned(i_wb_addr) >= unsigned(TENSOR_B_BASE) and
               unsigned(i_wb_addr) <  unsigned(TENSOR_B_BASE) + (TENSOR_WORDS*4)) then
          tensor_offset := get_tensor_offset(i_wb_addr, TENSOR_B_BASE);
          if tensor_offset < TENSOR_WORDS then
            data_r <= tensor_B(tensor_offset);
          else
            data_r <= (others => '0');
          end if;
        elsif (unsigned(i_wb_addr) >= unsigned(TENSOR_C_BASE) and
               unsigned(i_wb_addr) <  unsigned(TENSOR_C_BASE) + (TENSOR_WORDS*4)) then
          tensor_offset := get_tensor_offset(i_wb_addr, TENSOR_C_BASE);
          if tensor_offset < TENSOR_WORDS then
            data_r <= tensor_C(tensor_offset);
          else
            data_r <= (others => '0');
          end if;

        --Tensor R window read
        elsif (unsigned(i_wb_addr) >= unsigned(TENSOR_R_BASE) and
               unsigned(i_wb_addr) <  unsigned(TENSOR_R_BASE) + (TENSOR_WORDS*4)) then
          tensor_offset := get_tensor_offset(i_wb_addr, TENSOR_R_BASE);
          if tensor_offset < TENSOR_WORDS then
            data_r <= tensor_R(tensor_offset);
          else
            data_r <= (others => '0');
          end if;

        else
          data_r <= (others => '0');
        end if;
      end if;
    end if;
  end process;

  --Wishbone ACK generation: assert for valid mapped regions during active bus cycles
  process(clk)
    variable is_valid: std_ulogic; --address decode result
  begin
    if rising_edge(clk) then
      if reset = '1' then
        ack_r <= '0';
      else
        is_valid := '0';
        if (i_wb_addr = CTRL_REG_ADDRESS or
            i_wb_addr = STATUS_REG_ADDRESS or
            i_wb_addr = DIM_REG_ADDRESS or
            i_wb_addr = POOL_BASE_INDEX_ADDRESS or
            i_wb_addr = POOL_OUT_INDEX_ADDRESS or
            i_wb_addr = WORD_INDEX_ADDRESS or
            (unsigned(i_wb_addr) >= unsigned(TENSOR_A_BASE) and unsigned(i_wb_addr) < unsigned(TENSOR_A_BASE) + (TENSOR_WORDS*4)) or
            (unsigned(i_wb_addr) >= unsigned(TENSOR_B_BASE) and unsigned(i_wb_addr) < unsigned(TENSOR_B_BASE) + (TENSOR_WORDS*4)) or
            (unsigned(i_wb_addr) >= unsigned(TENSOR_C_BASE) and unsigned(i_wb_addr) < unsigned(TENSOR_C_BASE) + (TENSOR_WORDS*4)) or
            (unsigned(i_wb_addr) >= unsigned(TENSOR_R_BASE) and unsigned(i_wb_addr) < unsigned(TENSOR_R_BASE) + (TENSOR_WORDS*4))) then
          is_valid := '1';
        end if;

        if (i_wb_cyc = '1' and i_wb_stb = '1' and is_valid = '1') then
          ack_r <= '1';
        else
          ack_r <= '0';
        end if;
      end if;
    end if;
  end process;

  --Drive Wishbone outputs
  o_wb_ack  <= ack_r;
  o_wb_data <= data_r;
end architecture;
