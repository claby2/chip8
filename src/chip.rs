use crate::{
    error::{ChipError, ChipResult},
    pc::ProgramCounter,
    stack::Stack,
};
use rand::{rngs::ThreadRng, Rng};

// Screen has 2048 pixels (64 x 32).
pub const WIDTH: usize = 64;
pub const HEIGHT: usize = 32;
pub type Gfx = [u8; WIDTH * HEIGHT];

const STACK_SIZE: usize = 0x10;
const MEMORY_SIZE: usize = 0x1000;
const REGISTER_SIZE: usize = 0x10;

const KEYPAD_SIZE: usize = 0x10;
pub type Keypad = [u8; KEYPAD_SIZE];

const FONTSET: [u8; 80] = [
    0xF0, 0x90, 0x90, 0x90, 0xF0, // 0
    0x20, 0x60, 0x20, 0x20, 0x70, // 1
    0xF0, 0x10, 0xF0, 0x80, 0xF0, // 2
    0xF0, 0x10, 0xF0, 0x10, 0xF0, // 3
    0x90, 0x90, 0xF0, 0x10, 0x10, // 4
    0xF0, 0x80, 0xF0, 0x10, 0xF0, // 5
    0xF0, 0x80, 0xF0, 0x90, 0xF0, // 6
    0xF0, 0x10, 0x20, 0x40, 0x40, // 7
    0xF0, 0x90, 0xF0, 0x90, 0xF0, // 8
    0xF0, 0x90, 0xF0, 0x10, 0xF0, // 9
    0xF0, 0x90, 0xF0, 0x90, 0x90, // A
    0xE0, 0x90, 0xE0, 0x90, 0xE0, // B
    0xF0, 0x80, 0x80, 0x80, 0xF0, // C
    0xE0, 0x90, 0x90, 0x90, 0xE0, // D
    0xF0, 0x80, 0xF0, 0x80, 0xF0, // E
    0xF0, 0x80, 0xF0, 0x80, 0x80, // F
];

#[derive(Debug)]
pub struct Chip {
    // 35 opcodes. Each opcode is 2 bytes and store big-endian.
    opcode: u16,
    // 0x1000 memory locations.
    memory: [u8; MEMORY_SIZE],
    // CPU registers: V0, V1...VF. VF is the carry flag. Totals up to 16 8-bit registers.
    v: [u8; REGISTER_SIZE],
    index: usize,
    program_counter: ProgramCounter,
    gfx: Gfx,
    delay_timer: u8,
    sound_timer: u8,
    stack: Stack<u16, STACK_SIZE>,
    keypad: Keypad,
    draw: bool,
    rng: ThreadRng,
}

impl Chip {
    pub fn new() -> Self {
        // Clear memory.
        let mut memory: [u8; MEMORY_SIZE] = [0; MEMORY_SIZE];
        memory[..FONTSET.len()].clone_from_slice(&FONTSET[..FONTSET.len()]);
        // Reset registers.
        Self {
            opcode: 0,
            memory,
            v: [0; REGISTER_SIZE],
            index: 0,
            program_counter: ProgramCounter::default(),
            gfx: [0; WIDTH * HEIGHT],
            delay_timer: 0,
            sound_timer: 0,
            stack: Stack::default(),
            keypad: [0; KEYPAD_SIZE],
            draw: false,
            rng: rand::thread_rng(),
        }
    }

    pub fn load(mut self, buffer: &[u8]) -> ChipResult<Self> {
        let memory_offset = usize::from(ProgramCounter::DEFAULT_VALUE);
        self.memory[memory_offset..(buffer.len() + memory_offset)].clone_from_slice(buffer);
        Ok(self)
    }

    pub fn gfx(&self) -> Gfx {
        self.gfx
    }

    pub fn keypad(&mut self) -> &mut Keypad {
        &mut self.keypad
    }

    pub fn draw(&self) -> bool {
        self.draw
    }

    fn update_timer(&mut self) {
        if self.delay_timer > 0 {
            self.delay_timer -= 1;
        }
        if self.sound_timer > 0 {
            self.sound_timer -= 1;
        }
    }

    fn reset_keypad(&mut self) {
        self.keypad = [0; KEYPAD_SIZE];
    }

    // Emulate a single cycle.
    pub fn cycle(&mut self) -> ChipResult<()> {
        self.draw = false;
        // Fetch, Decode, and Execute one opcode.
        // Fetch: fetch opcode from memory in the location specified by program_counter.
        let memory_index = usize::from(self.program_counter.value());
        // Merge two adjacent bytes from memory.
        self.opcode =
            u16::from(self.memory[memory_index]) << 8 | u16::from(self.memory[memory_index + 1]);
        self.execute(self.opcode)?;
        self.update_timer();
        self.reset_keypad();
        Ok(())
    }

    fn execute(&mut self, opcode: u16) -> ChipResult<()> {
        // Initialize opcode arguments.
        let x = usize::from((opcode & 0x0F00) >> 8);
        let y = usize::from((opcode & 0x00F0) >> 4);
        let n = usize::from(opcode & 0x000F);
        let nn = (opcode & 0x00FF) as u8;
        let nnn = (opcode & 0x0FFF) as u16;
        // Decode: refer to opcode table: https://en.wikipedia.org/wiki/CHIP-8#Opcode_table
        match opcode & 0xF000 {
            0x0000 => match opcode & 0x00FF {
                0x00E0 => self.op_00e0(),
                0x00EE => self.op_00ee()?,
                _ => return Err(ChipError::UnknownOpcode(opcode)),
            },
            0x1000 => self.op_1nnn(nnn),
            0x2000 => self.op_2nnn(nnn)?,
            0x3000 => self.op_3xnn(x, nn),
            0x4000 => self.op_4xnn(x, nn),
            0x5000 => self.op_5xy0(x, y),
            0x6000 => self.op_6xnn(x, nn),
            0x7000 => self.op_7xnn(x, nn),
            0x8000 => match opcode & 0x000F {
                0x0000 => self.op_8xy0(x, y),
                0x0001 => self.op_8xy1(x, y),
                0x0002 => self.op_8xy2(x, y),
                0x0003 => self.op_8xy3(x, y),
                0x0004 => self.op_8xy4(x, y),
                0x0005 => self.op_8xy5(x, y),
                0x0006 => self.op_8x06(x),
                0x0007 => self.op_8xy7(x, y),
                0x000E => self.op_8x0e(x),
                _ => return Err(ChipError::UnknownOpcode(opcode)),
            },
            0x9000 => self.op_9xy0(x, y),
            0xA000 => self.op_annn(nnn),
            0xB000 => self.op_bnnn(nnn),
            0xC000 => self.op_cxnn(x, nn),
            0xD000 => self.op_dxyn(x, y, n),
            0xE000 => match opcode & 0x00FF {
                0x009E => self.op_ex9e(x),
                0x00A1 => self.op_exa1(x),
                _ => return Err(ChipError::UnknownOpcode(opcode)),
            },
            0xF000 => match opcode & 0x00FF {
                0x0007 => self.op_fx07(x),
                0x000A => self.op_fx0a(x),
                0x0015 => self.op_fx15(x),
                0x0018 => self.op_fx18(x),
                0x001E => self.op_fx1e(x),
                0x0029 => self.op_fx29(x),
                0x0033 => self.op_fx33(x),
                0x0055 => self.op_fx55(x),
                0x0065 => self.op_fx65(x),
                _ => return Err(ChipError::UnknownOpcode(opcode)),
            },
            _ => return Err(ChipError::UnknownOpcode(opcode)),
        };
        Ok(())
    }

    fn op_00e0(&mut self) {
        // Display. Clear the screen.
        for pixel in self.gfx.as_mut() {
            *pixel = 0x0
        }
        self.draw = true;
        self.program_counter.increment();
    }

    fn op_00ee(&mut self) -> ChipResult<()> {
        // Flow. Return from subroutine.
        self.program_counter.set(self.stack.pop()?);
        Ok(())
    }

    fn op_1nnn(&mut self, nnn: u16) {
        // Flow. 0x1NNN, jump to address NNN
        self.program_counter.set(nnn);
    }

    fn op_2nnn(&mut self, nnn: u16) -> ChipResult<()> {
        // Flow. 0x2NNN, calls subroutine at NNN.
        // Temporarily jump to address NNN, store program counter in stack and increment
        // stack pointer to prevent overwriting it.
        self.stack
            .push(self.program_counter.value() + ProgramCounter::INCREMENT)?;
        self.program_counter.set(nnn);
        Ok(())
    }

    fn op_3xnn(&mut self, x: usize, nn: u8) {
        // Cond. 0x3XNN, check if VX equals NN.
        self.program_counter.skip_if(self.v[x] == nn);
    }

    fn op_4xnn(&mut self, x: usize, nn: u8) {
        // Cond. 0x4XNN, check if VX does not equal NN.
        self.program_counter.skip_if(self.v[x] != nn);
    }

    fn op_5xy0(&mut self, x: usize, y: usize) {
        // Cond. 0x5XY0, check if VX equals VY.
        self.program_counter.skip_if(self.v[x] == self.v[y]);
    }

    fn op_6xnn(&mut self, x: usize, nn: u8) {
        // Const. 0x6XNN, set VX to NN.
        self.v[x] = nn;
        self.program_counter.increment();
    }

    fn op_7xnn(&mut self, x: usize, nn: u8) {
        // Const. 0x7XNN, add NN to VX.
        let result = (u16::from(self.v[x]) + u16::from(nn)) as u8;
        self.v[x] = result;
        self.program_counter.increment();
    }

    fn op_8xy0(&mut self, x: usize, y: usize) {
        // Assig. 0x8XY0, set VX to VY.
        self.v[x] = self.v[y];
        self.program_counter.increment();
    }

    fn op_8xy1(&mut self, x: usize, y: usize) {
        // BitOp. 0x8XY1, set VX to VX | VY.
        self.v[x] |= self.v[y];
        self.program_counter.increment();
    }

    fn op_8xy2(&mut self, x: usize, y: usize) {
        // BitOp. 0x8XY2, set VX to VX & VY.
        self.v[x] &= self.v[y];
        self.program_counter.increment();
    }

    fn op_8xy3(&mut self, x: usize, y: usize) {
        // BitOp. 0x8XY3, set VX to VX ^ VY.
        self.v[x] ^= self.v[y];
        self.program_counter.increment();
    }

    fn op_8xy4(&mut self, x: usize, y: usize) {
        // Math. 0x8XY4, adds the value of VY to VX.
        let result = u16::from(self.v[x]) + u16::from(self.v[y]);
        if result > 0xFF {
            // Flag carry due to overflow.
            self.v[0xF] = 1;
        } else {
            self.v[0xF] = 0;
        }
        self.v[x] = result as u8;
        self.program_counter.increment();
    }

    fn op_8xy5(&mut self, x: usize, y: usize) {
        // Math. 0x8XY5, set VX to VX - VY.
        if self.v[y] > self.v[x] {
            // Borrow.
            self.v[0xF] = 0;
        } else {
            self.v[0xF] = 1;
        }
        self.v[x] = self.v[x].wrapping_sub(self.v[y]);
        self.program_counter.increment();
    }

    fn op_8x06(&mut self, x: usize) {
        // BitOp. 0x8X06, store least significant bit of VX in VF.
        self.v[0xF] = self.v[x] & 0x1;
        // Shift VX rightwards by 1.
        self.v[x] >>= 1;
        self.program_counter.increment();
    }

    fn op_8xy7(&mut self, x: usize, y: usize) {
        // BitOp. 0x8XY7, set VX to VY - VX.
        if self.v[x] > self.v[y] {
            // Borrow.
            self.v[0xF] = 0;
        } else {
            self.v[0xF] = 1;
        }
        self.v[x] = self.v[y].wrapping_sub(self.v[x]);
        self.program_counter.increment();
    }

    fn op_8x0e(&mut self, x: usize) {
        // BitOp. 0x8X0E, store the most significant bit of VX in VF.
        self.v[0xF] = self.v[x] >> 7;
        // Shift VX leftwards by 1.
        self.v[x] <<= 1;
        self.program_counter.increment();
    }

    fn op_9xy0(&mut self, x: usize, y: usize) {
        // Cond. 0x9XY0, skip next iteration if VX != VY.
        self.program_counter.skip_if(self.v[x] != self.v[y]);
    }

    fn op_annn(&mut self, nnn: u16) {
        // MEM. 0xANNN, sets index to the address NNN.
        self.index = usize::from(nnn);
        self.program_counter.increment();
    }

    fn op_bnnn(&mut self, nnn: u16) {
        // Flow. 0xBNNN, jump to address V0 + NNN.
        self.program_counter.set(nnn + u16::from(self.v[0]));
    }

    fn op_cxnn(&mut self, x: usize, nn: u8) {
        // Rand. 0xCXNN, set VX to rand() & NN.
        self.v[x] = self.rng.gen::<u8>() & nn;
        self.program_counter.increment();
    }

    fn op_dxyn(&mut self, x: usize, y: usize, n: usize) {
        // Disp. 0xDXYN, draw coordinates (X, Y) with height N + 1.
        self.v[0xF] = 0;
        for byte in 0..n {
            let y = (self.v[y] as usize + byte) % HEIGHT;
            for bit in 0..8 {
                let x = (self.v[x] as usize + bit) % WIDTH;
                let color = (self.memory[self.index + byte] >> (7 - bit)) & 1;
                let gfx_index = (WIDTH * y) + x;
                self.v[0xF] |= color & self.gfx[gfx_index];
                self.gfx[gfx_index] ^= color;
            }
        }
        self.draw = true;
        self.program_counter.increment();
    }

    fn op_ex9e(&mut self, x: usize) {
        // KeyOp. 0xEX9E, skip next instruction if the key in VX is pressed.
        self.program_counter
            .skip_if(self.keypad[usize::from(self.v[x])] != 0);
    }

    fn op_exa1(&mut self, x: usize) {
        // KeyOp. 0xEXA1, skip next instruction if the key in VX is not pressed.
        self.program_counter
            .skip_if(self.keypad[usize::from(self.v[x])] == 0);
    }

    fn op_fx07(&mut self, x: usize) {
        // Timer. 0xFX07, set VX to delay_timer.
        self.v[x] = self.delay_timer;
        self.program_counter.increment();
    }

    fn op_fx0a(&mut self, x: usize) {
        // KeyOp. 0xFX0A, await key press and store in VX.
        if let Some(key) = self.keypad.iter().position(|&key| key != 0_u8) {
            self.v[x] = key as u8;
        } else {
            return;
        }
        self.program_counter.increment();
    }

    fn op_fx15(&mut self, x: usize) {
        // Timer. 0xFX15, set delay_timer to VX.
        self.delay_timer = self.v[x];
        self.program_counter.increment();
    }

    fn op_fx18(&mut self, x: usize) {
        // Sound. 0xFX18, set sound_timer to VX.
        self.sound_timer = self.v[x];
        self.program_counter.increment();
    }

    fn op_fx1e(&mut self, x: usize) {
        // MEM. 0xFX1E, add VX to index.
        // Overflow is not checked and VF should remain unaffected.
        self.index += usize::from(self.v[x]);
        self.program_counter.increment();
    }

    fn op_fx29(&mut self, x: usize) {
        // MEM. 0xFX29, set index to the location of the sprite character in VX.
        self.index = usize::from(self.v[x]) * 0x5;
        self.program_counter.increment();
    }

    fn op_fx33(&mut self, x: usize) {
        // BCD. 0xFX33, stores binary-coded decimal.
        self.memory[self.index] = self.v[x] / 100; // Hundreds digit.
        self.memory[self.index + 1] = (self.v[x] / 10) % 10; // Tens digit.
        self.memory[self.index + 2] = (self.v[x] % 100) % 10; // Ones digit.
        self.program_counter.increment();
    }

    fn op_fx55(&mut self, x: usize) {
        // MEM. 0xFX55, stores V0 to VX in memory starting at index.
        for i in 0..=x {
            self.memory[self.index + i] = self.v[i];
        }
        self.index += x + 1;
        self.program_counter.increment();
    }

    fn op_fx65(&mut self, x: usize) {
        // MEM. 0xFX65, fills V0 to VX from memory starting at index.
        for i in 0..=x {
            self.v[i] = self.memory[self.index + i];
        }
        self.index += x + 1;
        self.program_counter.increment();
    }
}

#[cfg(test)]
mod tests {
    use super::{Chip, ChipError, ProgramCounter, HEIGHT, KEYPAD_SIZE, REGISTER_SIZE, WIDTH};

    const PC_START: u16 = ProgramCounter::DEFAULT_VALUE;
    const PC_NEXT: u16 = ProgramCounter::DEFAULT_VALUE + ProgramCounter::INCREMENT;
    const PC_SKIP: u16 = ProgramCounter::DEFAULT_VALUE + ProgramCounter::SKIP;

    // Utility assertion function to ensure correctness of math operation.
    #[track_caller]
    fn assert_math(vx: u8, vy: u8, op: u16, result: u8, vf: u8) {
        let mut chip = Chip::new();
        chip.v[0] = vx;
        chip.v[1] = vy;
        chip.execute(0x8010 + op).unwrap();
        assert_eq!(chip.v[0], result);
        assert_eq!(chip.v[0xF], vf);
        assert_eq!(chip.program_counter.value(), PC_NEXT);
    }

    #[test]
    fn chip_new() {
        let chip = Chip::new();
        assert_eq!(chip.program_counter.value(), PC_START);
        assert_eq!(chip.v, [0; REGISTER_SIZE]);
        assert!(chip.stack.is_empty());
        assert_eq!(chip.keypad, [0; KEYPAD_SIZE]);
    }

    #[test]
    fn chip_load() {
        let buffer = [1, 2, 3];
        let chip = Chip::new().load(&buffer).unwrap();
        let memory_index = usize::from(ProgramCounter::DEFAULT_VALUE);
        assert_eq!(
            chip.memory[memory_index..memory_index + buffer.len()],
            buffer
        );
    }

    #[test]
    fn chip_op_00e0() {
        let mut chip = Chip::new();
        chip.gfx = [1; WIDTH * HEIGHT];
        chip.execute(0x00e0).unwrap();
        assert!(chip.gfx.iter().all(|pixel| *pixel == 0));
        assert_eq!(chip.program_counter.value(), PC_NEXT);
    }

    #[test]
    fn chip_invalid_opcode() {
        let mut chip = Chip::new();
        let result = chip.execute(0x0000);
        assert_err!(result, Err(ChipError::UnknownOpcode(0x0000)));
    }

    #[test]
    fn chip_op_00ee() {
        let mut chip = Chip::new();
        chip.stack.push(0x2222).unwrap();
        chip.execute(0x00ee).unwrap();
        assert_eq!(chip.program_counter.value(), 0x2222);
    }

    #[test]
    fn chip_op_1nnn() {
        let mut chip = Chip::new();
        chip.execute(0x1222).unwrap();
        assert_eq!(chip.program_counter.value(), 0x0222);
    }

    #[test]
    fn chip_op_2nnn() {
        let mut chip = Chip::new();
        chip.execute(0x2222).unwrap();
        assert_eq!(chip.program_counter.value(), 0x0222);
        assert_eq!(chip.stack[0], PC_NEXT);
    }

    #[test]
    fn chip_op_3xnn_skip() {
        let mut chip = Chip::new();
        chip.execute(0x3200).unwrap();
        assert_eq!(chip.program_counter.value(), PC_SKIP);
    }

    #[test]
    fn chip_op_3xnn_no_skip() {
        let mut chip = Chip::new();
        chip.execute(0x3201).unwrap();
        assert_eq!(chip.program_counter.value(), PC_NEXT);
    }

    #[test]
    fn chip_op_4xnn_skip() {
        let mut chip = Chip::new();
        chip.execute(0x4200).unwrap();
        assert_eq!(chip.program_counter.value(), PC_NEXT);
    }

    #[test]
    fn chip_op_4xnn_no_skip() {
        let mut chip = Chip::new();
        chip.execute(0x4201).unwrap();
        assert_eq!(chip.program_counter.value(), PC_SKIP);
    }

    #[test]
    fn chip_op_5xy0_skip() {
        let mut chip = Chip::new();
        chip.execute(0x5010).unwrap();
        assert_eq!(chip.program_counter.value(), PC_SKIP);
    }

    #[test]
    fn chip_op_5xy0_no_skip() {
        let mut chip = Chip::new();
        chip.v[2] = 1;
        chip.execute(0x5120).unwrap();
        assert_eq!(chip.program_counter.value(), PC_NEXT);
    }

    #[test]
    fn chip_op_6xnn() {
        let mut chip = Chip::new();
        chip.execute(0x65FF).unwrap();
        assert_eq!(chip.v[5], 0xFF);
        assert_eq!(chip.program_counter.value(), PC_NEXT);
    }

    #[test]
    fn chip_op_7xnn() {
        let mut chip = Chip::new();
        chip.v[5] = 1;
        chip.execute(0x75F0).unwrap();
        assert_eq!(chip.v[5], 0xF1);
        assert_eq!(chip.program_counter.value(), PC_NEXT);
    }

    #[test]
    fn chip_op_8xy0() {
        let mut chip = Chip::new();
        chip.v[1] = 1;
        chip.execute(0x8010).unwrap();
        assert_eq!(chip.v[0], chip.v[1]);
        assert_eq!(chip.program_counter.value(), PC_NEXT);
    }

    #[test]
    fn chip_op_8xy1() {
        // OR.
        assert_math(0x0F, 0xF0, 1, 0xFF, 0);
    }

    #[test]
    fn chip_op_8xy2() {
        // AND.
        assert_math(0x0F, 0xFF, 2, 0x0F, 0);
    }

    #[test]
    fn chip_op_8xy3() {
        // XOR.
        assert_math(0x0F, 0xFF, 3, 0xF0, 0);
    }

    #[test]
    fn chip_op_8xy4() {
        // ADD.
        assert_math(0x0F, 0x0F, 4, 0x1E, 0);
        assert_math(0xFF, 0xFF, 4, 0xFE, 1);
    }

    #[test]
    fn chip_op_8xy5() {
        // SUB. VX - VY.
        assert_math(0x0F, 0x01, 5, 0x0E, 1);
        assert_math(0x0F, 0xFF, 5, 0x10, 0);
    }

    #[test]
    fn chip_op_8x06() {
        // >>.
        assert_math(0x04, 0, 6, 0x02, 0);
        assert_math(0x05, 0, 6, 0x02, 1);
    }

    #[test]
    fn chip_op_8xy7() {
        // SUB. VY - VX.
        assert_math(0x01, 0x0F, 7, 0x0E, 1);
        assert_math(0xFF, 0x0F, 7, 0x10, 0);
    }

    #[test]
    fn chip_op_8x0e() {
        // <<.
        assert_math(0xC0, 0, 0x0E, 0x80, 1);
        assert_math(0x07, 0, 0x0E, 0x0E, 0);
    }

    #[test]
    fn chip_op_9xy0_skip() {
        let mut chip = Chip::new();
        chip.v[1] = 1;
        chip.execute(0x9010).unwrap();
        assert_eq!(chip.program_counter.value(), PC_SKIP);
    }

    #[test]
    fn chip_op_9xy0_no_skip() {
        let mut chip = Chip::new();
        chip.execute(0x9010).unwrap();
        assert_eq!(chip.program_counter.value(), PC_NEXT);
    }

    #[test]
    fn chip_op_annn() {
        let mut chip = Chip::new();
        chip.execute(0xA123).unwrap();
        assert_eq!(chip.index, 0x123);
        assert_eq!(chip.program_counter.value(), PC_NEXT);
    }

    #[test]
    fn chip_op_bnnn() {
        let mut chip = Chip::new();
        chip.v[0] = 1;
        chip.execute(0xB123).unwrap();
        assert_eq!(chip.program_counter.value(), 0x124);
    }

    #[test]
    fn chip_op_cxnn() {
        let mut chip = Chip::new();
        chip.execute(0xC000).unwrap();
        assert_eq!(chip.v[0], 0);
        assert_eq!(chip.program_counter.value(), PC_NEXT);
        chip.execute(0xC00F).unwrap();
        assert_eq!(chip.v[0] & 0xF0, 0);
        // Since chip.execute has been called twice, PC value should be equivalent to PC_SKIP.
        assert_eq!(chip.program_counter.value(), PC_SKIP);
    }

    #[test]
    fn chip_op_dxyn() {
        let mut chip = Chip::new();
        chip.index = 0;
        chip.memory[0] = 0xFF;
        chip.memory[1] = 0x00;
        chip.gfx[0] = 1;
        chip.gfx[1] = 0;
        chip.gfx[WIDTH] = 1;
        chip.gfx[WIDTH + 1] = 0;
        chip.execute(0xD002).unwrap();
        assert_eq!(chip.gfx[0], 0);
        assert_eq!(chip.gfx[1], 1);
        assert_eq!(chip.gfx[WIDTH], 1);
        assert_eq!(chip.gfx[WIDTH + 1], 0);
        assert_eq!(chip.v[0x0F], 1);
        assert_eq!(chip.program_counter.value(), PC_NEXT);
    }

    #[test]
    fn chip_op_dxyn_horizontal() {
        let mut chip = Chip::new();
        let x = WIDTH - 4;
        chip.index = 0;
        chip.memory[0] = 0xFF;
        chip.v[0] = x as u8;
        chip.execute(0xD011).unwrap();

        assert_eq!(chip.gfx[x - 1..=x + 3], [0, 1, 1, 1, 1]);
        assert_eq!(chip.gfx[0..=4], [1, 1, 1, 1, 0]);
        assert_eq!(chip.v[0x0F], 0);
        assert_eq!(chip.program_counter.value(), PC_NEXT);
    }

    #[test]
    fn chip_op_dxyn_vertical() {
        let mut chip = Chip::new();
        let y = HEIGHT - 1;
        chip.index = 0;
        chip.memory[0] = 0xFF;
        chip.memory[1] = 0xFF;
        chip.v[1] = y as u8;
        chip.execute(0xD012).unwrap();
        assert_eq!(chip.gfx[y * WIDTH], 1);
        assert_eq!(chip.gfx[0], 1);
        assert_eq!(chip.v[0xF], 0);
        assert_eq!(chip.program_counter.value(), PC_NEXT);
    }

    #[test]
    fn chip_op_ex9e_skip() {
        let mut chip = Chip::new();
        let key: u8 = 2;
        chip.keypad[usize::from(key)] = 1;
        chip.v[0] = key;
        chip.execute(0xE09E).unwrap();
        assert_eq!(chip.program_counter.value(), PC_SKIP);
    }

    #[test]
    fn chip_op_ex9e_no_skip() {
        let mut chip = Chip::new();
        let key: u8 = 2;
        chip.keypad[usize::from(key)] = 0;
        chip.v[0] = key;
        chip.execute(0xE09E).unwrap();
        assert_eq!(chip.program_counter.value(), PC_NEXT);
    }

    #[test]
    fn chip_op_exa1_skip() {
        let mut chip = Chip::new();
        let key: u8 = 2;
        chip.keypad[usize::from(key)] = 0;
        chip.v[0] = key;
        chip.execute(0xE0A1).unwrap();
        assert_eq!(chip.program_counter.value(), PC_SKIP);
    }

    #[test]
    fn chip_op_exa1_no_skip() {
        let mut chip = Chip::new();
        let key: u8 = 2;
        chip.keypad[usize::from(key)] = 1;
        chip.v[0] = key;
        chip.execute(0xE0A1).unwrap();
        assert_eq!(chip.program_counter.value(), PC_NEXT);
    }

    #[test]
    fn chip_op_fx07() {
        let mut chip = Chip::new();
        chip.delay_timer = 10;
        chip.execute(0xF007).unwrap();
        assert_eq!(chip.v[0], 10);
        assert_eq!(chip.program_counter.value(), PC_NEXT);
    }

    #[test]
    fn chip_op_fx0a_key() {
        let mut chip = Chip::new();
        let key: u8 = 2;
        chip.keypad[usize::from(key)] = 1;
        chip.execute(0xF00A).unwrap();
        assert_eq!(chip.v[0], key);
        assert_eq!(chip.program_counter.value(), PC_NEXT);
    }

    #[test]
    fn chip_op_fx0a_no_key() {
        let mut chip = Chip::new();
        chip.execute(0xF00A).unwrap();
        assert_eq!(chip.v[0], 0);
        // No key is active so program counter should not increment.
        assert_eq!(chip.program_counter.value(), PC_START);
    }

    #[test]
    fn chip_op_fx15() {
        let mut chip = Chip::new();
        let timer_value = 2;
        chip.v[1] = timer_value;
        chip.execute(0xF115).unwrap();
        assert_eq!(chip.delay_timer, timer_value);
        assert_eq!(chip.program_counter.value(), PC_NEXT);
    }

    #[test]
    fn chip_op_fx18() {
        let mut chip = Chip::new();
        let timer_value = 2;
        chip.v[1] = timer_value;
        chip.execute(0xF118).unwrap();
        assert_eq!(chip.sound_timer, timer_value);
        assert_eq!(chip.program_counter.value(), PC_NEXT);
    }

    #[test]
    fn chip_op_fx1e() {
        let mut chip = Chip::new();
        chip.index = 0xFF;
        chip.v[0] = 0xFF;
        chip.execute(0xF01E).unwrap();
        assert_eq!(chip.index, 0x1FE);
        assert_eq!(chip.program_counter.value(), PC_NEXT);
    }

    #[test]
    fn chip_op_fx29() {
        let mut chip = Chip::new();
        chip.v[0] = 2;
        chip.execute(0xF029).unwrap();
        assert_eq!(chip.index, 2 * 5);
        assert_eq!(chip.program_counter.value(), PC_NEXT);
    }

    #[test]
    fn chip_op_fx33() {
        let mut chip = Chip::new();
        chip.v[0] = 123;
        let index = 100;
        chip.index = index;
        chip.execute(0xF033).unwrap();
        assert_eq!(chip.memory[index..=index + 2], [1, 2, 3]);
        assert_eq!(chip.program_counter.value(), PC_NEXT);
    }

    #[test]
    fn chip_op_fx55() {
        let mut chip = Chip::new();
        // Set v as 1, 2 ... 16
        for i in 1..=chip.v.len() {
            chip.v[i - 1] = i as u8;
        }
        let index = 100;
        chip.index = index;
        // Store all registers.
        chip.execute(0xFF55).unwrap();
        for i in 0..chip.v.len() {
            assert_eq!(chip.memory[index + i], chip.v[i]);
        }
        assert_eq!(chip.program_counter.value(), PC_NEXT);
    }

    #[test]
    fn chip_op_fx65() {
        let mut chip = Chip::new();
        let index = 100;
        chip.index = 100;
        for i in 1..=chip.v.len() {
            chip.memory[index + i - 1] = i as u8;
        }
        chip.execute(0xFF65).unwrap();
        for i in 0..chip.v.len() {
            assert_eq!(chip.v[i], chip.memory[index + i]);
        }
        assert_eq!(chip.program_counter.value(), PC_NEXT);
    }
}
