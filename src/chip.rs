use crate::{
    error::{ChipError, ChipResult},
    pc::ProgramCounter,
};
use rand::{rngs::ThreadRng, Rng};
use std::{
    fs::{self, File},
    io::Read,
};

// Screen has 2048 pixels (64 x 32).
pub const WIDTH: usize = 64;
pub const HEIGHT: usize = 32;
pub type Gfx = [u8; WIDTH * HEIGHT];

const MEMORY_SIZE: usize = 0x1000;
const REGISTER_SIZE: usize = 0x10;
const STACK_SIZE: usize = 0x10;
const KEY_SIZE: usize = 0x10;

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
    stack: [u16; STACK_SIZE],
    stack_pointer: usize,
    key: [u8; KEY_SIZE],
    draw: bool,
    rng: ThreadRng,
}

impl Chip {
    pub fn initialize(path: &str) -> ChipResult<Self> {
        // Clear memory.
        let mut memory: [u8; MEMORY_SIZE] = [0; MEMORY_SIZE];
        memory[..FONTSET.len()].clone_from_slice(&FONTSET[..FONTSET.len()]);
        let mut file = File::open(&path)?;
        let mut buffer = vec![0; fs::metadata(&path)?.len() as usize];
        file.read_exact(&mut buffer)?;
        memory[0x200..(buffer.len() + 0x200)].clone_from_slice(&buffer[..]);
        // Reset registers.
        Ok(Self {
            opcode: 0,
            memory,
            v: [0; REGISTER_SIZE],
            index: 0,
            // Begin at memory location 512 (0x200).
            // Most systems do not access any memory below this.
            program_counter: ProgramCounter::default(),
            gfx: [0; WIDTH * HEIGHT],
            delay_timer: 0,
            sound_timer: 0,
            stack: [0; STACK_SIZE],
            stack_pointer: 0,
            key: [0; KEY_SIZE],
            draw: false,
            rng: rand::thread_rng(),
        })
    }

    pub fn gfx(&self) -> Gfx {
        self.gfx
    }

    pub fn draw(&self) -> bool {
        self.draw
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
        // Initialize opcode arguments.
        let x = usize::from((self.opcode & 0x0F00) >> 8);
        let y = usize::from((self.opcode & 0x00F0) >> 4);
        let n = usize::from(self.opcode & 0x000F);
        let nn = (self.opcode & 0x00FF) as u8;
        let nnn = (self.opcode & 0x0FFF) as u16;
        // Decode: refer to opcode table: https://en.wikipedia.org/wiki/CHIP-8#Opcode_table
        match self.opcode & 0xF000 {
            0x0000 => match self.opcode & 0x00FF {
                0x00E0 => self.op_00e0(),
                0x00EE => self.op_00ee(),
                _ => return Err(ChipError::UnknownOpcode(self.opcode)),
            },
            0x1000 => self.op_1nnn(nnn),
            0x2000 => self.op_2nnn(nnn),
            0x3000 => self.op_3xnn(x, nn),
            0x4000 => self.op_4xnn(x, nn),
            0x5000 => self.op_5xy0(x, y),
            0x6000 => self.op_6xnn(x, nn),
            0x7000 => self.op_7xnn(x, nn),
            0x8000 => match self.opcode & 0x000F {
                0x0000 => self.op_8xy0(x, y),
                0x0001 => self.op_8xy1(x, y),
                0x0002 => self.op_8xy2(x, y),
                0x0003 => self.op_8xy3(x, y),
                0x0004 => self.op_8xy4(x, y),
                0x0005 => self.op_8xy5(x, y),
                0x0006 => self.op_8x06(x),
                0x0007 => self.op_8xy7(x, y),
                0x000E => self.op_8x0e(x),
                _ => return Err(ChipError::UnknownOpcode(self.opcode)),
            },
            0x9000 => self.op_9xy0(x, y),
            0xA000 => self.op_annn(nnn),
            0xB000 => self.op_bnnn(nnn),
            0xC000 => self.op_cxnn(x, nn),
            0xD000 => self.op_dxyn(x, y, n),
            0xE000 => match self.opcode & 0x00FF {
                0x009E => self.op_ex9e(x),
                0x00A1 => self.op_exa1(x),
                _ => return Err(ChipError::UnknownOpcode(self.opcode)),
            },
            0xF000 => match self.opcode & 0x00FF {
                0x0007 => self.op_fx07(x),
                0x000A => self.op_fx0a(x),
                0x0015 => self.op_fx15(x),
                0x0018 => self.op_fx18(x),
                0x001E => self.op_fx1e(x),
                0x0029 => self.op_fx29(x),
                0x0033 => self.op_fx33(x),
                0x0055 => self.op_fx55(x),
                0x0065 => self.op_fx65(x),
                _ => return Err(ChipError::UnknownOpcode(self.opcode)),
            },
            _ => return Err(ChipError::UnknownOpcode(self.opcode)),
        }
        self.update_timer();
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

    fn op_00ee(&mut self) {
        // Flow. Return from subroutine.
        self.stack_pointer -= 1;
        self.program_counter.set(self.stack[self.stack_pointer]);
        // TODO: Verify if program_counter increment is necessary.
        self.program_counter.increment();
    }

    fn op_1nnn(&mut self, nnn: u16) {
        // Flow. 0x1NNN, jump to address NNN
        self.program_counter.set(nnn);
    }

    fn op_2nnn(&mut self, nnn: u16) {
        // Flow. 0x2NNN, calls subroutine at NNN.
        // Temporarily jump to address NNN, store program counter in stack and increment
        // stack pointer to prevent overwriting it.
        self.stack[self.stack_pointer] = self.program_counter.value();
        self.stack_pointer += 1;
        self.program_counter.set(nnn);
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
        self.v[x] = self.v[y].wrapping_sub(self.v[x]);
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
        self.v[x] = self.v[y] - self.v[x];
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
        for y_line in 0..n {
            let pixel = self.memory[self.index + y_line];
            // 8 is fixed constant by definition of opcode.
            for x_line in 0..8 {
                // Check if the current pixel is set to 1.
                if (pixel & (0x80 >> x_line)) != 0 {
                    let gfx_index = x + x_line + ((y + y_line) * 0x40);
                    if self.gfx[gfx_index] == 1 {
                        self.v[0xF] = 1;
                    }
                    self.gfx[gfx_index] ^= 1;
                }
            }
        }
        self.draw = true;
        self.program_counter.increment();
    }

    fn op_ex9e(&mut self, x: usize) {
        // KeyOp. 0xEX9E, skip next instruction if the key in VX is pressed.
        self.program_counter
            .skip_if(self.key[usize::from(self.v[x])] != 0);
    }

    fn op_exa1(&mut self, x: usize) {
        // KeyOp. 0xEXA1, skip next instruction if the key in VX is not pressed.
        self.program_counter
            .skip_if(self.key[usize::from(self.v[x])] == 0);
    }

    fn op_fx07(&mut self, x: usize) {
        // Timer. 0xFX07, set VX to delay_timer.
        self.v[x] = self.delay_timer;
        self.program_counter.increment();
    }

    fn op_fx0a(&mut self, x: usize) {
        // KeyOp. 0xFX0A, await key press and store in VX.
        if let Some(key) = self.key.iter().find(|key| **key != 0_u8) {
            self.v[x] = *key;
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
        // TODO: Verify if 0xFX1E affects VF on overflow.
        let result = usize::from(self.v[x]) + self.index;
        if result > 0xFFF {
            // Flag carry due to overflow.
            self.v[0xF] = 1;
        } else {
            self.v[0xF] = 0;
        }
        self.index = result;
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

    fn update_timer(&mut self) {
        if self.delay_timer > 0 {
            self.delay_timer -= 1;
        }
        if self.sound_timer > 0 {
            self.sound_timer -= 1;
        }
    }
}
