use crate::gpu::{Gpu, Primitive};
use crate::{elf, Memory, Store};
use byteorder::{ByteOrder, LittleEndian};

pub const RAM_BASE: u32 = 0x80000000;
const RAM_SIZE: u32 = RAM_BASE + 0x100000;
const GPU_BASE: u32 = 0x60000000;
const GPU_SIZE: u32 = GPU_BASE + 0x1;

#[derive(Debug)]
enum Format {
    R,
    I,
    IL,
    S,
    B,
    J,
    JI,
    U,
    UI,
    E,
}

enum Opcode {
    Add,
    Sub,
    Xor,
    Or,
    And,
    Sll,
    Srl,
    Sra,
    Slt,
    Sltu,
    Addi,
    Xori,
    Ori,
    Andi,
    Slli,
    Srli,
    Srai,
    Slti,
    Sltiu,
    Lb,
    Lh,
    Lw,
    Lbu,
    Lhu,
    Sb,
    Sh,
    Sw,
    Beq,
    Bne,
    Blt,
    Bge,
    Bltu,
    Bgeu,
    Jal,
    Jalr,
    Lui,
    Auipc,
    Ecall,
    Ebreak,
}

struct Ram {
    mem: Vec<u8>,
}

struct Bus {
    ram: Box<dyn Memory>,
    gpu: Gpu,
}

pub struct Cpu {
    pub reg: [u32; 31],
    pub pc: u32,
    bus: Bus,
}

struct Instruction {
    op: Format,
    rd: u8,
    f3: u8,
    f7: u8,
    rs1: u8,
    rs2: u8,
    imm: u32,
    simm: u32,
    uimm: u32,
    bimm: u32,
    jimm: u32,
}

impl Ram {
    fn new() -> Self {
        let mut mem = Vec::new();
        mem.resize((RAM_SIZE - RAM_BASE) as usize, 0);
        Self { mem }
    }
}

impl Memory for Ram {
    fn read(&self, addr: u32) -> u32 {
        let addr = addr as usize;
        // should check for unaligned reads
        LittleEndian::read_u32(&self.mem[addr..addr + 4])
    }

    fn write(&mut self, addr: u32, val: Store) {
        let addr = addr as usize;

        match val {
            Store::Byte(b) => self.mem[addr] = b,
            Store::Half(h) => LittleEndian::write_u16(&mut self.mem[addr..addr + 2], h),
            Store::Word(w) => LittleEndian::write_u32(&mut self.mem[addr..addr + 4], w),
        }
    }
}

impl Format {
    fn new(ins: u32) -> Self {
        match ins & 0x7f {
            0x33 => Self::R,
            0x13 => Self::I,
            0x03 => Self::IL,
            0x23 => Self::S,
            0x63 => Self::B,
            0x6f => Self::J,
            0x67 => Self::JI,
            0x37 => Self::U,
            0x17 => Self::UI,
            0x73 => Self::E,
            _ => panic!(),
        }
    }
}

macro_rules! op {
    ($op: ident) => {
        Instruction {
            op: Format::$op,
            ..
        }
    };
}

macro_rules! f3f7 {
    ($op: ident, $f3: expr, $f7: expr) => {
        Instruction {
            op: Format::$op,
            f3: $f3,
            f7: $f7,
            ..
        }
    };
}

macro_rules! f3 {
    ($op: ident, $f3: expr) => {
        Instruction {
            op: Format::$op,
            f3: $f3,
            ..
        }
    };
}

impl Opcode {
    fn new(ins: &Instruction) -> Self {
        match ins {
            f3f7!(R, 0x00, 0x00) => Self::Add,
            f3f7!(R, 0x00, 0x20) => Self::Sub,
            f3f7!(R, 0x04, 0x00) => Self::Xor,
            f3f7!(R, 0x06, 0x00) => Self::Or,
            f3f7!(R, 0x07, 0x00) => Self::And,
            f3f7!(R, 0x01, 0x00) => Self::Sll,
            f3f7!(R, 0x05, 0x00) => Self::Srl,
            f3f7!(R, 0x05, 0x20) => Self::Sra,
            f3f7!(R, 0x02, 0x00) => Self::Slt,
            f3f7!(R, 0x03, 0x00) => Self::Sltu,
            f3!(I, 0x00) => Self::Addi,
            f3!(I, 0x04) => Self::Xori,
            f3!(I, 0x06) => Self::Ori,
            f3!(I, 0x07) => Self::Andi,
            f3!(I, 0x01) => Self::Slli,
            f3f7!(I, 0x05, 0x00) => Self::Srli,
            f3f7!(I, 0x05, 0x20) => Self::Srai,
            f3!(I, 0x02) => Self::Slti,
            f3!(I, 0x03) => Self::Sltiu,
            f3!(IL, 0x00) => Self::Lb,
            f3!(IL, 0x01) => Self::Lh,
            f3!(IL, 0x02) => Self::Lw,
            f3!(IL, 0x04) => Self::Lbu,
            f3!(IL, 0x05) => Self::Lhu,
            f3!(S, 0x00) => Self::Sb,
            f3!(S, 0x01) => Self::Sh,
            f3!(S, 0x02) => Self::Sw,
            f3!(B, 0x00) => Self::Beq,
            f3!(B, 0x01) => Self::Bne,
            f3!(B, 0x04) => Self::Blt,
            f3!(B, 0x05) => Self::Bge,
            f3!(B, 0x06) => Self::Bltu,
            f3!(B, 0x07) => Self::Bgeu,
            op!(J) => Self::Jal,
            f3!(JI, 0x0) => Self::Jalr,
            op!(U) => Self::Lui,
            op!(UI) => Self::Auipc,
            op!(E) => match ins.imm {
                0x0 => Self::Ecall,
                _ => Self::Ebreak,
            },
            _ => panic!("bad instruction"),
        }
    }
}

impl Instruction {
    fn new(ins: u32) -> Self {
        Self {
            op: Format::new(ins),
            rd: ((ins >> 7) & 0x1f) as u8,
            f3: ((ins >> 12) & 0x7) as u8,
            f7: ((ins >> 25) & 0x7f) as u8,
            rs1: ((ins >> 15) & 0x1f) as u8,
            rs2: ((ins >> 20) & 0x1f) as u8,
            imm: ((ins & 0xfff00000) as i64 as i32 >> 20) as u32,
            simm: (((ins & 0xfe000000) as i64 as i32 >> 20) as u32 | ((ins >> 7) & 0x1f)),
            uimm: ins >> 12,
            bimm: ((ins & 0x80000000) as i64 as i32 >> 19) as u32
                | ((ins & 0x80) << 4)
                | ((ins >> 20) & 0x7e0)
                | ((ins >> 7) & 0x1e),
            jimm: ((ins & 0x80000000) as i64 as i32 >> 11) as u32
                | (ins & 0xff000)
                | ((ins >> 9) & 0x800)
                | ((ins >> 20) & 0x7fe),
        }
    }

    fn exec_r(&self, cpu: &mut Cpu, op: fn(u32, u32) -> u32) {
        let rs1 = cpu.reg_read(self.rs1);
        let rs2 = cpu.reg_read(self.rs2);
        cpu.reg_write(self.rd, op(rs1, rs2));
    }

    fn exec_i(&self, cpu: &mut Cpu, op: fn(u32, u32) -> u32) {
        let rs1 = cpu.reg_read(self.rs1);
        cpu.reg_write(self.rd, op(rs1, self.imm));
    }

    fn exec_il(&self, cpu: &mut Cpu, op: fn(u32) -> u32) {
        let rs1 = cpu.reg_read(self.rs1);
        let load = cpu.read(rs1.wrapping_add_signed(self.imm as i32));
        cpu.reg_write(self.rd, op(load));
    }

    fn exec_s(&self, cpu: &mut Cpu, op: fn(u32) -> Store) {
        let rs1 = cpu.reg_read(self.rs1);
        let rs2 = cpu.reg_read(self.rs2);
        cpu.write(rs1.wrapping_add_signed(self.simm as i32), op(rs2))
    }

    fn exec_b(&self, cpu: &mut Cpu, op: fn(u32, u32) -> bool) {
        let rs1 = cpu.reg_read(self.rs1);
        let rs2 = cpu.reg_read(self.rs2);
        if op(rs1, rs2) {
            cpu.pc = cpu.pc.wrapping_add_signed(self.bimm as i32) - 4;
        }
    }

    fn execute(&self, cpu: &mut Cpu) {
        match Opcode::new(self) {
            Opcode::Add => self.exec_r(cpu, |r1, r2| r1.wrapping_add_signed(r2 as i32)),
            Opcode::Sub => self.exec_r(cpu, |r1, r2| r1.wrapping_sub(r2)),
            Opcode::Xor => self.exec_r(cpu, |r1, r2| r1 ^ r2),
            Opcode::Or => self.exec_r(cpu, |r1, r2| r1 | r2),
            Opcode::And => self.exec_r(cpu, |r1, r2| r1 & r2),
            Opcode::Sll => self.exec_r(cpu, |r1, r2| r1.wrapping_shl(r2)),
            Opcode::Srl => self.exec_r(cpu, |r1, r2| r1.wrapping_shr(r2)),
            Opcode::Sra => self.exec_r(cpu, |r1, r2| r1.wrapping_shr(r2)),
            Opcode::Slt => self.exec_r(cpu, |r1, r2| (r1 < (r2 as i32) as u32) as u32),
            Opcode::Sltu => self.exec_r(cpu, |r1, r2| (r1 < r2) as u32),

            Opcode::Addi => self.exec_i(cpu, |r1, imm| r1.wrapping_add_signed(imm as i32)),
            Opcode::Xori => self.exec_i(cpu, |r1, imm| r1 ^ imm),
            Opcode::Ori => self.exec_i(cpu, |r1, imm| r1 | imm),
            Opcode::Andi => self.exec_i(cpu, |r1, imm| r1 & imm),
            Opcode::Slli => self.exec_i(cpu, |r1, imm| r1.wrapping_shl(imm & 0x1f)),
            Opcode::Srli => self.exec_i(cpu, |r1, imm| r1.wrapping_shr(imm & 0x1f)),
            Opcode::Srai => self.exec_i(cpu, |r1, imm| r1.wrapping_shr(imm & 0x1f) as i32 as u32),
            Opcode::Slti => self.exec_i(cpu, |r1, imm| (r1 < (imm as i32) as u32) as u32),
            Opcode::Sltiu => self.exec_i(cpu, |r1, imm| (r1 < imm) as u32),

            Opcode::Lb => self.exec_il(cpu, |imm| imm as u8 as u32),
            Opcode::Lh => self.exec_il(cpu, |imm| imm as u16 as u32),
            Opcode::Lw => self.exec_il(cpu, |imm| imm),
            Opcode::Lbu => self.exec_il(cpu, |imm| imm as u8 as u32),
            Opcode::Lhu => self.exec_il(cpu, |imm| imm as u16 as u32),

            Opcode::Sb => self.exec_s(cpu, |r1| Store::Byte(r1 as u8)),
            Opcode::Sh => self.exec_s(cpu, |r1| Store::Half(r1 as u16)),
            Opcode::Sw => self.exec_s(cpu, Store::Word),

            Opcode::Beq => self.exec_b(cpu, |r1, r2| (r1 as i32) == (r2 as i32)),
            Opcode::Bne => self.exec_b(cpu, |r1, r2| (r1 as i32) != (r2 as i32)),
            Opcode::Blt => self.exec_b(cpu, |r1, r2| (r1 as i32) < (r2 as i32)),
            Opcode::Bge => self.exec_b(cpu, |r1, r2| (r1 as i32) >= (r2 as i32)),
            Opcode::Bltu => self.exec_b(cpu, |r1, r2| r1 < r2),
            Opcode::Bgeu => self.exec_b(cpu, |r1, r2| r1 >= r2),

            Opcode::Jal => {
                cpu.reg_write(self.rd, cpu.pc);
                cpu.pc = cpu.pc.wrapping_add_signed(self.jimm as i32) - 4;
            }
            Opcode::Jalr => {
                let pc = cpu.pc;
                let rs1 = cpu.reg_read(self.rs1);
                cpu.pc = rs1.wrapping_add_signed(self.imm as i32);
                cpu.reg_write(self.rd, pc);
            }

            Opcode::Lui => cpu.reg_write(self.rd, self.uimm << 12),
            Opcode::Auipc => cpu.reg_write(self.rd, cpu.pc.wrapping_add(self.uimm << 12) - 4),

            Opcode::Ecall | Opcode::Ebreak => (),
        }
    }
}

impl Memory for Bus {
    fn read(&self, addr: u32) -> u32 {
        match addr {
            RAM_BASE..=RAM_SIZE => self.ram.read(addr - RAM_BASE),
            GPU_BASE..=GPU_SIZE => self.gpu.read(addr - GPU_BASE),
            _ => panic!(),
        }
    }

    fn write(&mut self, addr: u32, val: Store) {
        match addr {
            RAM_BASE..=RAM_SIZE => self.ram.write(addr - RAM_BASE, val),
            GPU_BASE..=GPU_SIZE => self.gpu.write(addr - GPU_BASE, val),
            _ => panic!(),
        }
    }
}

impl Cpu {
    pub fn new(mem: Vec<u8>) -> Result<Self, elf::Error> {
        let mut reg: [u32; 31] = [0; 31];
        let mut ram = Ram::new();

        // set sp to point to top of ram
        reg[2] = RAM_SIZE;

        let pc = elf::load(mem, &mut ram.mem)?;

        Ok(Self {
            reg,
            pc,
            bus: Bus {
                ram: Box::new(ram),
                gpu: Gpu::new(),
            },
        })
    }

    pub fn read(&self, addr: u32) -> u32 {
        self.bus.read(addr)
    }

    fn write(&mut self, addr: u32, val: Store) {
        self.bus.write(addr, val);
    }

    fn reg_read(&mut self, reg: u8) -> u32 {
        let reg = reg as usize;
        if reg >= self.reg.len() {
            panic!("invalid register read");
        }
        self.reg[reg]
    }

    fn reg_write(&mut self, reg: u8, val: u32) {
        let reg = reg as usize;
        if reg >= self.reg.len() {
            panic!("invalid register write");
        }
        self.reg[reg] = val;
        self.reg[0] = 0
    }

    pub fn gpu_queue(&self) -> &Vec<Primitive> {
        &self.bus.gpu.queue
    }

    pub fn gpu_clear(&mut self) {
        self.bus.gpu.update = false;
        self.bus.gpu.queue.clear();
    }

    pub fn update(&mut self) -> bool {
        self.bus.gpu.update
    }

    pub fn step(&mut self) -> bool {
        let inst = Instruction::new(self.read(self.pc));

        self.pc += 4;
        inst.execute(self);
        if self.pc == 0 {
            return false;
        }
        true
    }
}
