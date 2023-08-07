use crate::{Memory, Store};

enum Command {
    Control,
    Draw,
}

#[derive(PartialEq)]
enum DrawCommand {
    Rect,
    TexturedRect,
    Idle,
}

pub enum Primitive {
    Rect(u32, u32),
    TexturedRect(u32, u32),
}

struct Packet {
    cmd: DrawCommand,
    len: u32,
    count: u32,
    args: [u32; 2],
}

pub struct Gpu {
    pub update: bool,
    packet: Packet,
    pub queue: Vec<Primitive>,
}

impl Command {
    fn new(addr: u32) -> Self {
        match addr {
            0x0 => Self::Control,
            0x1 => Self::Draw,
            _ => panic!(),
        }
    }
}

impl DrawCommand {
    fn new(cmd: u32) -> Self {
        match cmd {
            0x0 => Self::Rect,
            0x1 => Self::TexturedRect,
            _ => panic!(),
        }
    }
}

impl Primitive {
    fn new(packet: &Packet) -> Self {
        let args = packet.args;

        match packet.cmd {
            DrawCommand::Rect => Self::Rect(args[0], args[1]),
            DrawCommand::TexturedRect => Self::TexturedRect(args[0], args[1]),
            _ => unreachable!(),
        }
    }
}

impl Packet {
    pub fn new(cmd: DrawCommand) -> Self {
        let len = match cmd {
            DrawCommand::Rect | DrawCommand::TexturedRect => 2,
            DrawCommand::Idle => 0,
        };

        Self {
            cmd,
            len,
            count: 0,
            args: [0; 2],
        }
    }

    pub fn idle(&self) -> bool {
        self.cmd == DrawCommand::Idle
    }

    fn reset(&mut self) {
        self.cmd = DrawCommand::Idle;
        self.len = 0;
    }

    fn transfer(&self) -> bool {
        self.len > 0 && self.count == self.len
    }

    fn read(&mut self, queue: &mut Vec<Primitive>, arg: u32) {
        self.args[self.count as usize] = arg;
        self.count += 1;

        if self.transfer() {
            queue.push(Primitive::new(self));
            self.reset();
        }
    }
}

impl Gpu {
    pub fn new() -> Self {
        Self {
            update: false,
            packet: Packet::new(DrawCommand::Idle),
            queue: vec![],
        }
    }
}

impl Memory for Gpu {
    fn read(&self, addr: u32) -> u32 {
        panic!("attempted to read at {:#08x?}", addr);
    }

    fn write(&mut self, addr: u32, val: Store) {
        if let Store::Word(w) = val {
            match Command::new(addr) {
                Command::Control => {
                    self.update = true;
                }
                Command::Draw => {
                    if self.packet.idle() {
                        self.packet = Packet::new(DrawCommand::new(w));
                    } else {
                        self.packet.read(&mut self.queue, w);
                    }
                }
            }
        } else {
            panic!("attempted to store {:#?}", val);
        }
    }
}
