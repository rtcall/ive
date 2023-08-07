use crate::cpu::RAM_BASE;
use byteorder::{LittleEndian, ReadBytesExt};
use std::io::Cursor;
use std::io::Read;

const MAGIC: [u8; 4] = [0x7f, 0x45, 0x4c, 0x46];
const LOAD: u32 = 1;

#[derive(Debug)]
pub enum Error {
    Eof,
    BadMagic,
    BadType,
    BadMachine,
}

struct Phdr {
    ptype: u32,
    offset: u32,
    vaddr: u32,
    memsz: u32,
}

fn read_u16<R: Read>(buf: &mut R) -> u16 {
    buf.read_u16::<LittleEndian>().unwrap()
}

fn read_u32<R: Read>(buf: &mut R) -> u32 {
    buf.read_u32::<LittleEndian>().unwrap()
}

impl Phdr {
    fn new<R: Read>(buf: &mut R) -> Self {
        let ptype = read_u32(buf);
        let offset = read_u32(buf);
        let vaddr = read_u32(buf);
        for _ in 0..2 {
            read_u32(buf);
        }
        let memsz = read_u32(buf);

        for _ in 0..2 {
            read_u32(buf);
        }

        Self {
            ptype,
            offset,
            vaddr,
            memsz,
        }
    }
}

pub fn load(elf: Vec<u8>, mem: &mut [u8]) -> Result<u32, Error> {
    let mut rdr = Cursor::new(elf);

    let mut ident: [u8; 16] = [0; 16];

    if rdr.read_exact(&mut ident).is_err() {
        return Err(Error::Eof);
    }

    if ident[..MAGIC.len()] != MAGIC {
        return Err(Error::BadMagic);
    }

    let etype = read_u16(&mut rdr);

    if etype != 0x2 {
        return Err(Error::BadType);
    }

    let machine = read_u16(&mut rdr);

    if machine != 0xf3 {
        return Err(Error::BadMachine);
    }

    // version
    read_u32(&mut rdr);

    let entry = read_u32(&mut rdr);
    let phoff = read_u32(&mut rdr);
    // shoff
    read_u32(&mut rdr);
    // flags
    read_u32(&mut rdr);
    // ehsize
    read_u16(&mut rdr);
    // phentsize
    read_u16(&mut rdr);
    let phnum = read_u16(&mut rdr);

    rdr.set_position(phoff as u64);

    for _ in 0..phnum {
        let phdr = Phdr::new(&mut rdr);

        if phdr.ptype != LOAD {
            continue;
        }

        let vaddr = (phdr.vaddr - RAM_BASE) as usize;
        let memsz = phdr.memsz as usize;

        let pos = rdr.position();
        rdr.set_position(phdr.offset as u64);
        if rdr.read_exact(&mut mem[vaddr..vaddr + memsz]).is_err() {
            return Err(Error::Eof);
        }
        rdr.set_position(pos);
    }

    Ok(entry)
}
