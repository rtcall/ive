#![no_std]

use core::panic::PanicInfo;
use core::ptr;

const CTRL_DRAW_ADDR: *mut u32 = 0x60000000 as *mut u32;
const DRAW_ADDR: *mut u32 = 0x60000001 as *mut u32;

pub const TILE_SIZE: usize = 16;

#[panic_handler]
pub fn panic(_panic: &PanicInfo<'_>) -> ! {
    loop {}
}

extern "C" fn ctrl_draw_write(val: u32) {
    unsafe {
        ptr::write_volatile(CTRL_DRAW_ADDR, val);
    }
}

extern "C" fn draw_write(val: u32) {
    unsafe {
        ptr::write_volatile(DRAW_ADDR, val);
    }
}

pub extern "C" fn redraw() {
    ctrl_draw_write(1);
}

pub extern "C" fn draw_rect_color(color: u32, x: u32, y: u32) {
    draw_write(0);
    draw_write((x << 16) | y);
    draw_write(color);
}

pub extern "C" fn draw_rect(data: *const u8, x: u32, y: u32) {
    draw_write(1);
    draw_write((x << 16) | y);
    draw_write(data as u32);
}
