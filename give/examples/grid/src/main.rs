#![no_std]
#![no_main]

fn draw_grid(grid: &[u8], data: &[u8], w: u32, h: u32, x: u32, y: u32) {
    for (i, c) in grid.iter().enumerate() {
        let cx = (x + (i as u32 % w)) * 16;
        let cy = (y + (i as u32 / h)) * 16;

        match c {
            0 => give::draw_rect_color(0x000000ff, cx, cy),
            1 => give::draw_rect(data.as_ptr(), cx, cy),
            _ => unreachable!(),
        }
    }
}

#[no_mangle]
pub extern "C" fn _start() -> ! {
    let tile = include_bytes!("../data/tile");

    #[rustfmt::skip]
    let formation = [
        0, 1, 0,
        1, 0, 1,
        0, 1, 0,

        1, 1, 1,
        1, 0, 1,
        1, 1, 1,

        1, 0, 1,
        0, 1, 0,
        1, 0, 1,
    ];

    let mut n = 0;
    let mut timer = 0;
    loop {
        draw_grid(&formation[n * 9..(n * 9) + 9], tile, 3, 3, 400, 300);
        timer += 1;
        if timer == 16 {
            n += 1;
            timer = 0;
        }
        if n == 3 {
            n = 0;
        }
        give::redraw();
    }
}
