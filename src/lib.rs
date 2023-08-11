use anyhow::{bail, Result};
use egui::{ClippedPrimitive, Context, TexturesDelta};
use egui_wgpu::renderer::{Renderer, ScreenDescriptor};
use pixels::{wgpu, Pixels, PixelsContext, SurfaceTexture};
use std::fs;
use std::io::Read;
use winit::dpi::LogicalSize;
use winit::event::{Event, VirtualKeyCode};
use winit::event_loop::{ControlFlow, EventLoop, EventLoopWindowTarget};
use winit::window::{Window, WindowBuilder};
use winit_input_helper::WinitInputHelper;

mod cpu;
mod elf;
mod gpu;

use crate::cpu::Cpu;
use crate::gpu::Primitive;

const FB_WIDTH: u32 = 800;
const FB_HEIGHT: u32 = 600;
const WINDOW_WIDTH: u32 = 1024;
const WINDOW_HEIGHT: u32 = 768;

const GPU_SCROLLBACK: usize = 128;

const TILE_SIZE: u32 = 16;

#[derive(Debug)]
enum Store {
    Byte(u8),
    Half(u16),
    Word(u32),
}

enum Order {
    Lsb,
    Msb,
}

trait Memory {
    fn read(&self, addr: u32) -> u32;
    fn write(&mut self, addr: u32, val: Store);
}

struct RingBuffer<T> {
    buf: Vec<T>,
    len: usize,
    head: usize,
}

struct Gui {
    window_open: bool,
    gpu_log: RingBuffer<String>,
}

struct State {
    egui_ctx: Context,
    egui_state: egui_winit::State,
    screen_descriptor: ScreenDescriptor,
    renderer: Renderer,
    paint_jobs: Vec<ClippedPrimitive>,
    textures: TexturesDelta,
    gui: Gui,
    cpu: Cpu,
}

impl<T> RingBuffer<T> {
    fn new(len: usize) -> Self {
        Self {
            buf: Vec::new(),
            len,
            head: 0,
        }
    }

    fn push(&mut self, value: T) {
        if self.head == self.len {
            let len = self.len / 2;
            let _ = self.buf.drain(..len).collect::<Vec<_>>();
            self.head = len;
        }

        if self.head >= self.buf.len() {
            self.buf.push(value);
        } else {
            self.buf[self.head] = value;
        }

        self.head += 1;
    }

    fn vec(&mut self) -> &mut Vec<T> {
        &mut self.buf
    }
}

impl Gui {
    fn new() -> Self {
        Self {
            window_open: true,
            gpu_log: RingBuffer::new(GPU_SCROLLBACK),
        }
    }

    fn log(&mut self, buf: Vec<String>) {
        buf.iter().for_each(|l| self.gpu_log.push(l.to_string()));
    }

    fn ui(&mut self, ctx: &Context, cpu: &Cpu) {
        egui::Window::new("regs")
            .open(&mut self.window_open)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.monospace("pc:");
                    ui.monospace(format!("{:#08x}", cpu.pc));
                });
                for (i, reg) in cpu.reg.iter().enumerate() {
                    ui.horizontal(|ui| {
                        ui.monospace(format!("r{}:", i));
                        ui.monospace(format!("{:#08x}", reg));
                    });
                }
            });
        egui::Window::new("gpu")
            .open(&mut self.window_open)
            .constrain(true)
            .show(ctx, |ui| {
                let mut log = self.gpu_log.vec().join("\n");
                egui::ScrollArea::vertical()
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        ui.add(egui::TextEdit::multiline(&mut log).code_editor())
                    });
            });
    }
}

impl State {
    fn new<T>(
        event_loop: &EventLoopWindowTarget<T>,
        width: u32,
        height: u32,
        scale_factor: f32,
        pixels: &Pixels,
        cpu: Cpu,
    ) -> Self {
        let max_texture_size = pixels.device().limits().max_texture_dimension_2d as usize;

        let egui_ctx = Context::default();
        let mut egui_state = egui_winit::State::new(event_loop);
        egui_state.set_max_texture_side(max_texture_size);
        egui_state.set_pixels_per_point(scale_factor);
        let screen_descriptor = ScreenDescriptor {
            size_in_pixels: [width, height],
            pixels_per_point: scale_factor,
        };
        let renderer = Renderer::new(pixels.device(), pixels.render_texture_format(), None, 1);
        let textures = TexturesDelta::default();
        let gui = Gui::new();

        Self {
            egui_ctx,
            egui_state,
            screen_descriptor,
            renderer,
            paint_jobs: Vec::new(),
            textures,
            gui,
            cpu,
        }
    }

    pub fn handle_event(&mut self, event: &winit::event::WindowEvent) {
        let _ = self.egui_state.on_event(&self.egui_ctx, event);
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.screen_descriptor.size_in_pixels = [width, height];
        }
    }

    pub fn scale_factor(&mut self, scale_factor: f64) {
        self.screen_descriptor.pixels_per_point = scale_factor as f32;
    }

    pub fn prepare(&mut self, window: &Window) {
        let raw_input = self.egui_state.take_egui_input(window);
        let output = self.egui_ctx.run(raw_input, |egui_ctx| {
            self.gui.ui(egui_ctx, &self.cpu);
        });

        self.textures.append(output.textures_delta);
        self.egui_state
            .handle_platform_output(window, &self.egui_ctx, output.platform_output);
        self.paint_jobs = self.egui_ctx.tessellate(output.shapes);
    }

    pub fn render(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        render_target: &wgpu::TextureView,
        context: &PixelsContext,
    ) {
        for (id, image_delta) in &self.textures.set {
            self.renderer
                .update_texture(&context.device, &context.queue, *id, image_delta);
        }
        self.renderer.update_buffers(
            &context.device,
            &context.queue,
            encoder,
            &self.paint_jobs,
            &self.screen_descriptor,
        );

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: render_target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });

            self.renderer
                .render(&mut rpass, &self.paint_jobs, &self.screen_descriptor);
        }

        let textures = std::mem::take(&mut self.textures);
        for id in &textures.free {
            self.renderer.free_texture(id);
        }
    }

    fn draw_rect(frame: &mut [u8], order: Order, offset: u32, color: u32) {
        let offset = offset as usize * 4;

        let (r, g, b, a) = match order {
            Order::Lsb => (
                (color >> 24) as u8,
                (color >> 16) as u8,
                (color >> 8) as u8,
                color as u8,
            ),
            Order::Msb => (
                color as u8,
                (color >> 8) as u8,
                (color >> 16) as u8,
                (color >> 24) as u8,
            ),
        };

        if a == 0xff {
            frame[offset..offset + 4].copy_from_slice(&[r, g, b, a]);
        }
    }

    fn draw(&mut self, frame: &mut [u8]) {
        if !self.update() {
            return;
        }

        let log: Vec<String> = self
            .gpu_queue()
            .iter()
            .map(|cmd| match cmd {
                Primitive::Rect(offset, color) => {
                    let (x, y) = ((offset >> 16), (offset & 0xff));

                    for i in 0..TILE_SIZE * TILE_SIZE {
                        let x = x + (i % TILE_SIZE);
                        let y = (y + (i / TILE_SIZE)) * FB_WIDTH;
                        Self::draw_rect(frame, Order::Lsb, x + y, *color);
                    }

                    format!("Rect: {:#08x} {:#08x}", offset, color)
                }
                Primitive::TexturedRect(offset, addr) => {
                    let (x, y) = ((offset >> 16), (offset & 0xff));

                    for i in 0..TILE_SIZE * TILE_SIZE {
                        let data = self.read(addr + (i * 4));
                        let x = x + (i % TILE_SIZE);
                        let y = (y + (i / TILE_SIZE)) * FB_WIDTH;
                        Self::draw_rect(frame, Order::Msb, x + y, data);
                    }

                    format!("TexturedRect: {:#08x} {:#08x}", offset, addr)
                }
            })
            .collect();

        self.gui.log(log);
        self.gpu_clear();
    }

    fn read(&self, offset: u32) -> u32 {
        self.cpu.read(offset)
    }

    fn gpu_queue(&self) -> &Vec<Primitive> {
        self.cpu.gpu_queue()
    }

    fn gpu_clear(&mut self) {
        self.cpu.gpu_clear();
    }

    fn update(&mut self) -> bool {
        self.cpu.update()
    }

    fn step(&mut self) {
        self.cpu.step();
    }
}

pub fn run(path: String) -> Result<()> {
    env_logger::init();
    let mut file = fs::File::open(path)?;
    let mut data = Vec::new();

    file.read_to_end(&mut data)?;

    let cpu = match Cpu::new(data) {
        Ok(cpu) => cpu,
        Err(e) => match e {
            elf::Error::Eof => bail!("eof"),
            elf::Error::BadMagic => bail!("bad magic"),
            elf::Error::BadType => bail!("invalid executable"),
            elf::Error::BadMachine => bail!("foreign elf architecture"),
        },
    };

    let event_loop = EventLoop::new();
    let mut input = WinitInputHelper::new();
    let window = {
        let size = LogicalSize::new(WINDOW_WIDTH as f64, WINDOW_HEIGHT as f64);
        WindowBuilder::new()
            .with_title("ive")
            .with_inner_size(size)
            .with_min_inner_size(size)
            .with_resizable(false)
            .build(&event_loop)
            .unwrap()
    };

    let (mut pixels, mut state) = {
        let window_size = window.inner_size();
        let scale_factor = 1.0;
        let surface_texture = SurfaceTexture::new(window_size.width, window_size.height, &window);
        let pixels = Pixels::new(FB_WIDTH, FB_HEIGHT, surface_texture).unwrap();
        let state = State::new(
            &event_loop,
            window_size.width,
            window_size.height,
            scale_factor,
            &pixels,
            cpu,
        );
        (pixels, state)
    };

    event_loop.run(move |event, _, control_flow| {
        if input.update(&event) {
            if input.key_pressed(VirtualKeyCode::Escape) || input.close_requested() {
                *control_flow = ControlFlow::Exit;
                return;
            }

            if let Some(scale_factor) = input.scale_factor() {
                state.scale_factor(scale_factor);
            }

            if let Some(size) = input.window_resized() {
                if pixels.resize_surface(size.width, size.height).is_err() {
                    *control_flow = ControlFlow::Exit;
                    return;
                }
                state.resize(size.width, size.height);
            }

            window.request_redraw();
        }

        match event {
            Event::WindowEvent { event, .. } => {
                state.handle_event(&event);
            }
            Event::RedrawRequested(_) => {
                state.draw(pixels.frame_mut());
                state.prepare(&window);
                let _ = pixels.render_with(|encoder, target, ctx| {
                    ctx.scaling_renderer.render(encoder, target);
                    state.render(encoder, target, ctx);
                    Ok(())
                });
            }
            _ => (),
        }

        state.step();
        if state.update() {
            window.request_redraw();
        }
    });
}
