use crate::{
    chip::{self, Gfx},
    error::ChipResult,
};
use sdl2::{
    self,
    event::Event,
    keyboard::Keycode,
    pixels::PixelFormatEnum,
    render::{Canvas, Texture},
    video::Window,
    EventPump,
};

pub const PIXEL_FORMAT: PixelFormatEnum = PixelFormatEnum::ABGR8888;

const SCALE_FACTOR: u32 = 20;

pub struct Renderer {
    pub canvas: Canvas<Window>,
    event_pump: EventPump,
}

impl Renderer {
    pub fn initialize(title: &str) -> ChipResult<Self> {
        let sdl_context = sdl2::init()?;
        let event_pump = sdl_context.event_pump()?;
        let video_subsystem = sdl_context.video()?;
        let window = video_subsystem
            .window(
                title,
                chip::WIDTH as u32 * SCALE_FACTOR,
                chip::HEIGHT as u32 * SCALE_FACTOR,
            )
            .build()
            .unwrap();
        let canvas = window.into_canvas().build()?;
        Ok(Self { canvas, event_pump })
    }

    pub fn has_quit(&mut self) -> bool {
        for event in self.event_pump.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Q),
                    ..
                } => return true,
                _ => {}
            }
        }
        false
    }

    pub fn render(&mut self, gfx: &mut Gfx, texture: &mut Texture) -> ChipResult<()> {
        const PIXEL_SIZE: usize = 4;
        self.canvas.clear();
        let mut pixel_data: [u8; chip::WIDTH * chip::HEIGHT * PIXEL_SIZE] =
            [0; chip::WIDTH * chip::HEIGHT * PIXEL_SIZE];
        let mut index: usize = 0;
        for pixel in gfx.as_mut() {
            if *pixel != 0 {
                for i in 0..PIXEL_SIZE {
                    pixel_data[index + i] = 0xFF;
                }
            } else {
                for i in 0..PIXEL_SIZE {
                    pixel_data[index + i] = 0x00;
                }
            }
            index += 4;
        }
        texture.update(None, &pixel_data, chip::WIDTH * PIXEL_SIZE)?;
        self.canvas.copy(&texture, None, None)?;
        self.canvas.present();
        Ok(())
    }
}
