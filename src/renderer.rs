use crate::{
    chip::{self, Gfx, Keypad},
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
    quit: bool,
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
        Ok(Self {
            canvas,
            event_pump,
            quit: false,
        })
    }

    pub fn quit(&self) -> bool {
        self.quit
    }

    pub fn process_event(&mut self, keypad: &mut Keypad) {
        for event in self.event_pump.poll_iter() {
            if let Event::Quit { .. } = event {
                self.quit = true;
                break;
            }
        }
        for key in self
            .event_pump
            .keyboard_state()
            .pressed_scancodes()
            .filter_map(Keycode::from_scancode)
            .collect::<Vec<Keycode>>()
            .iter()
        {
            match key {
                Keycode::Num1 => keypad[0x1] = 1,
                Keycode::Num2 => keypad[0x2] = 1,
                Keycode::Num3 => keypad[0x3] = 1,
                Keycode::Num4 => keypad[0xc] = 1,
                Keycode::Q => keypad[0x4] = 1,
                Keycode::W => keypad[0x5] = 1,
                Keycode::E => keypad[0x6] = 1,
                Keycode::R => keypad[0xd] = 1,
                Keycode::A => keypad[0x7] = 1,
                Keycode::S => keypad[0x8] = 1,
                Keycode::D => keypad[0x9] = 1,
                Keycode::F => keypad[0xe] = 1,
                Keycode::Z => keypad[0xa] = 1,
                Keycode::X => keypad[0x0] = 1,
                Keycode::C => keypad[0xb] = 1,
                Keycode::V => keypad[0xf] = 1,
                _ => {}
            }
        }
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
