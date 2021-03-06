#[macro_use]
mod error;
mod chip;
mod pc;
mod renderer;
mod stack;

use chip::Chip;
use error::ChipResult;
use renderer::Renderer;
use std::{
    env,
    fs::{self, File},
    io::Read,
    thread,
    time::Duration,
};

const DELAY: u64 = 2;

fn main() -> ChipResult<()> {
    let path = &env::args().collect::<Vec<String>>()[1];
    let mut file = File::open(&path)?;
    let mut buffer = vec![0; fs::metadata(&path)?.len() as usize];
    file.read_exact(&mut buffer)?;
    let mut chip = Chip::new().load(&buffer)?;
    let mut renderer = Renderer::initialize(path)?;
    let texture_creator = renderer.canvas.texture_creator();
    let mut texture = texture_creator
        .create_texture_streaming(
            renderer::PIXEL_FORMAT,
            chip::WIDTH as u32,
            chip::HEIGHT as u32,
        )
        .unwrap();
    loop {
        if renderer.quit() {
            break;
        }
        renderer.process_event(chip.keypad());
        chip.cycle()?;
        if chip.draw() {
            renderer.render(&mut chip.gfx(), &mut texture)?;
        }
        thread::sleep(Duration::from_millis(DELAY));
    }
    Ok(())
}
