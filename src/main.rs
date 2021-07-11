mod chip;
mod error;
mod pc;
mod renderer;

use chip::Chip;
use error::ChipResult;
use renderer::Renderer;
use std::{
    env,
    fs::{self, File},
    io::Read,
};

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
        if renderer.has_quit() {
            break;
        }
        chip.cycle()?;
        if chip.draw() {
            renderer.render(&mut chip.gfx(), &mut texture)?;
        }
    }
    Ok(())
}
