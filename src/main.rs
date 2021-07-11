mod chip;
mod error;
mod pc;
mod renderer;

use chip::Chip;
use error::ChipResult;
use renderer::Renderer;
use std::env;

fn main() -> ChipResult<()> {
    let path = &env::args().collect::<Vec<String>>()[1];
    let mut chip = Chip::initialize(path)?;
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
