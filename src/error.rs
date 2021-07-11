use sdl2::{render::UpdateTextureError, IntegerOrSdlError};
use std::{
    fmt::{self, Display, Formatter},
    io,
};

pub type ChipResult<T> = Result<T, ChipError>;

#[derive(Debug)]
pub enum ChipError {
    Io(io::Error),
    IntegerOr(IntegerOrSdlError),
    UpdateTexture(UpdateTextureError),
    UnknownOpcode(u16),
    Other(String),
}

impl Display for ChipError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        use ChipError::*;
        match self {
            Io(ref e) => e.fmt(f),
            IntegerOr(ref e) => e.fmt(f),
            UpdateTexture(ref e) => e.fmt(f),
            UnknownOpcode(ref o) => write!(f, "Unknown opcode: {:#02x}", o),
            Other(ref s) => f.write_str(s.as_str()),
        }
    }
}

impl From<io::Error> for ChipError {
    fn from(err: io::Error) -> ChipError {
        ChipError::Io(err)
    }
}

impl From<IntegerOrSdlError> for ChipError {
    fn from(err: IntegerOrSdlError) -> ChipError {
        ChipError::IntegerOr(err)
    }
}

impl From<UpdateTextureError> for ChipError {
    fn from(err: UpdateTextureError) -> ChipError {
        ChipError::UpdateTexture(err)
    }
}

impl From<String> for ChipError {
    fn from(err: String) -> ChipError {
        ChipError::Other(err)
    }
}
