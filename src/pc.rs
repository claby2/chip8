#[derive(Debug)]
pub struct ProgramCounter {
    value: u16,
}

impl Default for ProgramCounter {
    fn default() -> Self {
        Self {
            value: Self::DEFAULT_VALUE,
        }
    }
}

impl ProgramCounter {
    pub const DEFAULT_VALUE: u16 = 0x200;
    pub const INCREMENT: u16 = 2;
    pub const SKIP: u16 = Self::INCREMENT * 2;

    pub fn value(&self) -> u16 {
        self.value
    }

    pub fn set(&mut self, value: u16) {
        self.value = value;
    }

    pub fn increment(&mut self) {
        self.value += Self::INCREMENT;
    }

    pub fn skip_if(&mut self, condition: bool) {
        self.value += if condition {
            Self::SKIP
        } else {
            Self::INCREMENT
        };
    }
}

#[cfg(test)]
mod tests {
    use super::ProgramCounter;

    #[test]
    fn pc_default_value() {
        let pc = ProgramCounter::default();
        assert_eq!(pc.value(), ProgramCounter::DEFAULT_VALUE);
    }

    #[test]
    fn pc_set() {
        let mut pc = ProgramCounter::default();
        pc.set(u16::MAX);
        assert_eq!(pc.value(), u16::MAX);
    }

    #[test]
    fn pc_increment() {
        let mut pc = ProgramCounter::default();
        pc.increment();
        assert_eq!(
            pc.value(),
            ProgramCounter::DEFAULT_VALUE + ProgramCounter::INCREMENT
        );
    }

    #[test]
    fn pc_skip_if_false() {
        let mut pc = ProgramCounter::default();
        pc.skip_if(false);
        assert_eq!(
            pc.value(),
            ProgramCounter::DEFAULT_VALUE + ProgramCounter::INCREMENT
        );
    }

    #[test]
    fn pc_skip_if_true() {
        let mut pc = ProgramCounter::default();
        pc.skip_if(true);
        assert_eq!(
            pc.value(),
            ProgramCounter::DEFAULT_VALUE + ProgramCounter::SKIP
        );
    }
}
