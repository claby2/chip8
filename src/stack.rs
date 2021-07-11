use crate::error::{ChipError, ChipResult};
#[cfg(test)]
use std::ops::Index;

#[derive(Debug)]
pub struct Stack<T: Clone, const SIZE: usize> {
    values: Vec<T>,
}

impl<T: Clone, const SIZE: usize> Default for Stack<T, SIZE> {
    fn default() -> Self {
        Self { values: Vec::new() }
    }
}

impl<T: Clone, const SIZE: usize> Stack<T, SIZE> {
    #[cfg(test)]
    pub fn set(mut self, values: Vec<T>) -> Self {
        self.values = values;
        self
    }

    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn pop(&mut self) -> ChipResult<T> {
        self.values
            .pop()
            .ok_or_else(|| ChipError::StackOperation("Attempted to pop empty stack".to_owned()))
    }

    pub fn push(&mut self, value: T) -> ChipResult<()> {
        if self.values.len() >= SIZE {
            return Err(ChipError::StackOperation(
                "Attempted to push value onto full stack".to_owned(),
            ));
        }
        self.values.push(value);
        Ok(())
    }
}

#[cfg(test)]
impl<T: Clone, const SIZE: usize> Index<usize> for Stack<T, SIZE> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.values[index]
    }
}

#[cfg(test)]
mod tests {
    use super::{ChipError, Stack};

    #[test]
    fn stack_default() {
        let stack = Stack::<u16, 1>::default();
        assert!(stack.is_empty());
    }

    #[test]
    fn stack_pop_ok() {
        let mut stack = Stack::<u16, 1>::default().set(vec![1]);
        let value = stack.pop().unwrap();
        assert_eq!(value, 1);
    }

    #[test]
    fn stack_pop_err() {
        let mut stack = Stack::<u16, 1>::default();
        let result = stack.pop();
        assert_err!(result, Err(ChipError::StackOperation(_)));
    }

    #[test]
    fn stack_push_ok() {
        let mut stack = Stack::<u16, 1>::default();
        stack.push(1).unwrap();
        assert_eq!(stack[0], 1);
    }

    #[test]
    fn stack_push_err() {
        let mut stack = Stack::<u16, 0>::default();
        let result = stack.push(1);
        assert_err!(result, Err(ChipError::StackOperation(_)));
    }
}
