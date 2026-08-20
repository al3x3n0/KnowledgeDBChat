//! Read a mined pattern and give every value in it a register.
//!
//! The miner names a candidate by its shape -- `"fmul fmadd | 0>1"` is a
//! multiply whose result feeds a multiply-accumulate -- because two
//! occurrences differing only in which registers they happened to use are the
//! same candidate. Costing one means turning that shape back into assembly an
//! assembler will accept, which means inventing registers that reproduce the
//! wiring exactly. Get that wrong and the sequence measured is not the
//! sequence found.

use std::collections::BTreeMap;
use std::fmt;

/// How many source operands a mnemonic reads, and in which register bank.
///
/// Only what is needed to emit a legal instruction. An unknown mnemonic is
/// reported rather than guessed: emitting `fwhatever s0, s1, s2` produces an
/// assembler error at the far end of the pipeline, where it is hard to trace
/// back to this table.
pub fn operand_arity(mnemonic: &str) -> Option<(usize, Bank)> {
    let bank = if mnemonic.starts_with('f') {
        Bank::Float
    } else {
        Bank::Int
    };
    let sources = match mnemonic {
        "fmov" | "fneg" | "fabs" | "fsqrt" | "frinta" | "frintn" | "fcvt" | "mov" | "neg"
        | "mvn" | "sxtw" | "uxtw" => 1,
        "fmul" | "fadd" | "fsub" | "fdiv" | "fmax" | "fmin" | "fcmp" | "add" | "sub"
        | "mul" | "and" | "orr" | "eor" | "lsl" | "lsr" | "asr" | "sdiv" | "udiv"
        | "cmp" => 2,
        "fmadd" | "fmsub" | "fnmadd" | "fnmsub" | "madd" | "msub" => 3,
        _ => return None,
    };
    Some((sources, bank))
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Bank {
    Int,
    Float,
}

impl Bank {
    fn name(self, index: usize) -> String {
        match self {
            Bank::Int => format!("x{index}"),
            Bank::Float => format!("s{index}"),
        }
    }
}

/// Instructions that write no register, so nothing downstream can read them.
pub fn writes_result(mnemonic: &str) -> bool {
    !matches!(mnemonic, "cmp" | "fcmp" | "cmn" | "tst")
}

#[derive(Debug, PartialEq, Eq)]
pub struct Pattern {
    pub mnemonics: Vec<String>,
    /// `(producer, consumer)` positions within the pattern.
    pub edges: Vec<(usize, usize)>,
}

#[derive(Debug)]
pub enum PatternError {
    Empty,
    UnknownMnemonic(String),
    EdgeOutOfRange(usize, usize),
    EdgeFromNonProducer(String),
}

impl fmt::Display for PatternError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PatternError::Empty => write!(f, "the pattern names no instructions"),
            PatternError::UnknownMnemonic(m) => write!(
                f,
                "unknown mnemonic '{m}': add it to operand_arity, since guessing its \
                 operand count emits assembly the assembler will reject"
            ),
            PatternError::EdgeOutOfRange(a, b) => {
                write!(f, "edge {a}>{b} names a position the pattern does not have")
            }
            PatternError::EdgeFromNonProducer(m) => write!(
                f,
                "edge starts at '{m}', which writes no register, so nothing can read it"
            ),
        }
    }
}

impl Pattern {
    /// Parse the miner's own pattern spelling: `"fmul fmadd | 0>1,1>2"`.
    pub fn parse(text: &str) -> Result<Pattern, PatternError> {
        let (left, right) = match text.split_once('|') {
            Some((l, r)) => (l, r),
            None => (text, ""),
        };
        let mnemonics: Vec<String> = left
            .split_whitespace()
            .map(|m| m.trim().to_lowercase())
            .filter(|m| !m.is_empty())
            .collect();
        if mnemonics.is_empty() {
            return Err(PatternError::Empty);
        }
        for mnemonic in &mnemonics {
            if operand_arity(mnemonic).is_none() {
                return Err(PatternError::UnknownMnemonic(mnemonic.clone()));
            }
        }

        let mut edges = Vec::new();
        for piece in right.split(',') {
            let piece = piece.trim();
            if piece.is_empty() {
                continue;
            }
            if let Some((a, b)) = piece.split_once('>') {
                let a: usize = a.trim().parse().unwrap_or(usize::MAX);
                let b: usize = b.trim().parse().unwrap_or(usize::MAX);
                if a >= mnemonics.len() || b >= mnemonics.len() {
                    return Err(PatternError::EdgeOutOfRange(a, b));
                }
                if !writes_result(&mnemonics[a]) {
                    return Err(PatternError::EdgeFromNonProducer(mnemonics[a].clone()));
                }
                edges.push((a, b));
            }
        }
        Ok(Pattern { mnemonics, edges })
    }

    /// External inputs the pattern needs: source slots no edge fills.
    pub fn external_inputs(&self) -> usize {
        let mut filled: BTreeMap<usize, usize> = BTreeMap::new();
        for (_, consumer) in &self.edges {
            *filled.entry(*consumer).or_insert(0) += 1;
        }
        self.mnemonics
            .iter()
            .enumerate()
            .map(|(i, m)| {
                let (sources, _) = operand_arity(m).expect("checked at parse");
                sources.saturating_sub(*filled.get(&i).unwrap_or(&0))
            })
            .sum()
    }

    /// Render the pattern as assembly, one instruction per line.
    ///
    /// `carry` is the register an edge-free first source should read, which is
    /// how a repetition is chained to the one before it: passing the previous
    /// copy's result makes the copies dependent, and passing a fresh register
    /// makes them independent. That choice changes the answer completely --
    /// it is the difference between measuring latency and throughput -- so the
    /// caller states it rather than this guessing.
    pub fn render(&self, results: &mut RegisterPool, carry: Option<String>) -> Vec<String> {
        let mut produced: BTreeMap<usize, String> = BTreeMap::new();
        let mut incoming: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (producer, consumer) in &self.edges {
            incoming.entry(*consumer).or_default().push(*producer);
        }

        let mut lines = Vec::new();
        let mut carry = carry;
        for (index, mnemonic) in self.mnemonics.iter().enumerate() {
            let (sources, bank) = operand_arity(mnemonic).expect("checked at parse");
            let mut operands: Vec<String> = Vec::new();

            let destination = if writes_result(mnemonic) {
                let register = results.next(bank);
                produced.insert(index, register.clone());
                Some(register)
            } else {
                None
            };

            let feeders = incoming.get(&index).cloned().unwrap_or_default();
            for slot in 0..sources {
                if let Some(producer) = feeders.get(slot) {
                    operands.push(
                        produced
                            .get(producer)
                            .cloned()
                            .unwrap_or_else(|| results.next(bank)),
                    );
                } else if slot == 0 && carry.is_some() {
                    // The chaining slot, consumed once so only the first
                    // unfilled source of the copy depends on the last one.
                    operands.push(carry.take().expect("checked"));
                } else {
                    operands.push(results.borrow_input(bank));
                }
            }

            let mut text = String::from(mnemonic.as_str());
            text.push(' ');
            if let Some(register) = &destination {
                text.push_str(register);
                if sources > 0 {
                    text.push_str(", ");
                }
            }
            text.push_str(&operands.join(", "));
            lines.push(text);
        }
        lines
    }

    /// The register holding the pattern's last result, for chaining copies.
    pub fn result_register(&self, lines: &[String]) -> Option<String> {
        for (index, mnemonic) in self.mnemonics.iter().enumerate().rev() {
            if writes_result(mnemonic) {
                return lines
                    .get(index)
                    .and_then(|l| l.split_whitespace().nth(1))
                    .map(|r| r.trim_end_matches(',').to_string());
            }
        }
        None
    }
}

/// Hands out registers, keeping results away from the read-only inputs.
pub struct RegisterPool {
    next_int: usize,
    next_float: usize,
    input_int: usize,
    input_float: usize,
}

impl RegisterPool {
    pub fn new() -> RegisterPool {
        // Results start low and inputs high so a long sequence does not wrap
        // its results onto the registers it is still reading.
        RegisterPool {
            next_int: 0,
            next_float: 0,
            input_int: 20,
            input_float: 20,
        }
    }

    pub fn next(&mut self, bank: Bank) -> String {
        match bank {
            Bank::Int => {
                let index = self.next_int % 16;
                self.next_int += 1;
                Bank::Int.name(index)
            }
            Bank::Float => {
                let index = self.next_float % 16;
                self.next_float += 1;
                Bank::Float.name(index)
            }
        }
    }

    /// A register the sequence only reads. Cycled over a small set so the
    /// snippet stays legal without inventing a dependence that was not in the
    /// mined pattern.
    pub fn borrow_input(&mut self, bank: Bank) -> String {
        match bank {
            Bank::Int => {
                let index = 20 + (self.input_int - 20 + 1) % 8;
                self.input_int = index;
                Bank::Int.name(index)
            }
            Bank::Float => {
                let index = 20 + (self.input_float - 20 + 1) % 8;
                self.input_float = index;
                Bank::Float.name(index)
            }
        }
    }
}

impl Default for RegisterPool {
    fn default() -> Self {
        RegisterPool::new()
    }
}
