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
    if let Some(form) = vector_form(mnemonic) {
        return Some((form.sources.len(), Bank::Float));
    }
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

/// How a NEON instruction arranges its operands.
///
/// Vector instructions name a lane arrangement (`v3.4s`), and the widening
/// ones use a *different* arrangement for the result than for the sources:
/// `sxtl v5.8h, v5.8b` reads bytes and writes halfwords. Emitting one
/// arrangement for both is an assembler error; guessing which is worse. Real
/// code is full of these -- an int8 attention loop mined from a live profile
/// produced sxtl, smlal, scvtf and fmla and none of them could be costed.
#[derive(Clone, Copy, Debug)]
pub struct VectorForm {
    pub dest: &'static str,
    pub sources: &'static [&'static str],
    /// True when the destination is also read: `fmla` and `smlal` accumulate
    /// into it, which is where the dependence in a dot-product loop lives.
    pub accumulates: bool,
}

pub fn vector_form(mnemonic: &str) -> Option<VectorForm> {
    let form = match mnemonic {
        // Widening sign/zero extend: bytes in, halfwords out.
        "sxtl" | "uxtl" => VectorForm { dest: "8h", sources: &["8b"], accumulates: false },
        "sxtl2" | "uxtl2" => VectorForm { dest: "8h", sources: &["16b"], accumulates: false },
        // Widening multiply-accumulate: halfwords in, words out, accumulating.
        "smlal" | "umlal" => VectorForm {
            dest: "4s",
            sources: &["4h", "4h"],
            accumulates: true,
        },
        "smull" | "umull" => VectorForm {
            dest: "4s",
            sources: &["4h", "4h"],
            accumulates: false,
        },
        // The instruction this whole exercise keeps rediscovering: a dot
        // product over bytes accumulating into words.
        "sdot" | "udot" => VectorForm {
            dest: "4s",
            sources: &["16b", "16b"],
            accumulates: true,
        },
        "scvtf" | "ucvtf" => VectorForm { dest: "4s", sources: &["4s"], accumulates: false },
        "fcvtzs" | "fcvtzu" => VectorForm { dest: "4s", sources: &["4s"], accumulates: false },
        "fmla" | "fmls" => VectorForm {
            dest: "4s",
            sources: &["4s", "4s"],
            accumulates: true,
        },
        "addv" | "saddlv" => VectorForm { dest: "4s", sources: &["4s"], accumulates: false },
        _ => return None,
    };
    Some(form)
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
        let mut produced: BTreeMap<usize, usize> = BTreeMap::new();
        let mut incoming: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (producer, consumer) in &self.edges {
            incoming.entry(*consumer).or_default().push(*producer);
        }

        let mut lines = Vec::new();
        let mut carry = carry.and_then(|c| register_index(&c));
        for (index, mnemonic) in self.mnemonics.iter().enumerate() {
            let (sources, bank) = operand_arity(mnemonic).expect("checked at parse");
            let form = vector_form(mnemonic);

            // An accumulating instruction reads its destination as well as
            // writing it, so in a dependent chain the accumulator *is* the
            // carried value: fmla and smlal are where the dependence in a
            // dot-product loop actually lives.
            let accumulates = form.map(|f| f.accumulates).unwrap_or(false);
            let destination = if writes_result(mnemonic) {
                let register = if accumulates {
                    carry.take().unwrap_or_else(|| results.next_index(bank))
                } else {
                    results.next_index(bank)
                };
                produced.insert(index, register);
                Some(register)
            } else {
                None
            };

            let feeders = incoming.get(&index).cloned().unwrap_or_default();
            let mut operands: Vec<String> = Vec::new();
            for slot in 0..sources {
                let arrangement = form.map(|f| f.sources[slot.min(f.sources.len() - 1)]);
                let register = if let Some(producer) = feeders.get(slot) {
                    *produced.get(producer).unwrap_or(&results.next_index(bank))
                } else if slot == 0 && carry.is_some() && !accumulates {
                    carry.take().expect("checked")
                } else {
                    results.borrow_index(bank)
                };
                operands.push(format_register(bank, register, arrangement));
            }

            let mut text = String::from(mnemonic.as_str());
            text.push(' ');
            if let Some(register) = destination {
                text.push_str(&format_register(bank, register, form.map(|f| f.dest)));
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

/// The numeric part of a register name, ignoring any lane arrangement.
///
/// A vector register is written differently by each instruction that touches
/// it -- `v5.8h` here, `v5.4h` there -- so a chain has to be carried by the
/// number and re-dressed at each use, not by the spelling.
pub fn register_index(name: &str) -> Option<usize> {
    let digits: String = name
        .trim_start_matches(|c: char| c.is_alphabetic())
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse().ok()
}

pub fn format_register(bank: Bank, index: usize, arrangement: Option<&str>) -> String {
    match (bank, arrangement) {
        (_, Some(arr)) => format!("v{index}.{arr}"),
        (Bank::Int, None) => format!("x{index}"),
        (Bank::Float, None) => format!("s{index}"),
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

    pub fn next_index(&mut self, bank: Bank) -> usize {
        match bank {
            Bank::Int => {
                let index = self.next_int % 16;
                self.next_int += 1;
                index
            }
            Bank::Float => {
                let index = self.next_float % 16;
                self.next_float += 1;
                index
            }
        }
    }

    pub fn borrow_index(&mut self, bank: Bank) -> usize {
        match bank {
            Bank::Int => {
                let index = 20 + (self.input_int - 20 + 1) % 8;
                self.input_int = index;
                index
            }
            Bank::Float => {
                let index = 20 + (self.input_float - 20 + 1) % 8;
                self.input_float = index;
                index
            }
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
