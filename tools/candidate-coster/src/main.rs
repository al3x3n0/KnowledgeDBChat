//! Turn a mined instruction pattern into the assembly llvm-mca must cost.
//!
//! `find_fusion_candidates` answers how often a shape runs. This is the other
//! half of "is it worth proposing": what one occurrence costs now, and what it
//! would cost fused. Benefit is frequency times saving, and neither number
//! means anything alone -- a pattern running ten million times that saves
//! nothing is not a candidate, and this project has already measured a change
//! that removed 300 instructions and saved zero cycles.
//!
//! What this emits is assembly, not a verdict. It writes the two snippets and
//! says what it assumed; llvm-mca costs them and the caller compares. The
//! separation is deliberate: the tool that decides what to measure should not
//! also be the tool that reports the answer.
//!
//!     candidate-coster emit --pattern "fmul fmadd | 0>1" --fused-as fmadd \
//!         --mode dependent --copies 20 --out-dir /tmp/cand
//!
//! Then run llvm-mca over `baseline.s` and `fused.s` with the same -mcpu.

mod emit;
mod pattern;

use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use emit::Mode;
use pattern::Pattern;

const USAGE: &str = "\
candidate-coster -- render a mined pattern as assembly for llvm-mca

USAGE:
  candidate-coster emit --pattern <SHAPE> [options]

OPTIONS:
  --pattern <SHAPE>     Required. The miner's spelling, e.g. \"fmul fmadd | 0>1\".
  --fused-as <SHAPE>    The instruction that stands in for the fused form.
                        llvm-mca cannot cost an instruction that does not
                        exist, so the fused cost is that of this stand-in and
                        the result says so. Default: fmadd.
  --mode <MODE>         dependent (latency) or independent (throughput).
                        Default dependent. The two answer different questions
                        and disagree by a lot; pick the one your loop matches.
  --copies <N>          Repetitions inside the region. Default 20.
  --out-dir <DIR>       Write baseline.s and fused.s here. Default: stdout.

The region is fenced with LLVM-MCA-BEGIN/END and ends with a newline, both of
which llvm-mca needs and neither of which it will tell you about clearly.
";

struct Args {
    pattern: Option<String>,
    fused_as: String,
    mode: Mode,
    copies: usize,
    out_dir: Option<PathBuf>,
}

fn parse_args(argv: &[String]) -> Result<Args, String> {
    let mut args = Args {
        pattern: None,
        fused_as: "fmadd".to_string(),
        mode: Mode::Dependent,
        copies: 20,
        out_dir: None,
    };
    let mut index = 0;
    while index < argv.len() {
        let flag = argv[index].as_str();
        let mut value = || -> Result<String, String> {
            index += 1;
            argv.get(index)
                .cloned()
                .ok_or_else(|| format!("{flag} needs a value"))
        };
        match flag {
            "--pattern" => args.pattern = Some(value()?),
            "--fused-as" => args.fused_as = value()?,
            "--mode" => {
                let text = value()?;
                args.mode = Mode::parse(&text).ok_or_else(|| {
                    format!("--mode must be dependent or independent, got '{text}'")
                })?;
            }
            "--copies" => {
                let text = value()?;
                args.copies = text
                    .parse()
                    .map_err(|_| format!("--copies must be a number, got '{text}'"))?;
                if args.copies == 0 || args.copies > 5000 {
                    return Err("--copies must be between 1 and 5000".to_string());
                }
            }
            "--out-dir" => args.out_dir = Some(PathBuf::from(value()?)),
            other => return Err(format!("unknown option '{other}'")),
        }
        index += 1;
    }
    Ok(args)
}

fn run(argv: &[String]) -> Result<String, String> {
    let args = parse_args(argv)?;
    let spec = args.pattern.ok_or("--pattern is required")?;
    let sequence = Pattern::parse(&spec).map_err(|e| e.to_string())?;
    let stand_in = Pattern::parse(&args.fused_as).map_err(|e| e.to_string())?;

    let baseline = emit::emit_sequence(&sequence, args.mode, args.copies, &spec);
    let fused = emit::emit_fused(
        &sequence,
        &stand_in,
        args.mode,
        args.copies,
        &format!("fused stand-in: {}", args.fused_as),
    );

    let mut report = String::new();
    report.push_str(&format!(
        "pattern           {spec}\n\
         external inputs   {}\n\
         mode              {}\n\
         copies            {}\n\
         baseline          {} instructions\n\
         fused stand-in    {} ({} instructions)\n",
        sequence.external_inputs(),
        args.mode.describe(),
        baseline.copies,
        baseline.instructions,
        args.fused_as,
        fused.instructions,
    ));
    report.push_str(
        "\nThe fused cost is the stand-in's cost, which is an assumption about\n\
         the instruction being proposed rather than a measurement of it. Cost\n\
         both files with the same -mcpu, and remember that fewer instructions\n\
         is not fewer cycles.\n",
    );

    match args.out_dir {
        Some(directory) => {
            fs::create_dir_all(&directory).map_err(|e| format!("{}: {e}", directory.display()))?;
            let baseline_path = directory.join("baseline.s");
            let fused_path = directory.join("fused.s");
            fs::write(&baseline_path, &baseline.text)
                .map_err(|e| format!("{}: {e}", baseline_path.display()))?;
            fs::write(&fused_path, &fused.text)
                .map_err(|e| format!("{}: {e}", fused_path.display()))?;
            report.push_str(&format!(
                "\nwrote {}\nwrote {}\n",
                baseline_path.display(),
                fused_path.display()
            ));
        }
        None => {
            report.push_str("\n=== baseline.s ===\n");
            report.push_str(&baseline.text);
            report.push_str("\n=== fused.s ===\n");
            report.push_str(&fused.text);
        }
    }
    Ok(report)
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    match argv.first().map(String::as_str) {
        None | Some("-h") | Some("--help") | Some("help") => {
            print!("{USAGE}");
            ExitCode::SUCCESS
        }
        Some("emit") => match run(&argv[1..]) {
            Ok(report) => {
                print!("{report}");
                ExitCode::SUCCESS
            }
            Err(problem) => {
                eprintln!("candidate-coster: {problem}");
                ExitCode::FAILURE
            }
        },
        Some(other) => {
            eprintln!("candidate-coster: unknown command '{other}'\n\n{USAGE}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn argv(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn a_pattern_is_required() {
        assert!(run(&argv(&[])).is_err());
    }

    #[test]
    fn an_unknown_mnemonic_is_named_rather_than_guessed() {
        let problem = run(&argv(&["--pattern", "fwhatever fadd"])).unwrap_err();

        assert!(problem.contains("fwhatever"), "{problem}");
        assert!(problem.contains("operand_arity"), "should say how to fix it");
    }

    #[test]
    fn an_edge_off_the_end_is_refused() {
        let problem = run(&argv(&["--pattern", "fmul fadd | 0>7"])).unwrap_err();

        assert!(problem.contains("does not have"), "{problem}");
    }

    #[test]
    fn an_edge_from_a_comparison_is_refused() {
        let problem = run(&argv(&["--pattern", "fcmp fadd | 0>1"])).unwrap_err();

        assert!(problem.contains("writes no register"), "{problem}");
    }

    #[test]
    fn the_mode_must_be_one_of_the_two_that_mean_something() {
        let problem = run(&argv(&["--pattern", "fmul fadd", "--mode", "fast"])).unwrap_err();

        assert!(problem.contains("dependent or independent"), "{problem}");
    }

    #[test]
    fn the_report_states_what_it_assumed() {
        let report = run(&argv(&["--pattern", "fmul fmadd | 0>1", "--copies", "8"])).unwrap();

        assert!(report.contains("assumption"), "the stand-in must be flagged");
        assert!(report.contains("dependent copies"));
        assert!(report.contains("=== baseline.s ==="));
        assert!(report.contains("=== fused.s ==="));
    }

    #[test]
    fn external_inputs_are_counted_from_unfilled_sources() {
        // fmul reads two, fmadd reads three, one of which the edge fills.
        let pattern = Pattern::parse("fmul fmadd | 0>1").unwrap();

        assert_eq!(pattern.external_inputs(), 4);
    }
}
