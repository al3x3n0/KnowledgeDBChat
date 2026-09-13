//! Count code idioms across a source tree, with provenance.
//!
//! This is the frequency half of the question "is this instruction-set
//! extension worth proposing": benefit is frequency times per-occurrence
//! saving, and the saving half is measured elsewhere, against a core model.
//!
//! What it reports is *static occurrence*: how often a token sequence appears
//! in the source. That is not how often it executes. A pattern occurring ten
//! thousand times in cold code is worth less than one occurring twice in an
//! inner loop, and nothing here can tell those apart. Every field is named for
//! what it actually counted so a report cannot quietly promote one to the
//! other.

mod lex;
mod pattern;

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use pattern::Pattern;

const DEFAULT_EXTENSIONS: &str = "c,cc,cpp,cxx,h,hh,hpp,hxx,inl";
const DEFAULT_MAX_FILE_BYTES: u64 = 4_000_000;
const DEFAULT_SAMPLES: usize = 3;

#[derive(Debug, Default)]
struct Skipped {
    too_large: usize,
    unreadable: usize,
    not_utf8: usize,
}

#[derive(Debug, Default)]
struct Counts {
    matches: usize,
    files: BTreeSet<usize>,
    samples: Vec<(usize, u32, String)>,
}

/// What one worker, or the whole scan, observed.
type ScanResult = (Vec<Counts>, Skipped, usize, u64, usize);

struct Options {
    root: PathBuf,
    commit: String,
    patterns: Vec<Pattern>,
    extensions: Vec<String>,
    samples: usize,
    threads: usize,
    max_file_bytes: u64,
}

fn json_escape(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 8);
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// Read `name: pattern` lines, ignoring blanks and `#` comments.
fn load_patterns(path: &Path) -> Result<Vec<Pattern>, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut patterns = Vec::new();
    for (number, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (name, body) = line.split_once(':').ok_or_else(|| {
            format!(
                "{}:{}: expected 'name: pattern'",
                path.display(),
                number + 1
            )
        })?;
        patterns.push(pattern::parse(name.trim(), body.trim())?);
    }
    if patterns.is_empty() {
        return Err(format!("{} defines no patterns", path.display()));
    }
    Ok(patterns)
}

fn collect_files(root: &Path, extensions: &[String], out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };
    let mut names: Vec<PathBuf> = entries.filter_map(|e| e.ok()).map(|e| e.path()).collect();
    // Sorted, so a bundle produced twice lists its evidence in the same order.
    names.sort();
    for path in names {
        let Ok(meta) = fs::symlink_metadata(&path) else {
            continue;
        };
        if meta.file_type().is_symlink() {
            continue; // following these can loop, and duplicates inflate counts
        }
        if meta.is_dir() {
            let skip = matches!(
                path.file_name().and_then(|n| n.to_str()),
                Some(".git") | Some(".svn") | Some("node_modules")
            );
            if !skip {
                collect_files(&path, extensions, out);
            }
            continue;
        }
        let matches_ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| extensions.iter().any(|want| want == e))
            .unwrap_or(false);
        if matches_ext {
            out.push(path);
        }
    }
}

fn scan(options: &Options) -> ScanResult {
    let mut files = Vec::new();
    collect_files(&options.root, &options.extensions, &mut files);
    let files = Arc::new(files);
    let patterns = Arc::new(options.patterns.clone());

    let results: Arc<Mutex<Vec<ScanResult>>> = Arc::new(Mutex::new(Vec::new()));
    let threads = options.threads.max(1).min(files.len().max(1));

    std::thread::scope(|scope| {
        for worker in 0..threads {
            let files = Arc::clone(&files);
            let patterns = Arc::clone(&patterns);
            let results = Arc::clone(&results);
            let max_file_bytes = options.max_file_bytes;
            scope.spawn(move || {
                let mut counts: Vec<Counts> =
                    (0..patterns.len()).map(|_| Counts::default()).collect();
                let mut skipped = Skipped::default();
                let (mut scanned, mut bytes, mut tokens_seen) = (0usize, 0u64, 0usize);

                for index in (worker..files.len()).step_by(threads) {
                    let path = &files[index];
                    let Ok(meta) = fs::metadata(path) else {
                        skipped.unreadable += 1;
                        continue;
                    };
                    if meta.len() > max_file_bytes {
                        skipped.too_large += 1;
                        continue;
                    }
                    let Ok(raw) = fs::read(path) else {
                        skipped.unreadable += 1;
                        continue;
                    };
                    let Ok(source) = String::from_utf8(raw) else {
                        skipped.not_utf8 += 1;
                        continue;
                    };
                    scanned += 1;
                    bytes += meta.len();
                    let tokens = lex::tokenize(&source);
                    tokens_seen += tokens.len();
                    for (slot, pat) in patterns.iter().enumerate() {
                        for hit in pattern::find(pat, &tokens) {
                            counts[slot].matches += 1;
                            counts[slot].files.insert(index);
                            counts[slot].samples.push((index, hit.line, hit.text));
                        }
                    }
                }
                results
                    .lock()
                    .unwrap()
                    .push((counts, skipped, scanned, bytes, tokens_seen));
            });
        }
    });

    let mut merged: Vec<Counts> = (0..options.patterns.len())
        .map(|_| Counts::default())
        .collect();
    let mut skipped = Skipped::default();
    let (mut scanned, mut bytes, mut tokens_seen) = (0usize, 0u64, 0usize);
    for (counts, part_skipped, part_scanned, part_bytes, part_tokens) in
        Arc::try_unwrap(results).unwrap().into_inner().unwrap()
    {
        for (slot, count) in counts.into_iter().enumerate() {
            merged[slot].matches += count.matches;
            merged[slot].files.extend(count.files);
            merged[slot].samples.extend(count.samples);
        }
        skipped.too_large += part_skipped.too_large;
        skipped.unreadable += part_skipped.unreadable;
        skipped.not_utf8 += part_skipped.not_utf8;
        scanned += part_scanned;
        bytes += part_bytes;
        tokens_seen += part_tokens;
    }
    for count in merged.iter_mut() {
        // Sort before truncating so the samples are the first occurrences in
        // the tree, not whichever thread happened to finish first.
        count.samples.sort_by_key(|sample| (sample.0, sample.1));
        count.samples.truncate(options.samples);
    }
    (merged, skipped, scanned, bytes, tokens_seen)
}

fn usage() -> String {
    "idiom-miner --root <dir> --patterns <file> [--commit <sha>] \
     [--ext c,cpp,h] [--samples 3] [--threads N] [--max-file-bytes N]"
        .to_string()
}

fn parse_args() -> Result<Options, String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut root: Option<PathBuf> = None;
    let mut patterns_path: Option<PathBuf> = None;
    let mut commit = String::new();
    let mut extensions = DEFAULT_EXTENSIONS.to_string();
    let mut samples = DEFAULT_SAMPLES;
    let mut threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let mut max_file_bytes = DEFAULT_MAX_FILE_BYTES;

    let mut index = 0;
    while index < args.len() {
        let take = |index: &mut usize| -> Result<String, String> {
            *index += 1;
            args.get(*index)
                .cloned()
                .ok_or_else(|| format!("{} needs a value\n{}", args[*index - 1], usage()))
        };
        match args[index].as_str() {
            "--root" => root = Some(PathBuf::from(take(&mut index)?)),
            "--patterns" => patterns_path = Some(PathBuf::from(take(&mut index)?)),
            "--commit" => commit = take(&mut index)?,
            "--ext" => extensions = take(&mut index)?,
            "--samples" => samples = take(&mut index)?.parse().map_err(|e| format!("{e}"))?,
            "--threads" => threads = take(&mut index)?.parse().map_err(|e| format!("{e}"))?,
            "--max-file-bytes" => {
                max_file_bytes = take(&mut index)?.parse().map_err(|e| format!("{e}"))?
            }
            "--help" | "-h" => return Err(usage()),
            other => return Err(format!("unknown argument: {other}\n{}", usage())),
        }
        index += 1;
    }

    let root = root.ok_or_else(|| format!("--root is required\n{}", usage()))?;
    if !root.is_dir() {
        return Err(format!("--root is not a directory: {}", root.display()));
    }
    let patterns_path =
        patterns_path.ok_or_else(|| format!("--patterns is required\n{}", usage()))?;

    Ok(Options {
        root,
        commit,
        patterns: load_patterns(&patterns_path)?,
        extensions: extensions
            .split(',')
            .map(|e| e.trim().trim_start_matches('.').to_string())
            .filter(|e| !e.is_empty())
            .collect(),
        samples,
        threads,
        max_file_bytes,
    })
}

fn main() {
    let options = match parse_args() {
        Ok(options) => options,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(2);
        }
    };

    let started = Instant::now();
    let (counts, skipped, scanned, bytes, tokens_seen) = scan(&options);
    let elapsed = started.elapsed();

    let mut files = Vec::new();
    collect_files(&options.root, &options.extensions, &mut files);
    let relative = |index: usize| -> String {
        files[index]
            .strip_prefix(&options.root)
            .unwrap_or(&files[index])
            .display()
            .to_string()
    };

    let mut out = String::new();
    out.push_str("{\n  \"corpus\": {\n");
    out.push_str(&format!(
        "    \"root\": \"{}\",\n    \"commit\": \"{}\",\n",
        json_escape(&options.root.display().to_string()),
        json_escape(&options.commit)
    ));
    out.push_str(&format!(
        "    \"files_matched_extension\": {},\n    \"files_scanned\": {},\n    \
         \"bytes_scanned\": {},\n    \"tokens_scanned\": {},\n",
        files.len(),
        scanned,
        bytes,
        tokens_seen
    ));
    out.push_str(&format!(
        "    \"files_skipped\": {{\"too_large\": {}, \"unreadable\": {}, \
         \"not_utf8\": {}}},\n",
        skipped.too_large, skipped.unreadable, skipped.not_utf8
    ));
    out.push_str(
        "    \"counts\": \"static occurrences in source; not execution frequency\"\n  },\n",
    );
    out.push_str("  \"patterns\": [\n");
    for (slot, pat) in options.patterns.iter().enumerate() {
        let count = &counts[slot];
        out.push_str(&format!(
            "    {{\"name\": \"{}\", \"pattern\": \"{}\", \"matches\": {}, \"files\": {}",
            json_escape(&pat.name),
            json_escape(&pat.source),
            count.matches,
            count.files.len()
        ));
        out.push_str(", \"samples\": [");
        for (position, (file_index, line, text)) in count.samples.iter().enumerate() {
            if position > 0 {
                out.push_str(", ");
            }
            out.push_str(&format!(
                "{{\"file\": \"{}\", \"line\": {}, \"text\": \"{}\"}}",
                json_escape(&relative(*file_index)),
                line,
                json_escape(text)
            ));
        }
        out.push(']');
        out.push('}');
        if slot + 1 < options.patterns.len() {
            out.push(',');
        }
        out.push('\n');
    }
    out.push_str("  ],\n");
    out.push_str(&format!("  \"elapsed_ms\": {}\n}}\n", elapsed.as_millis()));
    print!("{out}");
}
