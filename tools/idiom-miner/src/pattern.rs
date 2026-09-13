//! Token-sequence patterns: what counts as an occurrence of an idiom.
//!
//! A pattern is written as tokens separated by spaces, so it reads like the
//! code it matches:
//!
//! ```text
//! $id [ $id ] * $id [ $id ]      matches  a[i] * b[i]
//! $id = $id * $id + $id          matches  r = a * x + y
//! sqrtf ( $any                   matches  sqrtf(x
//! ```
//!
//! `$id` is any identifier, `$num` any numeric literal, `$any` any single
//! token; anything else must match a token exactly.

use crate::lex::{Kind, Token};

#[derive(Debug, Clone, PartialEq)]
pub enum Element {
    AnyIdent,
    AnyNum,
    Any,
    Literal(String),
}

#[derive(Debug, Clone)]
pub struct Pattern {
    pub name: String,
    pub source: String,
    pub elements: Vec<Element>,
}

#[derive(Debug, Clone)]
pub struct Match {
    pub line: u32,
    pub text: String,
}

pub fn parse(name: &str, source: &str) -> Result<Pattern, String> {
    let elements: Vec<Element> = source
        .split_whitespace()
        .map(|piece| match piece {
            "$id" => Element::AnyIdent,
            "$num" => Element::AnyNum,
            "$any" => Element::Any,
            other => Element::Literal(other.to_string()),
        })
        .collect();
    if elements.is_empty() {
        return Err(format!("pattern '{name}' is empty"));
    }
    Ok(Pattern {
        name: name.to_string(),
        source: source.to_string(),
        elements,
    })
}

fn matches_at(pattern: &[Element], tokens: &[Token], start: usize) -> bool {
    if start + pattern.len() > tokens.len() {
        return false;
    }
    pattern.iter().enumerate().all(|(offset, element)| {
        let token = &tokens[start + offset];
        match element {
            Element::Any => true,
            Element::AnyIdent => token.kind == Kind::Ident,
            Element::AnyNum => token.kind == Kind::Num,
            Element::Literal(text) => token.text == *text,
        }
    })
}

/// Find every occurrence. Matches do not overlap: after one is found the scan
/// resumes past it, so `a[i]*b[i]*c[i]` counts as one multiply-pair and not two
/// that share a term.
pub fn find(pattern: &Pattern, tokens: &[Token]) -> Vec<Match> {
    let mut found = Vec::new();
    let mut index = 0usize;
    while index < tokens.len() {
        if matches_at(&pattern.elements, tokens, index) {
            let end = index + pattern.elements.len();
            let text = tokens[index..end]
                .iter()
                .map(|t| t.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            found.push(Match {
                line: tokens[index].line,
                text,
            });
            index = end;
        } else {
            index += 1;
        }
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lex::tokenize;

    fn run(pattern_source: &str, code: &str) -> Vec<Match> {
        let pattern = parse("t", pattern_source).unwrap();
        find(&pattern, &tokenize(code))
    }

    #[test]
    fn an_indexed_product_is_found_however_it_is_spaced() {
        let hits = run(
            "$id [ $id ] * $id [ $id ]",
            "s += a[i]*b[i]; t += c [ j ] * d [ j ];",
        );
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].text, "a [ i ] * b [ i ]");
    }

    #[test]
    fn a_literal_element_must_match_exactly() {
        assert_eq!(run("sqrtf ( $id )", "x = sqrtf(y);").len(), 1);
        assert_eq!(run("sqrtf ( $id )", "x = sqrt(y);").len(), 0);
    }

    #[test]
    fn a_number_element_does_not_match_an_identifier() {
        assert_eq!(run("$id * $num", "x = y * 2.0f;").len(), 1);
        assert_eq!(run("$id * $num", "x = y * z;").len(), 0);
    }

    #[test]
    fn matches_do_not_overlap() {
        // Three terms contain two adjacent pairs; counting both would report
        // more opportunities than a fused instruction could take.
        let hits = run("$id [ $id ] * $id [ $id ]", "a[i]*b[i]*c[i]");
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn a_match_reports_the_line_it_started_on() {
        let hits = run("$id [ $id ]", "int x;\nint y;\nz = a[i];");
        assert_eq!(hits[0].line, 3);
    }

    #[test]
    fn an_empty_pattern_is_rejected_rather_than_matching_everything() {
        assert!(parse("t", "   ").is_err());
    }
}
