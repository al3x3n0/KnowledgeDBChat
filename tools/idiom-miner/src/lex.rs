//! A tokenizer for C and C++ source, good enough to count idioms.
//!
//! Idiom counts drive which instruction-set extensions get proposed, so a
//! count that includes matches from comments and string literals is worse than
//! no count: it is wrong in a direction that looks like evidence. Tokenizing
//! drops both, and it makes a pattern independent of formatting, so
//! `a[i]*b[i]` and `a[ i ] * b[ i ]` are one idiom rather than two.
//!
//! This is not a C++ parser and does not pretend to be. It does not expand
//! macros or understand templates, so a match is "this token sequence occurs in
//! the source", never "this operation executes". Any report built on it has to
//! say so.

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Kind {
    Ident,
    Num,
    Punct,
}

#[derive(Debug, Clone)]
pub struct Token {
    pub kind: Kind,
    pub text: String,
    pub line: u32,
}

/// Multi-character operators, longest first so `<<=` wins over `<<` and `<`.
const OPERATORS: &[&str] = &[
    "<<=", ">>=", "...", "->*", "::", "->", "++", "--", "<<", ">>", "<=", ">=", "==", "!=", "&&",
    "||", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", ".*",
];

pub fn tokenize(source: &str) -> Vec<Token> {
    let bytes = source.as_bytes();
    let mut tokens = Vec::new();
    let mut i = 0usize;
    let mut line = 1u32;

    while i < bytes.len() {
        let c = bytes[i];

        if c == b'\n' {
            line += 1;
            i += 1;
            continue;
        }
        if c.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        // Comments carry prose that looks like code often enough to matter.
        if c == b'/' && i + 1 < bytes.len() {
            if bytes[i + 1] == b'/' {
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
                continue;
            }
            if bytes[i + 1] == b'*' {
                i += 2;
                while i + 1 < bytes.len() && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                    if bytes[i] == b'\n' {
                        line += 1;
                    }
                    i += 1;
                }
                i = (i + 2).min(bytes.len());
                continue;
            }
        }
        // String and character literals, escapes included.
        if c == b'"' || c == b'\'' {
            let quote = c;
            i += 1;
            while i < bytes.len() && bytes[i] != quote {
                if bytes[i] == b'\\' {
                    i += 1;
                } else if bytes[i] == b'\n' {
                    line += 1;
                }
                i += 1;
            }
            i = (i + 1).min(bytes.len());
            continue;
        }

        if c.is_ascii_alphabetic() || c == b'_' {
            let start = i;
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            tokens.push(Token {
                kind: Kind::Ident,
                text: source[start..i].to_string(),
                line,
            });
            continue;
        }

        if c.is_ascii_digit() {
            let start = i;
            // Suffixes and hex digits are part of the literal: 0x1p-3f, 1.5e6f,
            // 42ull. Splitting them would turn one literal into two tokens and
            // break any pattern that mentions a number.
            while i < bytes.len()
                && (bytes[i].is_ascii_alphanumeric()
                    || bytes[i] == b'.'
                    || ((bytes[i] == b'+' || bytes[i] == b'-')
                        && matches!(bytes[i - 1], b'e' | b'E' | b'p' | b'P')))
            {
                i += 1;
            }
            tokens.push(Token {
                kind: Kind::Num,
                text: source[start..i].to_string(),
                line,
            });
            continue;
        }

        let rest = &source[i..];
        let operator = OPERATORS.iter().find(|op| rest.starts_with(**op));
        let text = match operator {
            Some(op) => (*op).to_string(),
            None => (c as char).to_string(),
        };
        i += text.len();
        tokens.push(Token {
            kind: Kind::Punct,
            text,
            line,
        });
    }

    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texts(source: &str) -> Vec<String> {
        tokenize(source).into_iter().map(|t| t.text).collect()
    }

    #[test]
    fn comments_and_strings_do_not_produce_tokens() {
        let source = r#"
            // sum += a[i] * b[i];
            const char *s = "a[i] * b[i]";
            /* a[i] * b[i] */
            int x = 1;
        "#;
        let joined = texts(source).join(" ");
        assert!(!joined.contains('['), "found array syntax in {joined}");
    }

    #[test]
    fn spacing_does_not_change_the_token_sequence() {
        assert_eq!(texts("a[i]*b[i]"), texts("a [ i ] * b [ i ]"));
    }

    #[test]
    fn multi_character_operators_stay_whole() {
        assert_eq!(
            texts("a->b::c += 1"),
            ["a", "->", "b", "::", "c", "+=", "1"]
        );
    }

    #[test]
    fn numeric_literals_keep_their_suffixes_and_exponents() {
        assert_eq!(texts("1.5e-6f + 0x1fULL"), ["1.5e-6f", "+", "0x1fULL"]);
    }

    #[test]
    fn line_numbers_survive_comments_and_strings() {
        let tokens = tokenize("int a;\n/* two\nlines */\nint b;\n\"x\\ny\"\nint c;");
        let lines: Vec<u32> = tokens
            .iter()
            .filter(|t| t.text == "int")
            .map(|t| t.line)
            .collect();
        assert_eq!(lines, [1, 4, 6]);
    }

    #[test]
    fn an_unterminated_literal_does_not_hang_or_panic() {
        assert!(tokenize("const char *s = \"unterminated").len() >= 4);
        assert!(tokenize("/* unterminated").is_empty());
    }
}
