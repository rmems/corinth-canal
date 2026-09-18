// SPDX-License-Identifier: Apache-2.0 OR MIT
//! CLI argument parsing for `spikenaut_ingest`.

use std::iter::Peekable;
use std::path::{Path, PathBuf};

use super::domain::SpikenautDomain;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cli {
    pub input: PathBuf,
    pub output: PathBuf,
    pub domain: Option<SpikenautDomain>,
    pub limit: Option<usize>,
    pub smoke: bool,
    pub output_root: PathBuf,
}

#[derive(Debug, Default)]
struct RawFlags {
    positional: Vec<PathBuf>,
    domain: Option<SpikenautDomain>,
    limit: Option<usize>,
    smoke: bool,
    output_root: Option<PathBuf>,
}

/// Collect process arguments as UTF-8 and validate path operands.
#[allow(dead_code)] // used by the example binary; unused in the #[path] test
pub fn from_os_args() -> Result<Cli, String> {
    parse_argv(collect_utf8_argv()?)
}

fn collect_utf8_argv() -> Result<Vec<String>, String> {
    std::env::args_os()
        .skip(1)
        .map(|token| {
            token
                .into_string()
                .map_err(|_| "argument is not valid UTF-8".to_owned())
        })
        .collect()
}

pub fn parse_argv<I, S>(argv: I) -> Result<Cli, String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut parsed = RawFlags::default();
    let mut items = argv.into_iter().peekable();
    while let Some(raw) = items.next() {
        apply_token(&mut parsed, raw.as_ref(), &mut items)?;
    }
    finish_cli(parsed)
}

fn apply_token<I, S>(
    parsed: &mut RawFlags,
    token: &str,
    items: &mut Peekable<I>,
) -> Result<(), String>
where
    I: Iterator<Item = S>,
    S: AsRef<str>,
{
    if token.starts_with('-') {
        apply_flag(parsed, token, items)
    } else {
        parsed.positional.push(user_path(token)?);
        Ok(())
    }
}

fn apply_flag<I, S>(
    parsed: &mut RawFlags,
    flag: &str,
    items: &mut Peekable<I>,
) -> Result<(), String>
where
    I: Iterator<Item = S>,
    S: AsRef<str>,
{
    if flag == "--help" || flag == "-h" {
        return Err("convert Spikenaut JSONL to canonical replay CSV".to_owned());
    }
    if flag == "--smoke" {
        parsed.smoke = true;
        return Ok(());
    }
    apply_valued_flag(parsed, flag, items)
}

fn apply_valued_flag<I, S>(
    parsed: &mut RawFlags,
    flag: &str,
    items: &mut Peekable<I>,
) -> Result<(), String>
where
    I: Iterator<Item = S>,
    S: AsRef<str>,
{
    match flag {
        "--domain" => set_domain(parsed, items),
        "--limit" => set_limit(parsed, items),
        "--output-root" => set_output_root(parsed, items),
        _ => Err(format!("unknown flag '{flag}'")),
    }
}

fn set_domain<I, S>(parsed: &mut RawFlags, items: &mut Peekable<I>) -> Result<(), String>
where
    I: Iterator<Item = S>,
    S: AsRef<str>,
{
    parsed.domain = parse_domain_flag(&next_value(items, "--domain")?)?;
    Ok(())
}

fn set_limit<I, S>(parsed: &mut RawFlags, items: &mut Peekable<I>) -> Result<(), String>
where
    I: Iterator<Item = S>,
    S: AsRef<str>,
{
    parsed.limit = Some(parse_limit(&next_value(items, "--limit")?)?);
    Ok(())
}

fn set_output_root<I, S>(parsed: &mut RawFlags, items: &mut Peekable<I>) -> Result<(), String>
where
    I: Iterator<Item = S>,
    S: AsRef<str>,
{
    parsed.output_root = Some(user_path(&next_value(items, "--output-root")?)?);
    Ok(())
}

fn next_value<I, S>(items: &mut Peekable<I>, flag: &str) -> Result<String, String>
where
    I: Iterator<Item = S>,
    S: AsRef<str>,
{
    items
        .next()
        .map(|value| value.as_ref().to_owned())
        .ok_or_else(|| format!("missing value for {flag}"))
}

fn parse_domain_flag(value: &str) -> Result<Option<SpikenautDomain>, String> {
    if value.eq_ignore_ascii_case("auto") {
        return Ok(None);
    }
    SpikenautDomain::from_alias(value)
        .map(Some)
        .ok_or_else(|| format!("unknown --domain '{value}'"))
}

fn parse_limit(value: &str) -> Result<usize, String> {
    value
        .parse()
        .map_err(|_| format!("invalid --limit '{value}'"))
}

fn finish_cli(parsed: RawFlags) -> Result<Cli, String> {
    if parsed.positional.len() > 2 {
        return Err(format!(
            "unexpected extra argument '{}'",
            parsed.positional[2].display()
        ));
    }
    let input = parsed
        .positional
        .first()
        .cloned()
        .ok_or_else(|| "missing <input.jsonl>".to_owned())?;
    let output_root = parsed
        .output_root
        .unwrap_or_else(|| PathBuf::from("artifacts"));
    let output = resolve_output(&parsed.positional, parsed.smoke, &output_root);
    Ok(Cli {
        input,
        output,
        domain: parsed.domain,
        limit: parsed.limit,
        smoke: parsed.smoke,
        output_root,
    })
}

fn resolve_output(positional: &[PathBuf], smoke: bool, output_root: &Path) -> PathBuf {
    if let Some(path) = positional.get(1) {
        return path.clone();
    }
    if smoke {
        default_smoke_csv(&positional[0], output_root)
    } else {
        default_output_csv(&positional[0])
    }
}

fn default_output_csv(input: &Path) -> PathBuf {
    input.with_extension("csv")
}

fn default_smoke_csv(input: &Path, output_root: &Path) -> PathBuf {
    let name = input
        .file_stem()
        .map(Path::new)
        .unwrap_or_else(|| Path::new("spikenaut"))
        .with_extension("csv");
    output_root.join(name)
}

/// Reject empty, NUL, and control-character paths before filesystem use.
fn user_path(raw: &str) -> Result<PathBuf, String> {
    if raw.is_empty() {
        return Err("empty path".to_owned());
    }
    if raw.contains('\0') || raw.chars().any(char::is_control) {
        return Err("path contains control characters".to_owned());
    }
    Ok(PathBuf::from(raw))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_argv_rejects_extra_positionals() {
        let err = parse_argv(["in.jsonl", "out.csv", "extra"]).unwrap_err();
        assert_eq!(err, "unexpected extra argument 'extra'");
    }

    #[test]
    fn parse_argv_smoke_default_csv_uses_output_root() {
        let cli = parse_argv(["in.jsonl", "--smoke", "--output-root", "artifacts"]).unwrap();
        assert!(cli.smoke);
        assert_eq!(cli.output, PathBuf::from("artifacts").join("in.csv"));
        assert_eq!(cli.output_root, PathBuf::from("artifacts"));
    }

    #[test]
    fn user_path_rejects_control_characters() {
        assert!(user_path("").is_err());
        assert!(user_path("bad\0path").is_err());
        assert!(user_path("ok.jsonl").is_ok());
    }
}
