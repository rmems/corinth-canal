// SPDX-License-Identifier: Apache-2.0 OR MIT
//! CLI argument parsing for `spikenaut_ingest`.

use std::iter::Peekable;
use std::path::{Component, Path, PathBuf};

use super::domain::SpikenautDomain;
use super::smoke_artifacts::SmokePaths;

impl Cli {
    /// Prevent conversion from overwriting its source or smoke artifacts.
    pub fn validate_output_paths(&self, domain: SpikenautDomain) -> std::io::Result<()> {
        if same_existing_file(&self.input, &self.output)? {
            return Err(std::io::Error::other(format!(
                "CSV output '{}' would overwrite input '{}'",
                self.output.display(),
                self.input.display()
            )));
        }
        if !self.smoke {
            return Ok(());
        }
        let run_dir = self
            .output_root
            .join(domain.source_slug())
            .join("dual_saaq_smoke");
        let output = normalized_absolute_path(&self.output)?;
        let run_dir_absolute = normalized_absolute_path(&run_dir)?;
        let lexical_alias = output.starts_with(&run_dir_absolute)
            || symlink_target(&self.output)?
                .is_some_and(|target| target.starts_with(&run_dir_absolute));
        let resolved_alias = match (
            std::fs::canonicalize(&self.output),
            std::fs::canonicalize(&run_dir),
        ) {
            (Ok(output), Ok(run_dir)) => output.starts_with(run_dir),
            _ => false,
        };
        let paths = SmokePaths::new(&run_dir);
        let hardlink_alias = [&paths.latent, &paths.tick, &paths.manifest, &paths.summary]
            .into_iter()
            .try_fold(false, |found, reserved| {
                Ok::<bool, std::io::Error>(found || same_existing_file(&self.output, reserved)?)
            })?;
        if lexical_alias || resolved_alias || hardlink_alias {
            return Err(std::io::Error::other(format!(
                "smoke CSV output '{}' overlaps smoke artifacts in '{}'",
                self.output.display(),
                run_dir.display()
            )));
        }
        Ok(())
    }
}

fn same_existing_file(left: &Path, right: &Path) -> std::io::Result<bool> {
    if normalized_absolute_path(left)? == normalized_absolute_path(right)? {
        return Ok(true);
    }
    if let (Ok(left), Ok(right)) = (std::fs::canonicalize(left), std::fs::canonicalize(right))
        && left == right
    {
        return Ok(true);
    }
    #[cfg(unix)]
    if let (Ok(left), Ok(right)) = (std::fs::metadata(left), std::fs::metadata(right)) {
        use std::os::unix::fs::MetadataExt;
        return Ok(left.dev() == right.dev() && left.ino() == right.ino());
    }
    Ok(false)
}

fn symlink_target(path: &Path) -> std::io::Result<Option<PathBuf>> {
    let metadata = match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    };
    if !metadata.file_type().is_symlink() {
        return Ok(None);
    }
    let target = std::fs::read_link(path)?;
    let target = if target.is_absolute() {
        target
    } else {
        path.parent().unwrap_or_else(|| Path::new(".")).join(target)
    };
    normalized_absolute_path(&target).map(Some)
}

fn normalized_absolute_path(path: &Path) -> std::io::Result<PathBuf> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    let mut normalized = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::ParentDir => {
                normalized.pop();
            }
            Component::CurDir => {}
            Component::Prefix(_) | Component::RootDir | Component::Normal(_) => {
                normalized.push(component.as_os_str());
            }
        }
    }
    Ok(normalized)
}

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

/// Split a kernel cmdline blob (`NUL`-terminated tokens, argv[0] first).
pub fn tokens_from_cmdline(bytes: &[u8]) -> Result<Vec<String>, String> {
    let mut tokens = Vec::new();
    let mut parts = bytes.split(|byte| *byte == 0).collect::<Vec<_>>();
    if bytes.last() == Some(&0) {
        parts.pop();
    }
    for raw in parts {
        tokens.push(utf8_token(raw)?);
    }
    if tokens.is_empty() {
        return Ok(tokens);
    }
    Ok(tokens.split_off(1))
}

fn utf8_token(raw: &[u8]) -> Result<String, String> {
    std::str::from_utf8(raw)
        .map(str::to_owned)
        .map_err(|_| "argument is not valid UTF-8".to_owned())
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
    if items
        .peek()
        .is_none_or(|value| value.as_ref().starts_with('-'))
    {
        return Err(format!("missing value for {flag}"));
    }
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
    let mut path = output_root.join(if input.is_absolute() {
        "absolute"
    } else {
        "relative"
    });
    for component in input.components() {
        match component {
            Component::Normal(name) => {
                let mut encoded = std::ffi::OsString::from("n_");
                encoded.push(name);
                path.push(encoded);
            }
            Component::ParentDir => path.push("p_"),
            Component::CurDir | Component::RootDir | Component::Prefix(_) => {}
        }
    }
    path.as_mut_os_string().push(".csv");
    path
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
        assert_eq!(
            cli.output,
            PathBuf::from("artifacts/relative/n_in.jsonl.csv")
        );
        assert_eq!(cli.output_root, PathBuf::from("artifacts"));
    }

    #[test]
    fn smoke_default_csv_keeps_same_stem_inputs_distinct() {
        let gpu = parse_argv(["gpu/data.jsonl", "--smoke", "--output-root", "artifacts"]).unwrap();
        let hft = parse_argv(["hft/data.jsonl", "--smoke", "--output-root", "artifacts"]).unwrap();
        assert_ne!(gpu.output, hft.output);
        assert!(gpu.output.starts_with("artifacts"));
        assert!(hft.output.starts_with("artifacts"));
    }

    #[test]
    fn smoke_default_csv_distinguishes_parent_from_literal_component() {
        let parent = parse_argv(["../run.jsonl", "--smoke"]).unwrap();
        let literal = parse_argv(["__parent__/run.jsonl", "--smoke"]).unwrap();
        assert_ne!(parent.output, literal.output);
        let other_extension = parse_argv(["../run.txt", "--smoke"]).unwrap();
        assert_ne!(parent.output, other_extension.output);
    }

    #[test]
    fn smoke_rejects_csv_outputs_inside_its_artifact_directory() {
        for output in [
            "artifacts/spikenaut_gpu/dual_saaq_smoke/latent_telemetry.csv",
            "artifacts/spikenaut_gpu/dual_saaq_smoke/../dual_saaq_smoke/summary.json",
        ] {
            let cli = parse_argv(["in.jsonl", output, "--smoke"]).unwrap();
            assert!(cli.validate_output_paths(SpikenautDomain::Gpu).is_err());
        }
        let cli = parse_argv(["in.jsonl", "artifacts/safe.csv", "--smoke"]).unwrap();
        assert!(cli.validate_output_paths(SpikenautDomain::Gpu).is_ok());
    }

    #[test]
    fn output_cannot_alias_input_file() {
        let root = std::path::PathBuf::from("target/tmp-tests").join(format!(
            "spikenaut_input_alias_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let input = root.join("input.csv");
        std::fs::write(&input, b"source JSONL").unwrap();
        let input_arg = input.to_str().unwrap();
        let cli = parse_argv([input_arg]).unwrap();
        assert!(cli.validate_output_paths(SpikenautDomain::Gpu).is_err());
        let cli = parse_argv([input_arg, input_arg]).unwrap();
        assert!(cli.validate_output_paths(SpikenautDomain::Gpu).is_err());
        let hardlink = root.join("hardlink.csv");
        std::fs::hard_link(&input, &hardlink).unwrap();
        let cli = parse_argv([input_arg, hardlink.to_str().unwrap()]).unwrap();
        assert!(cli.validate_output_paths(SpikenautDomain::Gpu).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn smoke_output_symlink_cannot_alias_existing_artifact() {
        let root = std::path::PathBuf::from("target/tmp-tests").join(format!(
            "spikenaut_artifact_alias_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let run_dir = root.join("spikenaut_gpu/dual_saaq_smoke");
        std::fs::create_dir_all(&run_dir).unwrap();
        let latent = run_dir.join("latent_telemetry.csv");
        std::fs::write(&latent, b"old latent").unwrap();
        let alias = root.join("safe.csv");
        std::os::unix::fs::symlink(std::fs::canonicalize(&latent).unwrap(), &alias).unwrap();
        let cli = parse_argv([
            "in.jsonl",
            alias.to_str().unwrap(),
            "--smoke",
            "--output-root",
            root.to_str().unwrap(),
        ])
        .unwrap();
        assert!(cli.validate_output_paths(SpikenautDomain::Gpu).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn valued_flags_reject_the_next_flag_as_a_missing_value() {
        for flag in ["--output-root", "--domain", "--limit"] {
            let err = parse_argv(["in.jsonl", flag, "--smoke"]).unwrap_err();
            assert_eq!(err, format!("missing value for {flag}"));
        }
    }

    #[test]
    fn user_path_rejects_control_characters() {
        assert!(user_path("").is_err());
        assert!(user_path("bad\0path").is_err());
        assert!(user_path("ok.jsonl").is_ok());
    }

    #[test]
    fn tokens_from_cmdline_skips_argv0_and_rejects_non_utf8() {
        assert_eq!(
            tokens_from_cmdline(b"spikenaut_ingest\0in.jsonl\0--smoke\0").unwrap(),
            vec!["in.jsonl", "--smoke"]
        );
        assert_eq!(tokens_from_cmdline(b"").unwrap(), Vec::<String>::new());
        assert_eq!(
            tokens_from_cmdline(&[b'b', b'i', b'n', 0, 0xff]).unwrap_err(),
            "argument is not valid UTF-8"
        );
    }

    #[test]
    fn tokens_from_cmdline_preserves_empty_flag_value() {
        let tokens =
            tokens_from_cmdline(b"spikenaut_ingest\0in.jsonl\0--output-root\0\0--smoke\0").unwrap();
        assert_eq!(tokens, vec!["in.jsonl", "--output-root", "", "--smoke"]);
        assert_eq!(parse_argv(tokens).unwrap_err(), "empty path");
    }
}
