// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Path identity and smoke artifact protection for the ingest CLI.

use std::path::{Component, Path, PathBuf};

use super::cli::Cli;
use super::domain::SpikenautDomain;
use super::smoke_artifacts::SmokePaths;

impl Cli {
    /// Prevent conversion from overwriting its source or smoke artifacts.
    pub fn validate_output_paths(&self, domain: SpikenautDomain) -> std::io::Result<()> {
        self.validate_input_output()?;
        if !self.smoke {
            return Ok(());
        }
        let run_dir = self
            .output_root
            .join(domain.source_slug())
            .join("dual_saaq_smoke");
        if smoke_output_overlap(&self.input, &run_dir)? {
            return Err(std::io::Error::other(format!(
                "smoke input '{}' overlaps smoke artifacts in '{}'",
                self.input.display(),
                run_dir.display()
            )));
        }
        if smoke_output_overlap(&self.output, &run_dir)? {
            return Err(std::io::Error::other(format!(
                "smoke CSV output '{}' overlaps smoke artifacts in '{}'",
                self.output.display(),
                run_dir.display()
            )));
        }
        Ok(())
    }

    fn validate_input_output(&self) -> std::io::Result<()> {
        reject_output_symlink(&self.output)?;
        if same_existing_file(&self.input, &self.output)? {
            return Err(std::io::Error::other(format!(
                "CSV output '{}' would overwrite input '{}'",
                self.output.display(),
                self.input.display()
            )));
        }
        Ok(())
    }
}

fn reject_output_symlink(path: &Path) -> std::io::Result<()> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => Err(std::io::Error::other(format!(
            "CSV output '{}' must not be a symlink",
            path.display()
        ))),
        Ok(_) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

fn smoke_output_overlap(output: &Path, run_dir: &Path) -> std::io::Result<bool> {
    if lexical_smoke_overlap(output, run_dir)? || canonical_smoke_overlap(output, run_dir)? {
        return Ok(true);
    }
    reserved_artifact_overlap(output, run_dir)
}

fn lexical_smoke_overlap(output: &Path, run_dir: &Path) -> std::io::Result<bool> {
    let output = normalized_absolute_path(output)?;
    let run_dir = normalized_absolute_path(run_dir)?;
    Ok(output.starts_with(&run_dir)
        || symlink_target(&output)?.is_some_and(|target| target.starts_with(&run_dir)))
}

fn canonical_smoke_overlap(output: &Path, run_dir: &Path) -> std::io::Result<bool> {
    Ok(resolved_path(output)?.starts_with(resolved_path(run_dir)?))
}

/// Resolve symlinks in the longest existing prefix while retaining a new leaf.
fn resolved_path(path: &Path) -> std::io::Result<PathBuf> {
    let mut current = absolute_path(path)?;
    let mut missing = Vec::new();
    loop {
        match std::fs::canonicalize(&current) {
            Ok(mut resolved) => {
                for component in missing.into_iter().rev() {
                    resolved.push(component);
                }
                return normalized_absolute_path(&resolved);
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                let name = current.file_name().ok_or(error)?;
                missing.push(name.to_os_string());
                current = current
                    .parent()
                    .ok_or_else(|| std::io::Error::other("output path has no existing ancestor"))?
                    .to_path_buf();
            }
            Err(error) => return Err(error),
        }
    }
}

fn reserved_artifact_overlap(output: &Path, run_dir: &Path) -> std::io::Result<bool> {
    let paths = SmokePaths::new(run_dir);
    [&paths.latent, &paths.tick, &paths.manifest, &paths.summary]
        .into_iter()
        .try_fold(false, |found, reserved| {
            Ok(found || same_existing_file(output, reserved)?)
        })
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
    let absolute = absolute_path(path)?;
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

fn absolute_path(path: &Path) -> std::io::Result<PathBuf> {
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        Ok(std::env::current_dir()?.join(path))
    }
}
