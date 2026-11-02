// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Path validation and same-file checks for Safetensors shards (incl. Win32 FFI).

use super::model_load;
use crate::error::Result;
use std::fs;
#[cfg(unix)]
use std::os::unix::fs::MetadataExt;
use std::path::{Component, Path, PathBuf};

pub(super) fn paths_refer_to_same_file(left: &Path, right: &Path) -> std::io::Result<bool> {
    if !left.exists() || !right.exists() {
        return Ok(false);
    }

    #[cfg(unix)]
    {
        let left_meta = fs::metadata(left)?;
        let right_meta = fs::metadata(right)?;
        Ok(left_meta.dev() == right_meta.dev() && left_meta.ino() == right_meta.ino())
    }

    #[cfg(windows)]
    {
        windows_same_file_via_ffi(left, right)
    }

    #[cfg(not(any(unix, windows)))]
    {
        Ok(fs::canonicalize(left)? == fs::canonicalize(right)?)
    }
}

/// Windows file-identity comparison using direct Win32 FFI.
/// Avoids the unstable `windows_by_handle` feature (E0658 on stable Rust).
#[cfg(windows)]
pub(super) fn windows_same_file_via_ffi(left: &Path, right: &Path) -> std::io::Result<bool> {
    use std::ffi::c_void;
    use std::os::windows::ffi::OsStrExt;

    #[repr(C)]
    #[allow(clippy::upper_case_acronyms)]
    struct FILETIME {
        dw_low_date_time: u32,
        dw_high_date_time: u32,
    }

    #[repr(C)]
    struct BY_HANDLE_FILE_INFORMATION {
        dw_file_attributes: u32,
        ft_creation_time: FILETIME,
        ft_last_access_time: FILETIME,
        ft_last_write_time: FILETIME,
        dw_volume_serial_number: u32,
        n_file_size_high: u32,
        n_file_size_low: u32,
        n_number_of_links: u32,
        n_file_index_high: u32,
        n_file_index_low: u32,
    }

    unsafe extern "system" {
        fn CreateFileW(
            lp_file_name: *const u16,
            dw_desired_access: u32,
            dw_share_mode: u32,
            lp_security_attributes: *mut c_void,
            dw_creation_disposition: u32,
            dw_flags_and_attributes: u32,
            h_template_file: *mut c_void,
        ) -> *mut c_void;
        fn GetFileInformationByHandle(
            h_file: *mut c_void,
            lp_file_information: *mut BY_HANDLE_FILE_INFORMATION,
        ) -> i32;
        fn CloseHandle(h_object: *mut c_void) -> i32;
    }

    const GENERIC_READ: u32 = 0x80000000;
    const FILE_SHARE_READ: u32 = 0x00000001;
    const FILE_SHARE_WRITE: u32 = 0x00000002;
    const OPEN_EXISTING: u32 = 3;
    const FILE_FLAG_BACKUP_SEMANTICS: u32 = 0x02000000;
    const INVALID_HANDLE_VALUE: *mut c_void = -1isize as *mut c_void;

    fn open_for_info(path: &Path) -> *mut c_void {
        let wide: Vec<u16> = path
            .as_os_str()
            .encode_wide()
            .chain(std::iter::once(0))
            .collect();
        // SAFETY: CreateFileW is called with valid parameters and a null-terminated wide string.
        unsafe {
            CreateFileW(
                wide.as_ptr(),
                GENERIC_READ,
                FILE_SHARE_READ | FILE_SHARE_WRITE,
                std::ptr::null_mut(),
                OPEN_EXISTING,
                FILE_FLAG_BACKUP_SEMANTICS,
                std::ptr::null_mut(),
            )
        }
    }

    let h_left = open_for_info(left);
    if h_left == INVALID_HANDLE_VALUE {
        return Ok(false);
    }
    let h_right = open_for_info(right);
    if h_right == INVALID_HANDLE_VALUE {
        // SAFETY: h_left was successfully opened above.
        unsafe {
            CloseHandle(h_left);
        }
        return Ok(false);
    }

    let mut info_left: BY_HANDLE_FILE_INFORMATION =
        // SAFETY: zeroed is valid for this repr(C) struct with only integer fields.
        unsafe { std::mem::zeroed() };
    let mut info_right: BY_HANDLE_FILE_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: both handles are valid, and info structs are stack-allocated.
    let ok_left = unsafe { GetFileInformationByHandle(h_left, &mut info_left) } != 0;
    let ok_right = unsafe { GetFileInformationByHandle(h_right, &mut info_right) } != 0;
    // SAFETY: closing valid handles.
    unsafe {
        CloseHandle(h_left);
        CloseHandle(h_right);
    }

    if !ok_left || !ok_right {
        return Ok(false);
    }

    Ok(
        info_left.dw_volume_serial_number == info_right.dw_volume_serial_number
            && info_left.n_file_index_high == info_right.n_file_index_high
            && info_left.n_file_index_low == info_right.n_file_index_low,
    )
}

pub(super) fn canonical_existing_or_parent(path: &Path) -> Result<PathBuf> {
    if path.exists() {
        return fs::canonicalize(path).map_err(|e| model_load(path, e.to_string()));
    }
    let parent = parent_or_current(path);
    let file_name = path.file_name().ok_or_else(|| {
        model_load(
            path,
            "manifest output path must include a file name".to_string(),
        )
    })?;
    let parent = fs::canonicalize(parent).map_err(|e| model_load(parent, e.to_string()))?;
    Ok(parent.join(file_name))
}

pub(super) fn validate_path_stays_under_root(root: &Path, path: &Path) -> Result<()> {
    let canonical_root = fs::canonicalize(root).map_err(|e| model_load(root, e.to_string()))?;
    let canonical_path = fs::canonicalize(path).map_err(|e| model_load(path, e.to_string()))?;
    if !canonical_path.starts_with(&canonical_root) {
        return Err(model_load(
            path,
            "Safetensors shard path must stay within the checkpoint directory".into(),
        ));
    }
    Ok(())
}

pub(super) fn parent_or_current(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

pub(super) fn is_safetensors_index(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.ends_with(".safetensors.index.json"))
}

pub(super) fn index_shard_path(root: &Path, index_path: &Path, relative: &str) -> Result<PathBuf> {
    let relative_path = Path::new(relative);
    let mut normalized = PathBuf::new();
    let escapes_root = relative_path.is_absolute()
        || relative_path.components().any(|component| match component {
            Component::Normal(part) => {
                normalized.push(part);
                false
            }
            Component::CurDir => false,
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => true,
        });
    if escapes_root {
        return Err(model_load(
            index_path,
            format!("index shard path '{relative}' must stay within the checkpoint directory"),
        ));
    }
    if normalized.as_os_str().is_empty() {
        return Err(model_load(
            index_path,
            format!("index shard path '{relative}' must name a Safetensors shard"),
        ));
    }
    let candidate = root.join(normalized);
    validate_path_stays_under_root(root, &candidate)?;
    Ok(candidate)
}
