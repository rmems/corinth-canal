// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Timestamp parsing and ordinal fallback for Spikenaut records.

use serde_json::{Map, Value};

use super::fields::integer_field;

pub(super) fn timestamp_or_ordinal(fields: &Map<String, Value>, ordinal: u64) -> u64 {
    if let Some(ms) = parse_timestamp_field(fields.get("timestamp")) {
        return ms;
    }
    if let Some(ms) = parse_timestamp_field(fields.get("ts_utc")) {
        return ms;
    }
    if let Some(ms) = integer_field(fields, "row_index") {
        return ms;
    }
    if let Some(ms) = integer_field(fields, "step_idx") {
        return ms;
    }
    ordinal
}

fn parse_timestamp_field(value: Option<&Value>) -> Option<u64> {
    match value {
        Some(Value::Number(n)) => numeric_timestamp_ms(n.as_f64()?),
        Some(Value::String(s)) => parse_timestamp_string(s),
        _ => None,
    }
}

fn numeric_timestamp_ms(value: f64) -> Option<u64> {
    if !value.is_finite() || value < 0.0 {
        return None;
    }
    into_u64_millis(scale_epoch_or_ordinal(value))
}

fn scale_epoch_or_ordinal(value: f64) -> f64 {
    // ns since epoch (~1e18 in 2026), ms (~1e12), seconds (~1e9).
    if value >= 1.0e16 {
        value / 1.0e6
    } else if value >= 1.0e12 {
        value
    } else if value >= 1.0e9 {
        value * 1_000.0
    } else {
        // Small integers (row_index, step_idx) are ordinals, not unix time.
        value
    }
}

fn into_u64_millis(ms: f64) -> Option<u64> {
    (ms.is_finite() && ms >= 0.0 && ms <= u64::MAX as f64).then_some(ms as u64)
}

/// Parse ISO-8601 / space-separated datetimes. Returns `None` rather than
/// inventing a clock when the string is not a datetime.
pub fn parse_timestamp_string(raw: &str) -> Option<u64> {
    let s = normalize_timestamp_input(raw)?;
    let civil = parse_civil_prefix(s)?;
    let (frac, tz) = split_frac_and_tz(s.get(19..).unwrap_or(""));
    civil_to_unix_ms(civil, frac_millis(frac)?, tz)
}

fn normalize_timestamp_input(raw: &str) -> Option<&str> {
    let s = raw.trim();
    if s.is_empty() || s.eq_ignore_ascii_case("null") {
        return None;
    }
    // Refuse chain-tagged mining stamps such as `dynex:919876`.
    if s.contains(':') && !s.as_bytes().first().is_some_and(|b| b.is_ascii_digit()) {
        return None;
    }
    (s.len() >= 19).then_some(s)
}

struct CivilTime {
    year: i32,
    month: u32,
    day: u32,
    hour: u32,
    minute: u32,
    second: u32,
}

fn parse_civil_prefix(s: &str) -> Option<CivilTime> {
    if !is_datetime_separator(s.as_bytes().get(10).copied()?) {
        return None;
    }
    let civil = parse_civil_fields(s)?;
    civil_in_range(&civil).then_some(civil)
}

fn parse_civil_fields(s: &str) -> Option<CivilTime> {
    Some(CivilTime {
        year: parse_i32_range(s, 0, 4)?,
        month: parse_u32_range(s, 5, 7)?,
        day: parse_u32_range(s, 8, 10)?,
        hour: parse_u32_range(s, 11, 13)?,
        minute: parse_u32_range(s, 14, 16)?,
        second: parse_u32_range(s, 17, 19)?,
    })
}

fn is_datetime_separator(sep: u8) -> bool {
    sep == b'T' || sep == b' '
}

fn parse_i32_range(s: &str, start: usize, end: usize) -> Option<i32> {
    s.get(start..end)?.parse().ok()
}

fn parse_u32_range(s: &str, start: usize, end: usize) -> Option<u32> {
    s.get(start..end)?.parse().ok()
}

fn parse_i64_range(s: &str, start: usize, end: usize) -> Option<i64> {
    s.get(start..end)?.parse().ok()
}

fn civil_in_range(civil: &CivilTime) -> bool {
    date_in_range(civil) && time_in_range(civil)
}

fn date_in_range(civil: &CivilTime) -> bool {
    (1..=12).contains(&civil.month)
        && (1..=days_in_month(civil.year, civil.month)).contains(&civil.day)
}

fn time_in_range(civil: &CivilTime) -> bool {
    civil.hour <= 23 && civil.minute <= 59 && civil.second <= 60
}

fn frac_millis(frac: Option<&str>) -> Option<u32> {
    let Some(frac) = frac.filter(|f| !f.is_empty()) else {
        return Some(0);
    };
    let mut digits = frac.chars().take(3).collect::<String>();
    while digits.len() < 3 {
        digits.push('0');
    }
    digits.parse().ok()
}

fn civil_to_unix_ms(civil: CivilTime, millis: u32, tz: &str) -> Option<u64> {
    let unix_ms = i64::from(days_from_civil(civil.year, civil.month, civil.day))
        .checked_mul(86_400_000)?
        .checked_add(i64::from(civil.hour) * 3_600_000)?
        .checked_add(i64::from(civil.minute) * 60_000)?
        .checked_add(i64::from(civil.second) * 1_000)?
        .checked_add(i64::from(millis))?;
    u64::try_from(apply_tz_offset(unix_ms, tz)?).ok()
}

fn split_frac_and_tz(rest: &str) -> (Option<&str>, &str) {
    if rest.is_empty() {
        return (None, "");
    }
    if let Some(body) = rest.strip_prefix('.') {
        let tz_at = body.find(['Z', 'z', '+', '-']).unwrap_or(body.len());
        (Some(&body[..tz_at]), &body[tz_at..])
    } else {
        (None, rest)
    }
}

fn apply_tz_offset(unix_ms: i64, tz: &str) -> Option<i64> {
    let tz = tz.trim();
    if tz.is_empty() || tz.eq_ignore_ascii_case("Z") {
        return Some(unix_ms);
    }
    let (hh, mm) = tz_hours_minutes(&tz[1..])?;
    if !offset_in_range(hh, mm) {
        return None;
    }
    unix_ms.checked_sub(tz_sign(tz)? * (hh * 3_600_000 + mm * 60_000))
}

fn tz_sign(tz: &str) -> Option<i64> {
    match tz.as_bytes().first()? {
        b'+' => Some(1),
        b'-' => Some(-1),
        _ => None,
    }
}

fn offset_in_range(hh: i64, mm: i64) -> bool {
    (0..=23).contains(&hh) && (0..=59).contains(&mm)
}

fn tz_hours_minutes(body: &str) -> Option<(i64, i64)> {
    match offset_form(body) {
        OffsetForm::Colon => pair_i64(body, 0, 2, 3, 5),
        OffsetForm::Compact => pair_i64(body, 0, 2, 2, 4),
        OffsetForm::Hours => Some((parse_i64_range(body, 0, 2)?, 0)),
        OffsetForm::Invalid => None,
    }
}

enum OffsetForm {
    Colon,
    Compact,
    Hours,
    Invalid,
}

fn offset_form(body: &str) -> OffsetForm {
    if body.len() == 5 && body.as_bytes().get(2) == Some(&b':') {
        OffsetForm::Colon
    } else if body.len() == 4 {
        OffsetForm::Compact
    } else if body.len() == 2 {
        OffsetForm::Hours
    } else {
        OffsetForm::Invalid
    }
}

fn pair_i64(s: &str, a: usize, b: usize, c: usize, d: usize) -> Option<(i64, i64)> {
    Some((parse_i64_range(s, a, b)?, parse_i64_range(s, c, d)?))
}

fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap_year(year) => 29,
        2 => 28,
        _ => 0,
    }
}

fn is_leap_year(year: i32) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

pub(super) fn format_system_time_rfc3339_utc(now: std::time::SystemTime) -> String {
    let dur = now
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = dur.as_secs();
    let millis = dur.subsec_millis();
    let days = i32::try_from(secs / 86_400).unwrap_or(0);
    let tod = secs % 86_400;
    let (year, month, day) = civil_from_days(days);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{millis:03}Z",
        tod / 3600,
        (tod % 3600) / 60,
        tod % 60,
    )
}

/// Inverse of [`days_from_civil`] (Howard Hinnant).
fn civil_from_days(z: i32) -> (i32, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = u32::try_from(z - era * 146_097).unwrap_or(0);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = i32::try_from(yoe).unwrap_or(0) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { y + 1 } else { y };
    (year, month, day)
}

/// Howard Hinnant's public-domain `days_from_civil`.
fn days_from_civil(year: i32, month: u32, day: u32) -> i32 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = y.div_euclid(400);
    let yoe = (y - era * 400) as u32;
    let mp = if month > 2 { month - 3 } else { month + 9 };
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe as i32 - 719_468
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_space_separated_mining_timestamp() {
        let ms = parse_timestamp_string("2026-03-19 11:55:13.132").unwrap();
        assert_eq!(ms, 1_773_921_313_132);
    }

    #[test]
    fn parse_timestamp_refuses_chain_tag() {
        assert!(parse_timestamp_string("dynex:919876").is_none());
    }

    #[test]
    fn parse_timestamp_rejects_nonexistent_calendar_dates() {
        assert!(parse_timestamp_string("2026-02-30T12:00:00Z").is_none());
        assert!(parse_timestamp_string("2026-04-31T12:00:00Z").is_none());
        assert!(parse_timestamp_string("2026-02-29T12:00:00Z").is_none());
        assert!(parse_timestamp_string("2024-02-29T12:00:00Z").is_some());
    }

    #[test]
    fn parse_timestamp_rejects_out_of_range_offsets() {
        assert!(parse_timestamp_string("2026-03-19T12:00:00+99:99").is_none());
        assert!(parse_timestamp_string("2026-03-19T12:00:00+24:00").is_none());
        assert!(parse_timestamp_string("2026-03-19T12:00:00-00:60").is_none());
        assert!(parse_timestamp_string("2026-03-19T12:00:00+00:00").is_some());
    }

    #[test]
    fn parse_timestamp_rejects_trailing_offset_garbage() {
        assert!(parse_timestamp_string("2026-03-19T12:00:00+00:00junk").is_none());
        assert!(parse_timestamp_string("2026-03-19T12:00:00+0000junk").is_none());
        assert!(parse_timestamp_string("2026-03-19T12:00:00+00junk").is_none());
        assert!(parse_timestamp_string("2026-03-19T12:00:00+0000").is_some());
        assert!(parse_timestamp_string("2026-03-19T12:00:00+00").is_some());
    }

    #[test]
    fn format_system_time_rfc3339_utc_epoch() {
        assert_eq!(
            format_system_time_rfc3339_utc(std::time::UNIX_EPOCH),
            "1970-01-01T00:00:00.000Z"
        );
        assert_eq!(parse_timestamp_string("1970-01-01T00:00:00.000Z"), Some(0));
    }
}
