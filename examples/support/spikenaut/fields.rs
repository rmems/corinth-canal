// SPDX-License-Identifier: Apache-2.0 OR MIT
//! JSON field extraction shared by domain detection and mapping.

use serde_json::{Map, Number, Value};

pub(super) fn flatten_telemetry_object(object: &Map<String, Value>) -> Map<String, Value> {
    let mut flat = object.clone();
    if let Some(Value::Object(inner)) = object.get("telemetry") {
        for (key, value) in inner {
            flat.entry(key.clone()).or_insert_with(|| value.clone());
        }
    }
    flat
}

pub(super) fn has_key(fields: &Map<String, Value>, key: &str) -> bool {
    fields.contains_key(key)
}

pub(super) fn has_pair(fields: &Map<String, Value>, left: &str, right: &str) -> bool {
    has_key(fields, left) && has_key(fields, right)
}

pub(super) fn finite_field(fields: &Map<String, Value>, key: &str) -> Option<f32> {
    finite_value(fields.get(key)?)
}

pub(super) fn first_finite(fields: &Map<String, Value>, keys: &[&str]) -> Option<f32> {
    keys.iter().find_map(|key| finite_field(fields, key))
}

pub(super) fn integer_field(fields: &Map<String, Value>, key: &str) -> Option<u64> {
    match fields.get(key)? {
        Value::Number(n) => number_as_u64(n),
        Value::String(s) => s.parse().ok(),
        _ => None,
    }
}

fn number_as_u64(n: &Number) -> Option<u64> {
    n.as_u64().or_else(|| finite_trunc_u64(n.as_f64()?))
}

fn finite_trunc_u64(value: f64) -> Option<u64> {
    let in_range = value.is_finite() && value >= 0.0 && value == value.trunc();
    (in_range && value <= u64::MAX as f64).then_some(value as u64)
}

fn finite_value(value: &Value) -> Option<f32> {
    match value {
        Value::Number(n) => {
            let parsed = n.as_f64()? as f32;
            parsed.is_finite().then_some(parsed)
        }
        Value::String(s) => {
            let parsed: f32 = s.parse().ok()?;
            parsed.is_finite().then_some(parsed)
        }
        _ => None,
    }
}
