// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
pub fn pieces<'a>(bytes: impl Fn(usize) -> &'a [u8], ids: &[usize]) -> Vec<String> {
    let mut pending = Vec::new();
    let mut out: Vec<String> = Vec::with_capacity(ids.len());
    for &id in ids {
        pending.extend_from_slice(bytes(id));
        let done = match std::str::from_utf8(&pending) {
            Ok(_) => pending.len(),
            // An incomplete character at the end waits for the next token; anything else is invalid.
            Err(e) if e.error_len().is_none() => e.valid_up_to(),
            Err(_) => pending.len(),
        };
        out.push(String::from_utf8_lossy(&pending[..done]).into_owned());
        pending.drain(..done);
    }
    if let Some(last) = out.last_mut() {
        *last += &String::from_utf8_lossy(&pending);
    }
    out
}

/// Trims ASCII whitespace, as the reference does.
pub fn trim(s: &str) -> &str {
    s.trim_matches([' ', '\t', '\n', '\r'])
}
