use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    process::Command,
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn digest(path: &Path) -> Result<(u64, String)> {
    let mut file = fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut count = 0;
    let mut bytes = [0u8; 65536];
    loop {
        let n = file.read(&mut bytes)?;
        if n == 0 {
            break;
        }
        hash.update(&bytes[..n]);
        count += n as u64;
    }
    Ok((count, format!("{:x}", hash.finalize())))
}
fn verify(path: &Path, bytes: u64, hash: &str) -> Result<()> {
    let (actual, checksum) = digest(path)?;
    if actual != bytes || checksum != hash {
        return Err(format!("{} does not match its pinned checksum/size", path.display()).into());
    }
    Ok(())
}
fn copy_dir(source: &Path, dest: &Path) -> Result<()> {
    fs::create_dir_all(dest)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let to = dest.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir(&entry.path(), &to)?;
        } else {
            fs::copy(entry.path(), to)?;
        }
    }
    Ok(())
}
fn validate_destination(root: &Path, dest: &Path) -> Result<()> {
    let source = fs::canonicalize(root)?;
    let dist = fs::canonicalize(root.join("dist"))?;
    let resolved = if dest.exists() {
        fs::canonicalize(dest)?
    } else {
        fs::canonicalize(dest.parent().ok_or("destination needs a parent")?)?
            .join(dest.file_name().ok_or("destination needs a name")?)
    };
    if source.starts_with(&resolved) || resolved.starts_with(dist) {
        return Err(
            "destination must not replace the source repository or be inside its dist directory"
                .into(),
        );
    }
    Ok(())
}
fn install_model(
    file: &Path,
    stage: &Path,
    spec: &mut Value,
    part_bytes: Option<u64>,
) -> Result<()> {
    let hash = spec["sha256"].as_str().ok_or("missing model checksum")?;
    let bytes = spec["bytes"].as_u64().ok_or("missing model size")?;
    let name = format!("{hash}.gguf");
    let relative = format!("models/{name}");
    let object = spec.as_object_mut().ok_or("model must be an object")?;
    object.remove("parts");
    object.insert("url".into(), Value::String(relative.clone()));
    if let Some(limit) = part_bytes.filter(|&limit| bytes > limit) {
        if limit == 0 {
            return Err("model part size must be positive".into());
        }
        let mut input = fs::File::open(file)?;
        let mut remaining = bytes;
        let mut parts = Vec::new();
        while remaining > 0 {
            let count = remaining.min(limit);
            let part = format!("{name}.part-{:03}", parts.len());
            let mut output = fs::File::create(stage.join("models").join(&part))?;
            if std::io::copy(&mut (&mut input).take(count), &mut output)? != count {
                return Err("model changed while being installed".into());
            }
            output.flush()?;
            parts.push(serde_json::json!({"url": part, "bytes": count}));
            remaining -= count;
        }
        object.insert("parts".into(), Value::Array(parts));
    } else {
        fs::copy(file, stage.join(relative))?;
    }
    Ok(())
}
fn install() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mut options = BTreeMap::new();
    while let Some(key) = args.next() {
        if ![
            "--root",
            "--dest",
            "--model-dir",
            "--manifest",
            "--model-part-bytes",
        ]
        .contains(&key.as_str())
        {
            return Err(format!("unknown option {key}").into());
        }
        let value = args.next().ok_or("option needs a value")?;
        options.insert(key, PathBuf::from(value));
    }
    let root = options.get("--root").ok_or("--root is required")?;
    let dest = options.get("--dest").ok_or("--dest is required")?;
    let part_bytes = options
        .get("--model-part-bytes")
        .map(|value| -> Result<u64> {
            let bytes: u64 = value.to_str().ok_or("invalid model part size")?.parse()?;
            if !(1024 * 1024..=512 * 1024 * 1024).contains(&bytes) {
                return Err("model part size must be between 1 and 512 MiB".into());
            }
            Ok(bytes)
        })
        .transpose()?;
    let cache = options
        .get("--model-dir")
        .cloned()
        .unwrap_or_else(|| root.join("models"));
    let manifest_path = options
        .get("--manifest")
        .cloned()
        .unwrap_or_else(|| root.join("models.lock.json"));
    let mut manifest: Value = serde_json::from_slice(&fs::read(&manifest_path)?)?;
    if manifest["schema"] != 1 || manifest["sampleRate"] != 16000 {
        return Err("unsupported model manifest".into());
    }
    let defaults = manifest["defaults"].clone();
    let models = manifest["models"]
        .as_object()
        .ok_or("models must be an object")?;
    for key in ["draft", "final"] {
        if !models.contains_key(defaults[key].as_str().ok_or("missing default model")?) {
            return Err("default model not in manifest".into());
        }
    }
    for file in [
        "worker.js",
        "worker-entry.js",
        "speechworks.js",
        "pool-owner.js",
        "runtime-loader.js",
        "runtime-config.js",
        "model-cache.js",
        "model-source.js",
        "transcript.js",
        "capture-worklet.js",
        "speechworks_web.js",
        "speechworks_web_bg.wasm",
        "relaxed/speechworks_web.js",
        "relaxed/speechworks_web_bg.wasm",
    ] {
        if !root.join("dist").join(file).is_file() {
            return Err(format!("missing built artifact {file}; run tools/build-web.sh").into());
        }
    }
    fs::create_dir_all(&cache)?;
    let parent = dest
        .parent()
        .ok_or("destination requires parent directory")?;
    fs::create_dir_all(parent)?;
    validate_destination(root, dest)?;
    let stage = parent.join(format!(".speechworks-stage-{}", std::process::id()));
    if stage.exists() {
        return Err("staging directory already exists".into());
    }
    fs::create_dir(&stage)?;
    let result = (|| -> Result<()> {
        copy_dir(&root.join("dist"), &stage)?;
        fs::create_dir_all(stage.join("models"))?;
        for (id, spec) in manifest["models"]
            .as_object_mut()
            .ok_or("models must be an object")?
        {
            let hash = spec["sha256"]
                .as_str()
                .ok_or("missing model checksum")?
                .to_string();
            let bytes = spec["bytes"].as_u64().ok_or("missing model size")?;
            if hash.len() != 64
                || !hash
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                || bytes == 0
                || bytes > 512 * 1024 * 1024
            {
                return Err(format!("invalid checksum/size for {id}").into());
            }
            let source = spec["source"].as_str().ok_or("missing model source")?;
            if !source.starts_with("https://") || source.contains('\n') {
                return Err("model downloads require HTTPS".into());
            }
            let file = cache.join(format!("{hash}.gguf"));
            if !file.is_file() {
                let original =
                    cache.join(source.rsplit('/').next().ok_or("model filename missing")?);
                if original.is_file() {
                    verify(&original, bytes, &hash)?;
                    fs::copy(&original, &file)?;
                } else {
                    let temp = cache.join(format!("{hash}.{}.tmp", std::process::id()));
                    eprintln!("Downloading {id} ({bytes} bytes)");
                    let status = Command::new("curl")
                        .args([
                            "--fail",
                            "--location",
                            "--retry",
                            "3",
                            "--proto",
                            "=https",
                            "--proto-redir",
                            "=https",
                            "--output",
                        ])
                        .arg(&temp)
                        .arg(source)
                        .status()?;
                    if !status.success() {
                        let _ = fs::remove_file(&temp);
                        return Err(format!("download failed: {id}").into());
                    }
                    if let Err(e) = verify(&temp, bytes, &hash) {
                        let _ = fs::remove_file(&temp);
                        return Err(e);
                    }
                    fs::rename(temp, &file)?;
                }
            }
            verify(&file, bytes, &hash)?;
            install_model(&file, &stage, spec, part_bytes)?;
            // Download locations can include deployment-only signed URLs. The
            // browser only needs the verified same-origin artifact URL.
            spec.as_object_mut()
                .ok_or("model must be an object")?
                .remove("source");
        }
        fs::write(
            stage.join("models.json"),
            serde_json::to_vec_pretty(&manifest)?,
        )?;
        let backup = parent.join(format!(".speechworks-previous-{}", std::process::id()));
        if backup.exists() {
            return Err("backup destination already exists".into());
        }
        let previous = dest.exists();
        if previous {
            fs::rename(dest, &backup)?;
        }
        if let Err(error) = fs::rename(&stage, dest) {
            if previous {
                fs::rename(&backup, dest)?;
            }
            return Err(error.into());
        }
        if previous {
            fs::remove_dir_all(backup)?;
        }
        Ok(())
    })();
    if stage.exists() {
        let _ = fs::remove_dir_all(&stage);
    }
    result?;
    println!(
        "Installed verified Speechworks runtime and models at {}",
        dest.display()
    );
    Ok(())
}
fn main() {
    if let Err(error) = install() {
        eprintln!("speechworks-assets: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn model_parts_roundtrip_without_oversized_files_or_stale_manifest_parts() {
        let root = std::env::temp_dir().join(format!(
            "speechworks-model-parts-test-{}",
            std::process::id()
        ));
        fs::create_dir_all(root.join("models")).unwrap();
        let file = root.join("input");
        fs::write(&file, b"abcdefghijk").unwrap();
        let (_, hash) = digest(&file).unwrap();
        let mut spec = serde_json::json!({"sha256": hash, "bytes": 11});
        install_model(&file, &root, &mut spec, Some(4)).unwrap();
        assert!(!root.join(spec["url"].as_str().unwrap()).exists());
        let mut joined = Vec::new();
        let parts = spec["parts"].as_array().unwrap();
        assert_eq!(parts.len(), 3);
        for part in parts {
            let bytes = fs::read(root.join("models").join(part["url"].as_str().unwrap())).unwrap();
            assert_eq!(bytes.len() as u64, part["bytes"].as_u64().unwrap());
            assert!(bytes.len() <= 4);
            joined.extend(bytes);
        }
        assert_eq!(joined, b"abcdefghijk");
        for limit in [Some(11), Some(20), None] {
            install_model(&file, &root, &mut spec, limit).unwrap();
            assert!(spec.get("parts").is_none());
            verify(&root.join(spec["url"].as_str().unwrap()), 11, &hash).unwrap();
        }
        assert!(install_model(&file, &root, &mut spec, Some(0)).is_err());
        fs::write(&file, b"short").unwrap();
        assert!(install_model(&file, &root, &mut spec, Some(4)).is_err());
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn rejects_wrong_hash_and_wrong_size() {
        let path =
            std::env::temp_dir().join(format!("speechworks-checksum-test-{}", std::process::id()));
        fs::write(&path, b"abc").unwrap();
        let good = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        assert!(verify(&path, 3, good).is_ok());
        assert!(verify(&path, 4, good).is_err());
        assert!(verify(&path, 3, &"0".repeat(64)).is_err());
        fs::remove_file(path).unwrap();
    }
    #[test]
    fn rejects_recursive_or_destructive_install_destinations() {
        let root = std::env::temp_dir().join(format!(
            "speechworks-destination-test-{}",
            std::process::id()
        ));
        fs::create_dir_all(root.join("dist")).unwrap();
        assert!(validate_destination(&root, &root).is_err());
        assert!(validate_destination(&root, &root.join("dist/runtime")).is_err());
        assert!(validate_destination(&root, root.parent().unwrap()).is_err());
        assert!(validate_destination(&root, &root.join("installed")).is_ok());
        fs::remove_dir_all(root).unwrap();
    }
}
