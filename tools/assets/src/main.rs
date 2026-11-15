use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    io::Read,
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
fn install() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mut options = BTreeMap::new();
    while let Some(key) = args.next() {
        if !["--root", "--dest", "--model-dir", "--manifest"].contains(&key.as_str()) {
            return Err(format!("unknown option {key}").into());
        }
        let value = args.next().ok_or("option needs a value")?;
        options.insert(key, PathBuf::from(value));
    }
    let root = options.get("--root").ok_or("--root is required")?;
    let dest = options.get("--dest").ok_or("--dest is required")?;
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
            let relative = format!("models/{hash}.gguf");
            fs::copy(&file, stage.join(&relative))?;
            spec["url"] = Value::String(relative);
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
