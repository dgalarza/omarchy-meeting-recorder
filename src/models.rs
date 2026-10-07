//! Which whisper model transcribes, where it is on disk, and fetching it.
//!
//! The model is picked with `--model` on the command line, or `model = "…"`
//! in `~/.config/omarchy-meeting-recorder/config.toml`, and is
//! `large-v3-turbo` otherwise. A name from `MODELS` is looked for in the app's
//! own model folder and in voxtype's (same files, no need to have them twice),
//! and downloaded when it is in neither. A path to a `.bin` file is used as is.
//!
//! Remote transcription is opt-in in that same file (`transcription_url`, or
//! `transcription = "remote"`). `--model` forces the local model for that run.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use gtk::glib;
use whisper_rs::DtwModelPreset;

use crate::transcribe::{Abort, Events, download, models_dir};

pub const DEFAULT: &str = "large-v3-turbo";

pub struct Model {
    pub name: &'static str,
    /// Download size in MB, for the text in the app.
    pub size_mb: u32,
    preset: DtwModelPreset,
}

pub const MODELS: [Model; 10] = [
    Model {
        name: "tiny",
        size_mb: 75,
        preset: DtwModelPreset::Tiny,
    },
    Model {
        name: "tiny.en",
        size_mb: 75,
        preset: DtwModelPreset::TinyEn,
    },
    Model {
        name: "base",
        size_mb: 142,
        preset: DtwModelPreset::Base,
    },
    Model {
        name: "base.en",
        size_mb: 142,
        preset: DtwModelPreset::BaseEn,
    },
    Model {
        name: "small",
        size_mb: 466,
        preset: DtwModelPreset::Small,
    },
    Model {
        name: "small.en",
        size_mb: 466,
        preset: DtwModelPreset::SmallEn,
    },
    Model {
        name: "medium",
        size_mb: 1500,
        preset: DtwModelPreset::Medium,
    },
    Model {
        name: "medium.en",
        size_mb: 1500,
        preset: DtwModelPreset::MediumEn,
    },
    Model {
        name: "large-v3",
        size_mb: 3100,
        preset: DtwModelPreset::LargeV3,
    },
    Model {
        name: "large-v3-turbo",
        size_mb: 1600,
        preset: DtwModelPreset::LargeV3Turbo,
    },
];

/// Set by `--model`; wins over the config file.
static OVERRIDE: Mutex<Option<String>> = Mutex::new(None);
/// Held while a model downloads, so a second caller waits instead of
/// fetching the same file again.
static DOWNLOADING: Mutex<()> = Mutex::new(());

pub fn set_override(name: &str) {
    *OVERRIDE.lock().unwrap() = Some(name.trim().to_owned());
}

pub fn config_file() -> PathBuf {
    std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .unwrap_or_else(|| glib::home_dir().join(".config"))
        .join(crate::APP_NAME)
        .join("config.toml")
}

/// Where speech-to-text runs. Local is the default; remote is opt-in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Transcription {
    Local,
    Remote(Remote),
}

/// A whisper.cpp server, or an OpenAI-compatible `/v1/audio/transcriptions`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Remote {
    /// What was written in the config: a base URL, or the full endpoint.
    pub url: String,
    pub api_key: Option<String>,
    /// Sent as the `model` form field. whisper.cpp's `/inference` ignores it.
    pub model: String,
}

/// Local unless the config opts into a remote server. `--model` forces local.
pub fn transcription() -> Result<Transcription, String> {
    let force_local = OVERRIDE.lock().unwrap().is_some();
    let text = std::fs::read_to_string(config_file()).unwrap_or_default();
    resolve(force_local, &text)
}

fn resolve(force_local: bool, text: &str) -> Result<Transcription, String> {
    if force_local {
        return Ok(Transcription::Local);
    }
    parse_transcription(text)
}

/// Top-level keys only: a `[[action]]` table must not be able to switch this.
fn parse_transcription(text: &str) -> Result<Transcription, String> {
    let values = top_level_values(text);
    let mode = values
        .get("transcription")
        .map(|value| value.to_ascii_lowercase())
        .filter(|value| !value.is_empty());
    let url = values
        .get("transcription_url")
        .cloned()
        .filter(|value| !value.is_empty());
    let api_key = values
        .get("transcription_api_key")
        .cloned()
        .filter(|value| !value.is_empty());
    let model = values
        .get("transcription_model")
        .cloned()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "whisper-1".to_owned());
    if let Some(key) = &api_key
        && key.chars().any(char::is_control)
    {
        return Err("transcription_api_key cannot contain a line break".into());
    }
    let url = match mode.as_deref() {
        Some("local") => return Ok(Transcription::Local),
        Some("remote") => url.ok_or_else(|| {
            "transcription = \"remote\" needs transcription_url in config.toml".to_owned()
        })?,
        Some(other) => {
            return Err(format!(
                "transcription must be \"local\" or \"remote\", not \"{other}\""
            ));
        }
        None => match url {
            Some(url) => url,
            None => return Ok(Transcription::Local),
        },
    };
    let url = url.trim().trim_end_matches('/').to_owned();
    crate::remote::endpoint(&url)?;
    Ok(Transcription::Remote(Remote {
        url,
        api_key,
        model,
    }))
}

/// `key = "value"` pairs before the first `[table]`. `#` starts a comment
/// outside quotes. The same line style as the rest of this file.
fn top_level_values(text: &str) -> std::collections::HashMap<String, String> {
    let mut values = std::collections::HashMap::new();
    let mut in_table = false;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.starts_with('[') {
            in_table = true;
            continue;
        }
        if in_table {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim();
        if key.is_empty() {
            continue;
        }
        values.insert(key.to_owned(), unquote(value));
    }
    values
}

/// A TOML string: `"..."` with backslash escapes, or `'...'` taken literally.
fn unquote(value: &str) -> String {
    let value = value.trim();
    if let Some(inner) = value.strip_prefix('\'').and_then(|v| v.split('\'').next()) {
        return inner.trim().to_owned();
    }
    let Some(rest) = value.strip_prefix('"') else {
        return value.split('#').next().unwrap_or("").trim().to_owned();
    };
    let mut out = String::new();
    let mut chars = rest.chars();
    while let Some(c) = chars.next() {
        match c {
            '"' => break,
            '\\' => match chars.next() {
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some(other) => out.push(other),
                None => break,
            },
            c => out.push(c),
        }
    }
    out.trim().to_owned()
}

/// What a meeting file should record: the remote url, or the local model name.
pub fn active_model_label() -> String {
    match transcription() {
        Ok(Transcription::Remote(remote)) => format!("remote:{}", remote.url),
        _ => configured(),
    }
}

/// The configured model: a name from `MODELS` or a path to a model file.
pub fn configured() -> String {
    if let Some(name) = OVERRIDE.lock().unwrap().clone() {
        return name;
    }
    std::fs::read_to_string(config_file())
        .ok()
        .and_then(|text| {
            text.lines().find_map(|line| {
                let (key, value) = line.split_once('=')?;
                (key.trim() == "model").then(|| {
                    let value = value.split('#').next().unwrap_or("");
                    value.trim().trim_matches('"').to_owned()
                })
            })
        })
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| DEFAULT.to_owned())
}

fn known(name: &str) -> Option<&'static Model> {
    let name = name.strip_prefix("ggml-").unwrap_or(name);
    let name = name.strip_suffix(".bin").unwrap_or(name);
    MODELS.iter().find(|m| m.name == name)
}

fn file_name(model: &Model) -> String {
    format!("ggml-{}.bin", model.name)
}

fn data_dir() -> PathBuf {
    std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .unwrap_or_else(|| glib::home_dir().join(".local/share"))
}

/// A complete model file: at least most of its expected size.
fn usable(path: &Path, model: Option<&Model>) -> bool {
    let min = model.map_or(10_000_000, |m| u64::from(m.size_mb) * 800_000);
    std::fs::metadata(path).is_ok_and(|m| m.is_file() && m.len() >= min)
}

/// The configured model's file, when it is on disk.
pub fn find() -> Option<PathBuf> {
    let name = configured();
    match known(&name) {
        Some(model) => [models_dir(), data_dir().join("voxtype/models")]
            .into_iter()
            .map(|dir| dir.join(file_name(model)))
            .find(|path| usable(path, Some(model))),
        None => {
            let path = PathBuf::from(&name);
            usable(&path, None).then_some(path)
        }
    }
}

/// For the app: the model's name and its size when it still has to be downloaded.
/// Remote transcription does not need a ggml file on this machine.
pub fn missing() -> Option<(String, u32)> {
    match transcription() {
        Ok(Transcription::Local) => {}
        Ok(Transcription::Remote(_)) | Err(_) => return None,
    }
    if find().is_some() {
        return None;
    }
    let name = configured();
    known(&name).map(|m| (m.name.to_owned(), m.size_mb))
}

/// The model file, downloaded first when needed. Blocking.
pub fn ensure(events: &Events, abort: &Abort) -> Result<PathBuf, String> {
    let _one_at_a_time = DOWNLOADING.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(path) = find() {
        return Ok(path);
    }
    let name = configured();
    let Some(model) = known(&name) else {
        let names: Vec<&str> = MODELS.iter().map(|m| m.name).collect();
        return Err(format!(
            "unknown model \"{name}\": use one of {} or a path to a model file",
            names.join(", ")
        ));
    };
    let target = models_dir().join(file_name(model));
    let url = format!(
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{}",
        file_name(model)
    );
    download(
        &url,
        &target,
        "Downloading model",
        u64::from(model.size_mb) * 800_000,
        events,
        abort,
    )?;
    Ok(target)
}

/// The attention-head preset for word times; None for a model file of unknown kind.
pub fn dtw_preset() -> Option<DtwModelPreset> {
    known(&configured()).map(|m| m.preset.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_and_file_names_both_resolve() {
        assert_eq!(known("large-v3").map(|m| m.name), Some("large-v3"));
        assert_eq!(known("ggml-small.en.bin").map(|m| m.name), Some("small.en"));
        assert!(known("gpt-5").is_none());
    }

    #[test]
    fn remote_stays_off_until_a_url_is_set() {
        assert_eq!(
            parse_transcription("model = \"small\"\n").unwrap(),
            Transcription::Local
        );
        assert_eq!(parse_transcription("").unwrap(), Transcription::Local);
        assert_eq!(
            parse_transcription("transcription = \"local\"\ntranscription_url = \"not a url\"\n")
                .unwrap(),
            Transcription::Local
        );
    }

    #[test]
    fn a_url_turns_transcription_remote() {
        let remote = match parse_transcription(
            r#"
            model = "small" # still the local name, unused while remote is on
            transcription_url = "http://10.0.0.8:8080/" # lab machine
            transcription_api_key = "tok#en"
            transcription_model = "large-v3"
            "#,
        )
        .unwrap()
        {
            Transcription::Remote(remote) => remote,
            Transcription::Local => panic!("expected remote"),
        };
        assert_eq!(remote.url, "http://10.0.0.8:8080");
        assert_eq!(remote.api_key.as_deref(), Some("tok#en"));
        assert_eq!(remote.model, "large-v3");
    }

    #[test]
    fn remote_without_a_url_is_an_error() {
        let err = parse_transcription("transcription = \"remote\"\n").unwrap_err();
        assert!(err.contains("transcription_url"), "{err}");
        let err = parse_transcription("transcription = \"cloud\"\n").unwrap_err();
        assert!(err.contains("local"), "{err}");
        let err = parse_transcription("transcription_url = \"10.0.0.8:8080\"\n").unwrap_err();
        assert!(err.contains("http://"), "{err}");
    }

    #[test]
    fn an_action_table_cannot_switch_transcription() {
        let text = r#"
transcription_url = "http://10.0.0.8:8080"

[[action]]
name = "x"
command = "y"
transcription = "local"
transcription_url = "http://10.9.9.9:1"
"#;
        match parse_transcription(text).unwrap() {
            Transcription::Remote(remote) => {
                assert_eq!(remote.url, "http://10.0.0.8:8080");
                assert_eq!(remote.model, "whisper-1");
            }
            Transcription::Local => panic!("the action table turned transcription off"),
        }
    }

    fn model_default(text: &str) -> String {
        match parse_transcription(text).unwrap() {
            Transcription::Remote(remote) => remote.model,
            Transcription::Local => String::new(),
        }
    }

    #[test]
    fn omitted_remote_model_defaults_to_whisper_1() {
        assert_eq!(
            model_default("transcription_url = \"http://10.0.0.8:8080\"\n"),
            "whisper-1"
        );
    }

    #[test]
    fn openai_url_is_accepted() {
        let text = "transcription = \"Remote\"\ntranscription_url = \"http://10.0.0.8:8000/v1/audio/transcriptions\"\n";
        match parse_transcription(text).unwrap() {
            Transcription::Remote(remote) => {
                assert_eq!(remote.url, "http://10.0.0.8:8000/v1/audio/transcriptions");
            }
            Transcription::Local => panic!("expected remote"),
        }
    }

    #[test]
    fn a_model_override_forces_local_even_when_the_file_says_remote() {
        let text = "transcription_url = \"http://10.0.0.8:8080\"\n";
        assert_eq!(resolve(true, text).unwrap(), Transcription::Local);
        assert!(matches!(
            resolve(false, text).unwrap(),
            Transcription::Remote(_)
        ));
    }
}
