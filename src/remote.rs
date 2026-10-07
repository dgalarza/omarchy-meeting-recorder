//! Speech-to-text on another machine. Opt-in: with no remote config, nothing
//! here runs and whisper-rs stays on this computer.
//!
//! Two protocols, chosen from the URL:
//! - `http://host:8080` or `.../inference` is whisper.cpp's server
//!   (`POST /inference`, multipart). That is the one to run on a LAN machine.
//! - a URL whose path contains `audio/transcriptions` is the OpenAI-compatible
//!   `POST /v1/audio/transcriptions`.
//!
//! Each track is uploaded on its own as 16 kHz mono WAV. The server's
//! timestamps are seconds from the start of that file.

use std::io::Read;
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::time::Duration;

use ureq::unversioned::multipart::{Form, Part};

use crate::models::Remote;
use crate::transcribe::{Abort, CANCELLED, WHISPER_RATE};

/// How long one upload may take. Connect fails fast; a long recording is
/// allowed several times its own length, then given up.
#[derive(Debug, Clone, Copy)]
pub struct Timeouts {
    pub connect: Duration,
    pub total: Duration,
}

impl Timeouts {
    pub fn for_audio(samples: usize) -> Self {
        let secs = (samples / WHISPER_RATE) as u64;
        let total = secs
            .saturating_mul(4)
            .saturating_add(90)
            .clamp(180, 45 * 60);
        Self {
            connect: Duration::from_secs(10),
            total: Duration::from_secs(total),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Protocol {
    /// whisper.cpp `whisper-server`: `POST /inference`.
    WhisperCpp,
    /// `POST` the URL as given.
    OpenAi,
}

/// One word (or a slice of a segment) on the uploaded audio's timeline.
#[derive(Debug, Clone, PartialEq)]
pub struct TimedWord {
    pub text: String,
    pub start_ms: i64,
    pub end_ms: i64,
    pub no_speech: f32,
    pub segment: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RemoteTranscript {
    /// A whisper language code when the server named one.
    pub language: Option<String>,
    pub words: Vec<TimedWord>,
}

/// The URL to POST, and which form fields that server expects.
pub(crate) fn endpoint(url: &str) -> Result<(String, Protocol), String> {
    let url = url.trim().trim_end_matches('/');
    if url.is_empty() {
        return Err("transcription_url is empty".into());
    }
    if url.chars().any(char::is_whitespace) {
        return Err(format!(
            "transcription_url cannot contain spaces (got \"{url}\")"
        ));
    }
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        return Err(format!(
            "transcription_url must start with http:// or https:// (got \"{url}\")"
        ));
    }
    let rest = url.split_once("://").map(|(_, rest)| rest).unwrap_or("");
    if rest.is_empty() || rest.starts_with('/') || rest.starts_with(':') {
        return Err(format!("transcription_url \"{url}\" has no host"));
    }
    let path = rest.find('/').map(|i| &rest[i..]).unwrap_or("");
    let path = path.split('?').next().unwrap_or(path);
    let path_l = path.to_ascii_lowercase();
    if path_l.contains("audio/transcriptions") {
        Ok((url.to_owned(), Protocol::OpenAi))
    } else if path_l.ends_with("/inference") {
        Ok((url.to_owned(), Protocol::WhisperCpp))
    } else {
        Ok((format!("{url}/inference"), Protocol::WhisperCpp))
    }
}

fn origin_of(endpoint: &str) -> String {
    let Some((scheme, rest)) = endpoint.split_once("://") else {
        return endpoint.to_owned();
    };
    let host = rest.split(['/', '?']).next().unwrap_or(rest);
    format!("{scheme}://{host}/")
}

/// Opens a connection and reads any HTTP response. A host that is down fails
/// here, before a recording is uploaded. Any status counts as reachable; the
/// transcription request reports a broken server on its own.
pub fn check_reachable(remote: &Remote, abort: &Abort) -> Result<(), String> {
    probe(remote, abort, Duration::from_secs(10))
}

fn probe(remote: &Remote, abort: &Abort, budget: Duration) -> Result<(), String> {
    let (endpoint, _) = endpoint(&remote.url)?;
    let origin = origin_of(&endpoint);
    let timeouts = Timeouts {
        connect: budget,
        total: budget,
    };
    let gone = format!(
        "could not reach the transcription server at {origin}: the connection check stopped"
    );
    let origin_for_get = origin.clone();
    wait(abort, gone, move || {
        let agent = http_agent(timeouts);
        match agent.get(&origin_for_get).call() {
            Ok(_) => Ok(()),
            Err(err) => Err(transport_error(&origin_for_get, err)),
        }
    })
}

/// Uploads `samples` (16 kHz mono) and reads the transcript back.
pub fn fetch(
    remote: &Remote,
    samples: &[f32],
    language: &str,
    timeouts: Timeouts,
    abort: &Abort,
) -> Result<RemoteTranscript, String> {
    if samples.is_empty() {
        return Ok(RemoteTranscript {
            language: None,
            words: Vec::new(),
        });
    }
    let (endpoint, protocol) = endpoint(&remote.url)?;
    let wav = wav_bytes(samples);
    let language = language.to_owned();
    let model = remote.model.clone();
    let api_key = remote.api_key.clone();
    let endpoint_for_thread = endpoint.clone();
    let gone = format!("the transcription request to {endpoint} stopped before a response arrived");
    let body = wait(abort, gone, move || {
        post_audio(&Upload {
            endpoint: &endpoint_for_thread,
            protocol,
            api_key: api_key.as_deref(),
            model: &model,
            language: &language,
            wav: &wav,
            timeouts,
        })
    })?;
    let duration_ms = (samples.len() as i64).saturating_mul(1000) / WHISPER_RATE as i64;
    words_from_json(&body, duration_ms)
}

struct Upload<'a> {
    endpoint: &'a str,
    protocol: Protocol,
    api_key: Option<&'a str>,
    model: &'a str,
    language: &'a str,
    wav: &'a [u8],
    timeouts: Timeouts,
}

fn post_audio(upload: &Upload<'_>) -> Result<String, String> {
    let part = Part::bytes(upload.wav)
        .file_name("audio.wav")
        .mime_str("audio/wav")
        .map_err(|e| format!("could not build the audio upload: {e}"))?;
    let mut form = Form::new()
        .part("file", part)
        .text("response_format", "verbose_json")
        .text("temperature", "0");
    match upload.protocol {
        Protocol::WhisperCpp => {
            // `language=auto` is required: the server's own default is English.
            form = form
                .text("temperature_inc", "0.0")
                .text("language", upload.language)
                .text("no_speech_thold", "0.6")
                .text("suppress_nst", "true")
                .text("split_on_word", "true")
                .text("no_language_probabilities", "true");
        }
        Protocol::OpenAi => {
            form = form.text("model", upload.model);
            if upload.language != "auto" {
                form = form.text("language", upload.language);
            }
        }
    }
    let boundary = form.boundary().to_owned();
    let mut raw = Vec::new();
    form.read_to_end(&mut raw)
        .map_err(|e| format!("could not build the audio upload: {e}"))?;
    let content_type = format!("multipart/form-data; boundary={boundary}");

    let agent = http_agent(upload.timeouts);
    let mut request = agent.post(upload.endpoint).content_type(&content_type);
    if let Some(key) = upload.api_key.filter(|key| !key.is_empty()) {
        request = request.header("Authorization", format!("Bearer {key}"));
    }
    let mut response = request
        .send(&raw)
        .map_err(|err| transport_error(upload.endpoint, err))?;
    let status = response.status().as_u16();
    let body = response.body_mut().read_to_string().map_err(|e| {
        format!(
            "the transcription server at {} sent an unreadable response: {e}",
            upload.endpoint
        )
    })?;
    if !(200..300).contains(&status) {
        return Err(http_error(upload.endpoint, status, &body));
    }
    Ok(body)
}

/// Direct connection. A laptop HTTP proxy must not see meeting audio, and
/// must not be what we wait on when the LAN host is down.
fn http_agent(timeouts: Timeouts) -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_global(Some(timeouts.total))
        .timeout_connect(Some(timeouts.connect))
        .timeout_resolve(Some(timeouts.connect))
        .proxy(None)
        .http_status_as_error(false)
        .user_agent(crate::APP_NAME)
        .build()
        .into()
}

fn wait<T: Send + 'static>(
    abort: &Abort,
    disconnected: String,
    work: impl FnOnce() -> Result<T, String> + Send + 'static,
) -> Result<T, String> {
    if abort.load(Ordering::Relaxed) {
        return Err(CANCELLED.to_owned());
    }
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(work());
    });
    loop {
        if abort.load(Ordering::Relaxed) {
            return Err(CANCELLED.to_owned());
        }
        match rx.recv_timeout(Duration::from_millis(200)) {
            Ok(result) => {
                if abort.load(Ordering::Relaxed) {
                    return Err(CANCELLED.to_owned());
                }
                return result;
            }
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => return Err(disconnected),
        }
    }
}

fn transport_error(url: &str, err: ureq::Error) -> String {
    match &err {
        ureq::Error::Timeout(_)
        | ureq::Error::ConnectionFailed
        | ureq::Error::HostNotFound
        | ureq::Error::Io(_) => format!(
            "could not reach the transcription server at {url}: {err}. Check that it is running and that transcription_url is right"
        ),
        _ => format!("the transcription server at {url} failed: {err}"),
    }
}

fn http_error(url: &str, status: u16, body: &str) -> String {
    let snippet: String = body.chars().filter(|c| *c != '\r').take(240).collect();
    let snippet = snippet.split_whitespace().collect::<Vec<_>>().join(" ");
    if snippet.is_empty() {
        format!("the transcription server at {url} returned HTTP {status}")
    } else {
        format!("the transcription server at {url} returned HTTP {status}: {snippet}")
    }
}

fn wav_bytes(samples: &[f32]) -> Vec<u8> {
    let data_len = (samples.len() * 2) as u32;
    let mut data = Vec::with_capacity(44 + samples.len() * 2);
    data.extend(b"RIFF");
    data.extend(&(36 + data_len).to_le_bytes());
    data.extend(b"WAVEfmt ");
    data.extend(&16u32.to_le_bytes());
    data.extend(&1u16.to_le_bytes());
    data.extend(&1u16.to_le_bytes());
    data.extend(&(WHISPER_RATE as u32).to_le_bytes());
    data.extend(&((WHISPER_RATE as u32) * 2).to_le_bytes());
    data.extend(&2u16.to_le_bytes());
    data.extend(&16u16.to_le_bytes());
    data.extend(b"data");
    data.extend(&data_len.to_le_bytes());
    for sample in samples {
        let value = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
        data.extend(&value.to_le_bytes());
    }
    data
}

fn words_from_json(body: &str, duration_ms: i64) -> Result<RemoteTranscript, String> {
    let value: serde_json::Value = serde_json::from_str(body).map_err(|e| {
        let snippet: String = body.chars().take(160).collect();
        format!("the transcription server returned unreadable JSON ({e}): {snippet}")
    })?;
    if !value.is_object() {
        return Err("the transcription server returned JSON that is not an object".into());
    }
    let language = value
        .get("language")
        .or_else(|| value.get("detected_language"))
        .and_then(|v| v.as_str())
        .and_then(language_code);
    let duration_ms = duration_ms.max(0);
    let mut words = Vec::new();
    if let Some(segments) = value.get("segments").and_then(|v| v.as_array()) {
        for (index, segment) in segments.iter().enumerate() {
            words.extend(words_in_segment(segment, index, duration_ms));
        }
    }
    if words.is_empty() {
        let text = value
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim();
        if text.is_empty() {
            if let Some(message) = error_message(&value) {
                return Err(format!(
                    "the transcription server returned an error: {message}"
                ));
            }
        } else {
            words.extend(spread(text, 0, duration_ms, 0.0, 0));
        }
    }
    Ok(RemoteTranscript { language, words })
}

fn words_in_segment(segment: &serde_json::Value, index: usize, duration_ms: i64) -> Vec<TimedWord> {
    let no_speech = segment
        .get("no_speech_prob")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0) as f32;
    let start = ms(segment.get("start")).unwrap_or(0);
    let end = ms(segment.get("end")).unwrap_or(duration_ms).max(start);
    if let Some(list) = segment.get("words").and_then(|v| v.as_array()) {
        let parsed: Vec<(String, Option<i64>, Option<i64>)> = list
            .iter()
            .filter_map(|word| {
                let text = word.get("word").and_then(|v| v.as_str())?.trim().to_owned();
                if text.is_empty() {
                    None
                } else {
                    Some((text, ms(word.get("start")), ms(word.get("end"))))
                }
            })
            .collect();
        if !parsed.is_empty() {
            let span = (end - start).max(250 * parsed.len() as i64);
            let n = parsed.len() as i64;
            return parsed
                .into_iter()
                .enumerate()
                .map(|(i, (text, word_start, word_end))| {
                    let i = i as i64;
                    TimedWord {
                        text,
                        start_ms: word_start.unwrap_or(start + span * i / n),
                        end_ms: word_end.unwrap_or(start + span * (i + 1) / n),
                        no_speech,
                        segment: index,
                    }
                })
                .collect();
        }
    }
    let text = segment.get("text").and_then(|v| v.as_str()).unwrap_or("");
    spread(text, start, end, no_speech, index)
}

fn spread(text: &str, start: i64, end: i64, no_speech: f32, segment: usize) -> Vec<TimedWord> {
    let parts: Vec<&str> = text.split_whitespace().collect();
    if parts.is_empty() {
        return Vec::new();
    }
    let n = parts.len() as i64;
    let end = end.max(start + 250 * n);
    let span = end - start;
    parts
        .into_iter()
        .enumerate()
        .map(|(i, part)| {
            let i = i as i64;
            TimedWord {
                text: part.to_owned(),
                start_ms: start + span * i / n,
                end_ms: start + span * (i + 1) / n,
                no_speech,
                segment,
            }
        })
        .collect()
}

fn ms(value: Option<&serde_json::Value>) -> Option<i64> {
    value
        .and_then(|v| v.as_f64())
        .map(|secs| (secs * 1000.0).round().max(0.0) as i64)
}

fn language_code(reported: &str) -> Option<String> {
    let name = reported.trim().to_lowercase();
    if name.is_empty() {
        return None;
    }
    const NAMES: &[(&str, &str)] = &[
        ("english", "en"),
        ("dutch", "nl"),
        ("german", "de"),
        ("french", "fr"),
        ("spanish", "es"),
        ("italian", "it"),
        ("portuguese", "pt"),
    ];
    if let Some((_, code)) = NAMES.iter().find(|(full, _)| *full == name) {
        return Some((*code).to_owned());
    }
    let code = name.split(['-', '_']).next().unwrap_or(&name);
    if code.len() == 2 && code.chars().all(|c| c.is_ascii_alphabetic()) {
        Some(code.to_owned())
    } else {
        Some(name)
    }
}

fn error_message(value: &serde_json::Value) -> Option<String> {
    let err = value.get("error")?;
    err.as_str()
        .map(str::to_owned)
        .or_else(|| {
            err.get("message")
                .and_then(|m| m.as_str())
                .map(str::to_owned)
        })
        .filter(|message| !message.is_empty())
}

#[cfg(test)]
pub(crate) struct Captured {
    pub head: String,
    pub body: Vec<u8>,
}

#[cfg(test)]
pub(crate) fn serve_once(
    status: u16,
    reason: &'static str,
    body: String,
) -> (String, mpsc::Receiver<Captured>) {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let (mut sock, _) = listener.accept().unwrap();
        let (head, req_body) = read_request(&mut sock).unwrap();
        let _ = tx.send(Captured {
            head,
            body: req_body,
        });
        write_response(&mut sock, status, reason, &body).unwrap();
    });
    (url, rx)
}

#[cfg(test)]
pub(crate) fn serve_stall() -> String {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    std::thread::spawn(move || {
        if let Ok((mut sock, _)) = listener.accept() {
            let _ = read_request(&mut sock);
            std::thread::sleep(Duration::from_secs(5));
        }
    });
    url
}

#[cfg(test)]
fn read_request(sock: &mut std::net::TcpStream) -> std::io::Result<(String, Vec<u8>)> {
    sock.set_read_timeout(Some(Duration::from_secs(5)))?;
    let mut buf = Vec::new();
    let mut tmp = [0u8; 8192];
    let header_end = loop {
        if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break pos + 4;
        }
        let n = sock.read(&mut tmp)?;
        if n == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "eof before headers",
            ));
        }
        buf.extend_from_slice(&tmp[..n]);
    };
    let head = String::from_utf8_lossy(&buf[..header_end]).into_owned();
    let mut body = buf[header_end..].to_vec();
    if let Some(len) = content_length(&head) {
        while body.len() < len {
            let n = sock.read(&mut tmp)?;
            if n == 0 {
                break;
            }
            body.extend_from_slice(&tmp[..n]);
        }
        body.truncate(len);
    }
    Ok((head, body))
}

#[cfg(test)]
fn content_length(head: &str) -> Option<usize> {
    head.lines().find_map(|line| {
        let (name, value) = line.split_once(':')?;
        name.eq_ignore_ascii_case("content-length")
            .then(|| value.trim().parse().ok())
            .flatten()
    })
}

#[cfg(test)]
fn write_response(
    sock: &mut std::net::TcpStream,
    status: u16,
    reason: &str,
    body: &str,
) -> std::io::Result<()> {
    use std::io::Write;
    let bytes = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    sock.write_all(bytes.as_bytes())?;
    sock.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn remote(url: &str) -> Remote {
        Remote {
            url: url.to_owned(),
            api_key: None,
            model: "whisper-1".into(),
        }
    }

    fn quick() -> Timeouts {
        Timeouts {
            connect: Duration::from_secs(2),
            total: Duration::from_secs(5),
        }
    }

    #[test]
    fn urls_pick_a_protocol() {
        let (url, protocol) = endpoint("http://192.168.1.20:8080/").unwrap();
        assert_eq!(url, "http://192.168.1.20:8080/inference");
        assert_eq!(protocol, Protocol::WhisperCpp);
        let (url, protocol) = endpoint("http://192.168.1.20:8080/inference").unwrap();
        assert_eq!(url, "http://192.168.1.20:8080/inference");
        assert_eq!(protocol, Protocol::WhisperCpp);
        let (url, protocol) = endpoint("http://192.168.1.20:8000/v1/audio/transcriptions").unwrap();
        assert_eq!(url, "http://192.168.1.20:8000/v1/audio/transcriptions");
        assert_eq!(protocol, Protocol::OpenAi);
        assert!(endpoint("192.168.1.20:8080").is_err());
        assert!(endpoint("http://").is_err());
    }

    #[test]
    fn timeouts_have_a_floor_and_a_ceiling() {
        let short = Timeouts::for_audio(WHISPER_RATE);
        assert_eq!(short.connect, Duration::from_secs(10));
        assert_eq!(short.total, Duration::from_secs(180));
        let long = Timeouts::for_audio(WHISPER_RATE * 60 * 60 * 5);
        assert_eq!(long.total, Duration::from_secs(45 * 60));
    }

    #[test]
    fn wav_is_16khz_mono_pcm() {
        let wav = wav_bytes(&[0.0, 1.0, -1.0]);
        assert_eq!(&wav[..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(u32::from_le_bytes(wav[24..28].try_into().unwrap()), 16_000);
        assert_eq!(u16::from_le_bytes(wav[22..24].try_into().unwrap()), 1);
        assert_eq!(wav.len(), 44 + 6);
    }

    #[test]
    fn word_times_are_kept_and_a_bare_segment_is_spread() {
        let with_words = words_from_json(
            r#"{"language":"english","segments":[{"text":" Hello there","start":0.1,"end":0.8,"no_speech_prob":0.01,"words":[{"word":" Hello","start":0.1,"end":0.4},{"word":" there","start":0.4,"end":0.8}]}]}"#,
            1000,
        )
        .unwrap();
        assert_eq!(with_words.language.as_deref(), Some("en"));
        assert_eq!(with_words.words[0].text, "Hello");
        assert_eq!(with_words.words[0].start_ms, 100);
        assert_eq!(with_words.words[1].end_ms, 800);

        let spread = words_from_json(
            r#"{"language":"en","segments":[{"text":"Hello there","start":0.2,"end":1.4,"no_speech_prob":0.05}]}"#,
            2000,
        )
        .unwrap();
        assert_eq!(spread.words.len(), 2);
        assert_eq!(spread.words[0].start_ms, 200);
        assert_eq!(spread.words[1].end_ms, 1400);
        assert_eq!(spread.words[0].segment, spread.words[1].segment);

        let text_only = words_from_json(r#"{"text":"Just text"}"#, 2000).unwrap();
        assert_eq!(text_only.words.len(), 2);
        assert_eq!(text_only.words[0].start_ms, 0);
        assert_eq!(text_only.words[1].end_ms, 2000);
    }

    #[test]
    fn a_server_error_object_is_reported() {
        let err = words_from_json(r#"{"error":"model not loaded"}"#, 1000).unwrap_err();
        assert!(err.contains("model not loaded"), "{err}");
    }

    #[test]
    fn whisper_cpp_upload_posts_inference_with_the_wav() {
        let (url, hit) = serve_once(
            200,
            "OK",
            r#"{"language":"en","text":"Hi","segments":[{"text":"Hi","start":0.0,"end":0.5}]}"#
                .into(),
        );
        let mut settings = remote(&url);
        settings.api_key = Some("secret".into());
        let transcript = fetch(&settings, &[0.1, -0.1], "en", quick(), &Abort::default()).unwrap();
        assert_eq!(transcript.words[0].text, "Hi");
        let captured = hit.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(
            captured.head.starts_with("POST /inference "),
            "{}",
            captured.head
        );
        assert!(
            captured
                .head
                .to_ascii_lowercase()
                .contains("authorization: bearer secret"),
            "{}",
            captured.head
        );
        let body = String::from_utf8_lossy(&captured.body);
        assert!(body.contains("RIFF"), "wav missing");
        assert!(body.contains("name=\"language\""));
        assert!(body.contains("name=\"response_format\""));
        assert!(body.contains("verbose_json"));
        assert!(!body.contains("name=\"model\""));
    }

    #[test]
    fn openai_upload_posts_the_transcriptions_path() {
        let (base, hit) = serve_once(200, "OK", r#"{"text":"Hi"}"#.into());
        let url = format!("{base}/v1/audio/transcriptions");
        let settings = Remote {
            url,
            api_key: None,
            model: "faster-whisper".into(),
        };
        let transcript = fetch(&settings, &[0.2], "auto", quick(), &Abort::default()).unwrap();
        assert_eq!(transcript.words.len(), 1);
        let captured = hit.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(
            captured.head.starts_with("POST /v1/audio/transcriptions "),
            "{}",
            captured.head
        );
        let body = String::from_utf8_lossy(&captured.body);
        assert!(body.contains("faster-whisper"));
        assert!(!body.contains("name=\"language\""));
        assert!(!body.contains("suppress_nst"));
    }

    #[test]
    fn http_errors_include_the_status() {
        let (url, _hit) = serve_once(503, "Unavailable", "model is loading".into());
        let err = fetch(&remote(&url), &[0.1], "en", quick(), &Abort::default()).unwrap_err();
        assert!(err.contains("HTTP 503"), "{err}");
        assert!(err.contains("model is loading"), "{err}");
    }

    #[test]
    fn a_closed_port_fails_without_waiting_out_the_transcription_budget() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        drop(listener);
        let started = std::time::Instant::now();
        let err = fetch(
            &remote(&url),
            &[0.1],
            "en",
            Timeouts {
                connect: Duration::from_secs(2),
                total: Duration::from_secs(30),
            },
            &Abort::default(),
        )
        .unwrap_err();
        assert!(err.contains("could not reach"), "{err}");
        assert!(started.elapsed() < Duration::from_secs(8), "{err}");
    }

    #[test]
    fn a_stall_gives_up_at_the_timeout() {
        let url = serve_stall();
        let started = std::time::Instant::now();
        let err = fetch(
            &remote(&url),
            &[0.1],
            "en",
            Timeouts {
                connect: Duration::from_secs(1),
                total: Duration::from_secs(2),
            },
            &Abort::default(),
        )
        .unwrap_err();
        assert!(
            err.contains("could not reach") || err.contains("timed out") || err.contains("failed"),
            "{err}"
        );
        assert!(started.elapsed() < Duration::from_secs(8), "{err}");
    }

    #[test]
    fn cancel_returns_before_a_hung_server_finishes() {
        let url = serve_stall();
        let abort = Abort::default();
        let flag = abort.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(300));
            flag.store(true, Ordering::Relaxed);
        });
        let started = std::time::Instant::now();
        let err = fetch(
            &remote(&url),
            &[0.1],
            "en",
            Timeouts {
                connect: Duration::from_secs(2),
                total: Duration::from_secs(30),
            },
            &abort,
        )
        .unwrap_err();
        assert_eq!(err, CANCELLED);
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    #[test]
    fn an_unreachable_host_fails_the_check() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        drop(listener);
        let started = std::time::Instant::now();
        let err = probe(&remote(&url), &Abort::default(), Duration::from_secs(2)).unwrap_err();
        assert!(err.contains("could not reach"), "{err}");
        assert!(started.elapsed() < Duration::from_secs(8), "{err}");
    }
}
