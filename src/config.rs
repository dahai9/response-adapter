use std::collections::HashMap;
use std::env;
use std::fs;
use std::net::SocketAddr;
use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result};

const DEFAULT_BASE_URL: &str = "https://api.deepseek.com";
const DEFAULT_MODEL: &str = "deepseek-v4-pro";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    DeepSeek,
    OpenAICompatible,
}

impl Backend {
    pub fn from_env() -> Self {
        match env::var("ADAPTER_BACKEND").ok().as_deref() {
            Some("openai") | Some("openai-compatible") => Self::OpenAICompatible,
            _ => Self::DeepSeek,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Config {
    pub api_key: String,
    pub base_url: String,
    pub model_map: HashMap<String, String>,
    pub model_override: Option<String>,
    pub thinking: Option<ThinkingMode>,
    pub timeout: Duration,
    pub backend: Backend,
    pub listen: SocketAddr,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThinkingMode {
    Enabled,
    Disabled,
}

impl ThinkingMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Enabled => "enabled",
            Self::Disabled => "disabled",
        }
    }
}

impl Config {
    pub fn from_env() -> Result<Self> {
        load_dotenv(".env")?;
        let api_key = env::var("DEEPSEEK_API_KEY").context("DEEPSEEK_API_KEY is not set")?;
        let base_url =
            non_empty_env("DEEPSEEK_BASE_URL").unwrap_or_else(|| DEFAULT_BASE_URL.into());
        let model_override = non_empty_env("DEEPSEEK_MODEL");
        let model_map = non_empty_env("DEEPSEEK_MODEL_MAP")
            .and_then(|value| serde_json::from_str::<HashMap<String, String>>(&value).ok())
            .unwrap_or_default();
        let thinking = match non_empty_env("DEEPSEEK_THINKING").as_deref() {
            Some("enabled") => Some(ThinkingMode::Enabled),
            Some("disabled") => Some(ThinkingMode::Disabled),
            Some(other) => {
                anyhow::bail!("DEEPSEEK_THINKING must be enabled or disabled, got {other}")
            }
            None => None,
        };
        let timeout = non_empty_env("DEEPSEEK_TIMEOUT")
            .and_then(|value| value.parse::<u64>().ok())
            .map(Duration::from_secs)
            .unwrap_or_else(|| Duration::from_secs(120));
        let host = non_empty_env("ADAPTER_HOST").unwrap_or_else(|| "127.0.0.1".into());
        let port = non_empty_env("ADAPTER_PORT")
            .and_then(|value| value.parse::<u16>().ok())
            .unwrap_or(8787);
        let backend = Backend::from_env();
        let listen = format!("{host}:{port}")
            .parse()
            .with_context(|| format!("invalid listen address {host}:{port}"))?;

        Ok(Self {
            api_key,
            base_url,
            model_map,
            model_override,
            thinking,
            timeout,
            backend,
            listen,
        })
    }

    /// Resolve the upstream model name from the incoming request model.
    ///
    /// Priority:
    /// 1. `model_map` match on the incoming model name
    /// 2. `DEEPSEEK_MODEL` override (if set)
    /// 3. The incoming request's `model` field
    /// 4. Default model
    pub fn resolve_model(&self, request_model: Option<&str>) -> String {
        // Try mapping first
        if let Some(request_model) = request_model {
            if let Some(mapped) = self.model_map.get(request_model) {
                return mapped.clone();
            }
        }
        // Then override, then request model, then default
        self.model_override
            .clone()
            .or_else(|| request_model.map(ToOwned::to_owned))
            .unwrap_or_else(|| DEFAULT_MODEL.to_string())
    }

    pub fn default_model() -> &'static str {
        DEFAULT_MODEL
    }
}

fn non_empty_env(name: &str) -> Option<String> {
    env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn load_dotenv(path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    let Ok(contents) = fs::read_to_string(path) else {
        return Ok(());
    };
    for (key, value) in parse_dotenv(&contents) {
        if env::var_os(&key).is_none() {
            env::set_var(key, value);
        }
    }
    Ok(())
}

fn parse_dotenv(contents: &str) -> HashMap<String, String> {
    let mut values = HashMap::new();
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((key, raw_value)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim();
        if key.is_empty() {
            continue;
        }
        values.insert(key.to_string(), unquote(raw_value.trim()));
    }
    values
}

fn unquote(value: &str) -> String {
    if value.len() >= 2 {
        let bytes = value.as_bytes();
        if (bytes[0] == b'"' && bytes[value.len() - 1] == b'"')
            || (bytes[0] == b'\'' && bytes[value.len() - 1] == b'\'')
        {
            return value[1..value.len() - 1].to_string();
        }
    }
    value.to_string()
}

#[cfg(test)]
mod tests {
    use super::{Config, parse_dotenv};
    use std::collections::HashMap;

    #[test]
    fn parses_simple_dotenv() {
        let got = parse_dotenv("A=1\n# nope\nB=\"two\"\nC='three'\n");
        assert_eq!(got.get("A").unwrap(), "1");
        assert_eq!(got.get("B").unwrap(), "two");
        assert_eq!(got.get("C").unwrap(), "three");
    }

    #[test]
    fn resolve_model_uses_mapping() {
        let mut model_map = HashMap::new();
        model_map.insert("gpt-5.4".into(), "deepseek-v4-flash".into());
        model_map.insert("gpt-5.5".into(), "deepseek-v4-pro".into());
        let cfg = Config {
            api_key: "sk-test".into(),
            base_url: "https://api.deepseek.com".into(),
            model_map,
            model_override: None,
            thinking: None,
            timeout: std::time::Duration::from_secs(120),
            backend: super::Backend::DeepSeek,
            listen: "127.0.0.1:8787".parse().unwrap(),
        };
        assert_eq!(cfg.resolve_model(Some("gpt-5.4")), "deepseek-v4-flash");
        assert_eq!(cfg.resolve_model(Some("gpt-5.5")), "deepseek-v4-pro");
    }

    #[test]
    fn resolve_model_falls_through_to_override() {
        let cfg = Config {
            api_key: "sk-test".into(),
            base_url: "https://api.deepseek.com".into(),
            model_map: HashMap::new(),
            model_override: Some("deepseek-chat".into()),
            thinking: None,
            timeout: std::time::Duration::from_secs(120),
            backend: super::Backend::DeepSeek,
            listen: "127.0.0.1:8787".parse().unwrap(),
        };
        // Override beats request model
        assert_eq!(cfg.resolve_model(Some("gpt-5.5")), "deepseek-chat");
    }

    #[test]
    fn resolve_model_falls_through_to_request_model() {
        let cfg = Config {
            api_key: "sk-test".into(),
            base_url: "https://api.deepseek.com".into(),
            model_map: HashMap::new(),
            model_override: None,
            thinking: None,
            timeout: std::time::Duration::from_secs(120),
            backend: super::Backend::DeepSeek,
            listen: "127.0.0.1:8787".parse().unwrap(),
        };
        assert_eq!(
            cfg.resolve_model(Some("deepseek-chat")),
            "deepseek-chat"
        );
    }

    #[test]
    fn resolve_model_falls_through_to_default() {
        let cfg = Config {
            api_key: "sk-test".into(),
            base_url: "https://api.deepseek.com".into(),
            model_map: HashMap::new(),
            model_override: None,
            thinking: None,
            timeout: std::time::Duration::from_secs(120),
            backend: super::Backend::DeepSeek,
            listen: "127.0.0.1:8787".parse().unwrap(),
        };
        assert_eq!(cfg.resolve_model(None), "deepseek-v4-pro");
    }
}
