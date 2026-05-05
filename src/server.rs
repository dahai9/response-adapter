use std::convert::Infallible;
use std::sync::Arc;

use async_stream::stream;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use bytes::Bytes;
use futures_util::{Stream, StreamExt};
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE};
use serde_json::{json, Value};
use tokio::sync::Mutex;

use crate::adapter::{
    build_deepseek_body, models_response, response_created, response_failed, response_id,
    ReasoningStore, StreamingAccumulator,
};
use crate::config::Config;

#[derive(Clone)]
pub struct AppState {
    config: Config,
    client: reqwest::Client,
    reasoning_store: Arc<Mutex<ReasoningStore>>,
}

pub fn router(config: Config) -> anyhow::Result<Router> {
    let client = reqwest::Client::builder().timeout(config.timeout).build()?;
    let state = AppState {
        config,
        client,
        reasoning_store: Arc::new(Mutex::new(ReasoningStore::default())),
    };
    Ok(Router::new()
        .route("/health", get(health))
        .route("/healthz", get(health))
        .route("/models", get(models))
        .route("/v1/models", get(models))
        .route("/responses", post(responses))
        .route("/v1/responses", post(responses))
        .with_state(state))
}

pub async fn run(config: Config) -> anyhow::Result<()> {
    let listen = config.listen;
    let app = router(config)?;
    let listener = tokio::net::TcpListener::bind(listen).await?;
    tracing::info!("deepseek-responses-adapter listening on http://{listen}");
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

async fn health() -> Json<Value> {
    Json(json!({"ok": true}))
}

async fn models() -> Json<Value> {
    Json(models_response())
}

async fn responses(State(state): State<AppState>, Json(request): Json<Value>) -> Response {
    let resp_id = response_id();
    let converted = {
        let store = state.reasoning_store.lock().await;
        build_deepseek_body(&request, &state.config, &store)
    };
    if std::env::var("ADAPTER_DEBUG_BODY").ok().as_deref() == Some("1") {
        eprintln!("{}", converted.body);
    }

    let upstream = match open_deepseek_stream(&state, &converted.body).await {
        Ok(upstream) => upstream,
        Err(error) => {
            let events = vec![
                response_created(&resp_id, None),
                response_failed(&resp_id, &error.to_string()),
            ];
            return sse_from_events(events).into_response();
        }
    };

    let store = state.reasoning_store.clone();
    let stream = stream! {
        let mut accumulator = StreamingAccumulator::new(converted.mapper);
        let mut created_sent = false;
        let mut upstream = upstream;
        let mut decoder = SseDecoder::default();

        while let Some(next) = upstream.next().await {
            match next {
                Ok(chunk) => {
                    match decoder.push(&chunk) {
                        Ok(chunks) => {
                            for json_chunk in chunks {
                                let events = accumulator.ingest(&json_chunk);
                                if !created_sent {
                                    created_sent = true;
                                    yield event(response_created(accumulator.resp_id().unwrap_or(&resp_id), accumulator.model()));
                                }
                                for ev in events {
                                    yield event(ev);
                                }
                            }
                        }
                        Err(error) => {
                            yield event(response_failed(accumulator.resp_id().unwrap_or(&resp_id), &error.to_string()));
                            return;
                        }
                    }
                }
                Err(error) => {
                    yield event(response_failed(accumulator.resp_id().unwrap_or(&resp_id), &error.to_string()));
                    return;
                }
            }
        }

        if !created_sent {
            yield event(response_created(&resp_id, None));
        }
        let mut store = store.lock().await;
        for ev in accumulator.final_events(&mut store) {
            yield event(ev);
        }
    };

    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

async fn open_deepseek_stream(
    state: &AppState,
    body: &Value,
) -> Result<impl Stream<Item = reqwest::Result<Bytes>>, AdapterHttpError> {
    let url = format!(
        "{}/chat/completions",
        state.config.base_url.trim_end_matches('/')
    );
    let response = state
        .client
        .post(url)
        .header(AUTHORIZATION, format!("Bearer {}", state.config.api_key))
        .header(CONTENT_TYPE, "application/json")
        .json(body)
        .send()
        .await?;

    let status = response.status();
    if !status.is_success() {
        let text = response.text().await.unwrap_or_default();
        return Err(AdapterHttpError::Upstream(status, text));
    }

    Ok(response.bytes_stream())
}

fn sse_from_events(events: Vec<Value>) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    Sse::new(futures_util::stream::iter(events.into_iter().map(event)))
}

fn event(value: Value) -> Result<Event, Infallible> {
    let kind = value
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("message")
        .to_string();
    Ok(Event::default()
        .event(kind)
        .json_data(value)
        .expect("serializable SSE event"))
}

#[derive(Default)]
struct SseDecoder {
    buffer: String,
}

impl SseDecoder {
    fn push(&mut self, chunk: &[u8]) -> anyhow::Result<Vec<Value>> {
        self.buffer.push_str(&String::from_utf8_lossy(chunk));
        let mut values = Vec::new();

        while let Some(index) = self.buffer.find("\n\n") {
            let block = self.buffer[..index].to_string();
            self.buffer.drain(..index + 2);
            if let Some(value) = parse_sse_block(&block)? {
                values.push(value);
            }
        }

        Ok(values)
    }
}

fn parse_sse_block(block: &str) -> anyhow::Result<Option<Value>> {
    let mut data = String::new();
    for line in block.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with(':') {
            continue;
        }
        let Some(payload) = line.strip_prefix("data:") else {
            continue;
        };
        if !data.is_empty() {
            data.push('\n');
        }
        data.push_str(payload.trim());
    }
    if data.is_empty() || data == "[DONE]" {
        return Ok(None);
    }
    Ok(Some(serde_json::from_str(&data)?))
}

#[derive(Debug, thiserror::Error)]
enum AdapterHttpError {
    #[error(transparent)]
    Request(#[from] reqwest::Error),
    #[error("DeepSeek HTTP {0}: {1}")]
    Upstream(StatusCode, String),
}

#[cfg(test)]
mod tests {
    use super::SseDecoder;

    #[test]
    fn parses_openai_style_sse_chunk() {
        let mut decoder = SseDecoder::default();
        let got = decoder
            .push(
                br#"data: {"id":"1","choices":[{"delta":{"content":"hi"}}]}

data: [DONE]

"#,
            )
            .unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0]["id"], "1");
    }

    #[test]
    fn buffers_split_sse_events() {
        let mut decoder = SseDecoder::default();
        assert!(decoder
            .push(br#"data: {"id":"1","choices":["#)
            .unwrap()
            .is_empty());
        let got = decoder.push(br#"]}"#).unwrap();
        assert!(got.is_empty());
        let got = decoder.push(b"\n\n").unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0]["id"], "1");
    }
}
