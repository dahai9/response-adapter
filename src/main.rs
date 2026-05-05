use anyhow::Result;
use deepseek_responses_adapter::config::Config;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let config = Config::from_env()?;
    deepseek_responses_adapter::server::run(config).await
}
