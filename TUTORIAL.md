# responses-adapter 详细使用教程

> 本教程面向零基础用户，从安装到配置到使用，逐步讲解每一个环节。

---

## 目录

1. [这个项目是什么？](#1-这个项目是什么)
2. [它能解决什么问题？](#2-它能解决什么问题)
3. [工作原理](#3-工作原理)
4. [环境准备](#4-环境准备)
5. [安装方式](#5-安装方式)
6. [配置详解](#6-配置详解)
7. [启动服务](#7-启动服务)
8. [连接客户端（以 Codex 为例）](#8-连接客户端以-codex-为例)
9. [进阶配置](#9-进阶配置)
10. [Docker 部署](#10-docker-部署)
11. [常见问题排查](#11-常见问题排查)
12. [完整示例](#12-完整示例)

---

## 1. 这个项目是什么？

`responses-adapter` 是一个用 Rust 编写的轻量级**协议转换代理**。它的作用是：

> 把 OpenAI 的 **Responses API** 格式的请求，翻译成标准的 **Chat Completions API** 格式，转发给任意兼容 OpenAI 的上游服务。

简单来说：**让你用只支持 Responses API 的客户端（如 Codex），去对接任何支持 Chat Completions API 的服务（如 OpenAI、DeepSeek、Ollama 等）。**

## 2. 它能解决什么问题？

目前很多 AI 工具（例如 OpenAI Codex CLI）只支持 Responses API 协议。但市面上绝大多数 AI 提供商只提供 Chat Completions API。

没有这个 adapter 的情况：

```
Codex -> Responses API -> ??? (不兼容) -> DeepSeek / 本地模型
```

有了这个 adapter：

```
Codex -> Responses API -> responses-adapter (协议转换) -> Chat Completions API -> DeepSeek / 本地模型 / 任何兼容服务
```

## 3. 工作原理

```
客户端 (Codex)                    responses-adapter                 上游服务
    |                                    |                                |
    |  POST /v1/responses                |                                |
    |  (Responses API 格式)              |                                |
    +----------------------------------->|                                |
    |                                    |  POST /chat/completions        |
    |                                    |  (Chat Completions 格式)       |
    |                                    +------------------------------->|
    |                                    |                                |
    |                                    |  SSE 流式响应                  |
    |                                    |<-------------------------------+
    |                                    |                                |
    |  SSE 流式响应                      |  (翻译为 Responses 事件)       |
    |<-----------------------------------|                                |
```

adapter 做了以下转换：

- Responses `instructions` -> Chat `system` 消息
- Responses `message` 输入 -> Chat `system`/`user`/`assistant` 消息
- Responses `function_call_output` -> Chat `tool` 消息
- Responses `function` 工具定义 -> Chat `function` 工具定义
- 上游返回的 `delta.content` -> Responses `response.output_text.delta` 事件
- 上游返回的 tool calls -> Responses `function_call` 项

---

## 4. 环境准备

> **推荐方式**: 大多数用户应选择 [方式一：下载预编译二进制](#51-方式一下载预编译二进制推荐)，无需安装 Rust 和 Git。

### 4.1 确认操作系统

支持以下操作系统：

- **Linux** (Ubuntu 20.04+, Debian 11+, CentOS 8+, Arch 等)
- **macOS** (12 Monterey+)
- **Windows** (Windows 10+)

### 4.2 准备上游 API 密钥

你需要一个兼容 OpenAI Chat Completions API 的服务。常见选择：

| 服务商 | 获取 API Key 的地址 |
|---|---|
| OpenAI | https://platform.openai.com/api-keys |
| DeepSeek | https://platform.deepseek.com/api_keys |
| 硅基流动 (SiliconFlow) | https://cloud.siliconflow.cn/ |
| 本地 Ollama | 无需 Key，启动即用 |
| 任何 OpenAI 兼容服务 | 对应平台获取 |

> **重要**: 请妥善保管你的 API Key，不要泄露给他人，也不要提交到 Git 仓库。

---

## 5. 安装方式

### 5.1 方式一：下载预编译二进制（推荐）

**大多数人应该使用这种方式**——无需安装 Rust、Git 或任何编译工具，下载即用。

#### 5.1.1 下载

前往 [GitHub Releases](https://github.com/dahai9/response-adapter/releases) 页面，下载对应平台的压缩包：

| 操作系统 | 文件名 |
|---|---|
| Linux (x86_64) | `responses-adapter-vX.Y.Z-x86_64-unknown-linux-gnu.tar.gz` |
| macOS (Intel) | `responses-adapter-vX.Y.Z-x86_64-apple-darwin.tar.gz` |
| macOS (Apple Silicon) | `responses-adapter-vX.Y.Z-aarch64-apple-darwin.tar.gz` |
| Windows (x86_64) | `responses-adapter-vX.Y.Z-x86_64-pc-windows-gnu.zip` |

> **不确定选哪个？** macOS Apple Silicon 用户选 `aarch64-apple-darwin`；Intel Mac 或 Linux 用户选 `x86_64`；Windows 用户选 `x86_64-pc-windows-gnu`。

你也可以用命令行直接下载（以最新版本 v0.1.5 为例）：

```bash
# Linux x86_64
curl -LO https://github.com/dahai9/response-adapter/releases/download/v0.1.5/responses-adapter-v0.1.5-x86_64-unknown-linux-gnu.tar.gz
tar xzf responses-adapter-v0.1.5-x86_64-unknown-linux-gnu.tar.gz

# macOS Apple Silicon (M1/M2/M3/M4)
curl -LO https://github.com/dahai9/response-adapter/releases/download/v0.1.5/responses-adapter-v0.1.5-aarch64-apple-darwin.tar.gz
tar xzf responses-adapter-v0.1.5-aarch64-apple-darwin.tar.gz

# macOS Intel
curl -LO https://github.com/dahai9/response-adapter/releases/download/v0.1.5/responses-adapter-v0.1.5-x86_64-apple-darwin.tar.gz
tar xzf responses-adapter-v0.1.5-x86_64-apple-darwin.tar.gz
```

Windows 用户下载 `.zip` 文件后解压即可。

#### 5.1.2 放置二进制文件

解压后你会得到一个 `responses-adapter` 可执行文件（Windows 为 `responses-adapter.exe`）。建议将其放到一个固定位置：

```bash
# 创建一个目录存放 adapter
mkdir -p ~/responses-adapter
mv responses-adapter ~/responses-adapter/
```

#### 5.1.3 验证安装

```bash
~/responses-adapter/responses-adapter --version
# 预期输出: responses-adapter 0.1.5
```

### 5.2 方式二：从源码编译（开发者）

> 仅在你需要修改代码或平台没有预编译版本时才需要此方式。需要 Rust 1.75+。

<details>
<summary>展开查看源码编译步骤</summary>

#### 安装 Rust 工具链

```bash
# Linux / macOS
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# 验证
rustc --version   # >= 1.75
cargo --version
```

Windows 用户访问 https://rustup.rs/ 下载安装。

#### 克隆并编译

```bash
git clone https://github.com/dahai9/response-adapter.git
cd response-adapter
cargo build --release
```

编译完成后，二进制文件位于 `./target/release/responses-adapter`。

</details>

### 5.3 方式三：使用 Docker

如果你不想安装任何依赖，可以直接使用 Docker。详见 [第 10 节](#10-docker-部署)。

---

## 6. 配置详解

### 6.1 创建配置文件

在项目根目录下，将 `.env.example` 复制为 `.env`：

```bash
cp .env.example .env
```

### 6.2 编辑 `.env` 文件

用任意文本编辑器打开 `.env` 文件。以下逐项说明每个配置项：

#### 必填配置

```bash
# ============================================
# 上游 API 密钥（必填）
# ============================================
# 将下面的值替换为你自己的 API Key
UPSTREAM_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx

# ============================================
# 上游 API 地址（必填）
# ============================================
# 这是上游服务的 Chat Completions 端点的基础 URL
# 注意：不需要包含 /chat/completions 后缀，adapter 会自动添加
```

**不同服务商的 `UPSTREAM_BASE_URL` 设置示例：**

| 服务商 | UPSTREAM_BASE_URL |
|---|---|
| OpenAI | `https://api.openai.com/v1` |
| DeepSeek | `https://api.deepseek.com/v1` |
| 硅基流动 | `https://api.siliconflow.cn/v1` |
| 本地 Ollama | `http://localhost:11434/v1` |
| LiteLLM | `http://localhost:4000/v1` |
| vLLM | `http://localhost:8000/v1` |
| 任意 OpenAI 兼容 | 对应服务的 base URL |

#### 可选配置

```bash
# ============================================
# 模型映射（可选）
# ============================================
# 将客户端请求的模型名映射到上游实际的模型名
# 格式: {"客户端请求的模型名":"上游实际模型名", ...}
#
# 例如：Codex 请求 gpt-5.4，但你想让上游用 deepseek-chat：
# ADAPTER_MODEL_MAP={"gpt-5.4":"deepseek-chat","gpt-5.5":"deepseek-reasoner"}
#
# 如果不设置，则直接使用请求中的模型名

# ============================================
# 固定模型覆盖（可选）
# ============================================
# 如果设置，所有请求都使用这个模型（优先级低于 ADAPTER_MODEL_MAP）
# ADAPTER_MODEL=gpt-4o

# ============================================
# 思维/推理模式（可选）
# ============================================
# 某些模型（如 DeepSeek Reasoner）支持思维链推理
# 设置为 enabled 开启，disabled 关闭，不设置则不干预
# ADAPTER_THINKING=enabled

# ============================================
# 请求超时（可选）
# ============================================
# 上游请求的超时时间，单位秒，默认 120
# 如果你的上游服务响应较慢，可以增大这个值
# ADAPTER_TIMEOUT=120

# ============================================
# 监听地址和端口（可选）
# ============================================
# adapter 服务监听的地址和端口
# 默认: 127.0.0.1:8787
# 如果需要让局域网其他设备访问，改为 0.0.0.0
# ADAPTER_HOST=127.0.0.1
# ADAPTER_PORT=8787

# ============================================
# 模型列表（可选）
# ============================================
# 自定义 /v1/models 端点返回的模型列表
# ADAPTER_MODELS=[{"id":"deepseek-chat","name":"DeepSeek Chat"},{"id":"deepseek-reasoner","name":"DeepSeek Reasoner"}]

# ============================================
# 调试模式（可选）
# ============================================
# 设置为 1 可以在日志中看到转换后的请求体，用于排查问题
# ADAPTER_DEBUG_BODY=1

# ============================================
# 日志级别（可选）
# ============================================
# 可选: trace, debug, info, warn, error
# 默认: info
# RUST_LOG=info
```

### 6.3 完整的 `.env` 配置示例

**示例 1：连接 DeepSeek**

```bash
UPSTREAM_API_KEY=sk-你的deepseek密钥
UPSTREAM_BASE_URL=https://api.deepseek.com/v1
ADAPTER_MODEL_MAP={"gpt-5.4":"deepseek-chat","gpt-5.5":"deepseek-reasoner","gpt-5.3-codex":"deepseek-chat"}
ADAPTER_THINKING=enabled
```

**示例 2：连接本地 Ollama**

```bash
UPSTREAM_API_KEY=ollama
UPSTREAM_BASE_URL=http://localhost:11434/v1
ADAPTER_MODEL=qwen2.5:14b
```

**示例 3：连接 OpenAI 官方**

```bash
UPSTREAM_API_KEY=sk-你的openai密钥
UPSTREAM_BASE_URL=https://api.openai.com/v1
```

**示例 4：连接硅基流动 (SiliconFlow)**

```bash
UPSTREAM_API_KEY=sk-你的siliconflow密钥
UPSTREAM_BASE_URL=https://api.siliconflow.cn/v1
ADAPTER_MODEL_MAP={"gpt-5.4":"Qwen/Qwen2.5-72B-Instruct","gpt-5.5":"deepseek-ai/DeepSeek-V3"}
```

> **提示**: `.env` 文件中的值如果包含特殊字符，可以用双引号包裹，例如: `UPSTREAM_API_KEY="sk-abc=def"`

---

## 7. 启动服务

### 7.1 直接运行（开发模式）

```bash
# 确保在项目根目录下，且 .env 文件已配置好
cargo run
```

你会看到类似输出：

```
   Compiling responses-adapter v0.1.3 (...)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 5.23s
     Running `target/debug/responses-adapter`
responses-adapter listening on http://127.0.0.1:8787
```

### 7.2 运行编译好的 release 版本

```bash
./target/release/responses-adapter
```

输出：

```
responses-adapter listening on http://127.0.0.1:8787
```

### 7.3 后台运行（Linux/macOS）

如果你希望 adapter 在后台持续运行：

```bash
# 使用 nohup
nohup ./target/release/responses-adapter > adapter.log 2>&1 &

# 查看日志
tail -f adapter.log

# 停止服务
kill $(pgrep responses-adapter)
```

或者使用 `systemd` 创建系统服务（适合生产环境）：

```bash
sudo tee /etc/systemd/system/responses-adapter.service << 'EOF'
[Unit]
Description=Responses Adapter
After=network.target

[Service]
Type=simple
User=你的用户名
WorkingDirectory=/path/to/response-adapter
ExecStart=/path/to/response-adapter/target/release/responses-adapter
Restart=always
RestartSec=5
EnvironmentFile=/path/to/response-adapter/.env

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable responses-adapter
sudo systemctl start responses-adapter
sudo systemctl status responses-adapter
```

### 7.4 验证服务是否正常运行

```bash
# 健康检查
curl http://127.0.0.1:8787/health

# 预期输出: {"ok":true}
```

```bash
# 查看模型列表
curl http://127.0.0.1:8787/v1/models
```

---

## 8. 连接客户端（以 Codex 为例）

### 8.1 配置 OpenAI Codex CLI

编辑（或创建）文件 `~/.codex/config.toml`：

```bash
# Linux/macOS
mkdir -p ~/.codex
nano ~/.codex/config.toml
```

写入以下内容：

```toml
model = "gpt-5.4"
model_provider = "responses-adapter"

[model_providers.responses-adapter]
name = "Responses Adapter"
base_url = "http://127.0.0.1:8787/v1"
wire_api = "responses"
env_key = "UPSTREAM_API_KEY"
```

**配置项说明：**

| 字段 | 说明 |
|---|---|
| `model` | 你要使用的模型名。如果设置了 `ADAPTER_MODEL_MAP`，这里填映射前的名称 |
| `model_provider` | 指向你定义的 provider 名称 |
| `name` | provider 的显示名称（随便填） |
| `base_url` | adapter 的地址。必须以 `/v1` 结尾 |
| `wire_api` | 必须是 `"responses"`，表示使用 Responses API 协议 |
| `env_key` | 环境变量名，这里可以随便填一个，实际的 key 已经在 adapter 的 `.env` 中配置 |

### 8.2 测试连接

```bash
# 启动 Codex
codex

# 或者直接用 curl 模拟一个 Responses API 请求来测试
curl -X POST http://127.0.0.1:8787/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5.4",
    "input": [
      {"role": "user", "content": "Hello, say hi in one word."}
    ],
    "stream": true
  }'
```

如果看到 SSE 流式返回的事件数据，说明一切正常。

### 8.3 其他客户端配置

任何支持 Responses API 的客户端都可以使用。配置时只需将 API base URL 指向 `http://127.0.0.1:8787/v1` 即可。

---

## 9. 进阶配置

### 9.1 模型映射详解

模型映射解决了这样一个问题：客户端请求的模型名和上游服务支持的模型名不一致。

例如，Codex 可能只发送 `gpt-5.4` 这个模型名，但你想让它去调用 DeepSeek 的 `deepseek-chat`。

**模型解析优先级**（从高到低）：

1. `ADAPTER_MODEL_MAP` 中的映射匹配
2. `ADAPTER_MODEL` 固定覆盖
3. 请求中原样传递的 `model` 字段

```bash
# 示例：Codex 的不同模型名映射到不同的 DeepSeek 模型
ADAPTER_MODEL_MAP={"gpt-5.4":"deepseek-chat","gpt-5.5":"deepseek-reasoner","gpt-5.2":"deepseek-chat","gpt-5.3-codex":"deepseek-chat"}
```

### 9.2 开启思维/推理模式

对于支持推理的模型（如 DeepSeek Reasoner），开启思维模式：

```bash
ADAPTER_THINKING=enabled
```

开启后，adapter 会：

- 将 Codex 的 `model_reasoning_effort` 转发为上游的 `reasoning_effort`
- 在工具调用场景下，保持 reasoning_content 的上下文连续性

### 9.3 调试模式

当遇到问题时，开启调试日志：

```bash
ADAPTER_DEBUG_BODY=1
RUST_LOG=debug
```

这会在日志中输出转换后的完整请求体，方便排查问题。

### 9.4 局域网共享

如果你想让局域网内的其他设备也能使用这个 adapter：

```bash
ADAPTER_HOST=0.0.0.0
ADAPTER_PORT=8787
```

> **安全警告**: 绑定 `0.0.0.0` 会暴露服务到局域网，请确保你的网络环境安全。

### 9.5 自定义模型列表

配置 `/v1/models` 端点返回的模型列表（某些客户端会查询这个端点）：

```bash
ADAPTER_MODELS='[{"id":"deepseek-chat","name":"DeepSeek Chat"},{"id":"deepseek-reasoner","name":"DeepSeek Reasoner"}]'
```

---

## 10. Docker 部署

### 10.1 构建镜像

```bash
docker build -t responses-adapter .
```

### 10.2 运行容器

**方式 A：使用环境变量**

```bash
docker run -d \
  --name responses-adapter \
  -p 8787:8787 \
  -e UPSTREAM_API_KEY=sk-你的密钥 \
  -e UPSTREAM_BASE_URL=https://api.deepseek.com/v1 \
  -e 'ADAPTER_MODEL_MAP={"gpt-5.4":"deepseek-chat"}' \
  responses-adapter
```

**方式 B：使用 `.env` 文件**

```bash
docker run -d \
  --name responses-adapter \
  -p 8787:8787 \
  --env-file .env \
  responses-adapter
```

### 10.3 常用 Docker 命令

```bash
# 查看运行状态
docker ps | grep responses-adapter

# 查看日志
docker logs -f responses-adapter

# 停止容器
docker stop responses-adapter

# 删除容器
docker rm responses-adapter

# 重新构建并运行
docker stop responses-adapter && docker rm responses-adapter
docker build -t responses-adapter .
docker run -d --name responses-adapter -p 8787:8787 --env-file .env responses-adapter
```

### 10.4 Docker Compose（可选）

创建 `docker-compose.yml`：

```yaml
version: "3.8"
services:
  responses-adapter:
    build: .
    ports:
      - "8787:8787"
    env_file:
      - .env
    restart: unless-stopped
```

启动：

```bash
docker compose up -d
```

---

## 11. 常见问题排查

### Q1: 启动报错 `UPSTREAM_API_KEY is not set`

**原因**: 没有创建 `.env` 文件，或者文件中没有设置 `UPSTREAM_API_KEY`。

**解决**:

```bash
cp .env.example .env
# 编辑 .env，填入你的 API Key
```

### Q2: 启动报错 `UPSTREAM_BASE_URL is not set`

**原因**: `.env` 文件中没有设置 `UPSTREAM_BASE_URL`。

**解决**: 在 `.env` 中添加，例如: `UPSTREAM_BASE_URL=https://api.deepseek.com/v1`

### Q3: 端口被占用 `Address already in use`

**原因**: 8787 端口已被其他进程占用。

**解决**:

```bash
# 查看占用端口的进程
lsof -i :8787        # macOS/Linux
netstat -tlnp | grep 8787  # Linux

# 方法 1: 杀掉占用端口的进程
kill <PID>

# 方法 2: 更换端口
# 在 .env 中修改:
# ADAPTER_PORT=9999
```

### Q4: 连接上游超时

**原因**: 上游服务响应太慢或网络不通。

**解决**:

```bash
# 1. 检查网络连通性
curl -I https://api.deepseek.com

# 2. 增大超时时间
# 在 .env 中修改:
# ADAPTER_TIMEOUT=300

# 3. 如果是本地服务，确认服务已启动
curl http://localhost:11434/api/tags  # Ollama 示例
```

### Q5: 上游返回 401 Unauthorized

**原因**: API Key 不正确或已过期。

**解决**: 检查 `.env` 中的 `UPSTREAM_API_KEY` 是否正确。可以用 curl 直接测试：

```bash
curl https://api.deepseek.com/v1/chat/completions \
  -H "Authorization: Bearer sk-你的key" \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-chat","messages":[{"role":"user","content":"hi"}]}'
```

### Q6: 上游返回 404

**原因**: `UPSTREAM_BASE_URL` 地址不正确。

**解决**: 确认 URL 格式正确。注意：

- 不要包含 `/chat/completions`，adapter 会自动添加
- 确保包含 API 版本路径（通常是 `/v1`）

### Q7: Codex 连接不上 adapter

**检查清单**:

1. adapter 是否已启动？运行 `curl http://127.0.0.1:8787/health`
2. Codex 的 `config.toml` 中 `base_url` 是否正确？必须是 `http://127.0.0.1:8787/v1`
3. `wire_api` 是否设置为 `"responses"`？

### Q8: 编译失败

**可能原因**:

- Rust 版本过低。运行 `rustc --version` 确认 >= 1.75
- 网络问题导致依赖下载失败。可以尝试设置镜像源：

```bash
# 中国用户可以使用字节 rsproxy 镜像加速
export RUSTUP_DIST_SERVER=https://rsproxy.cn
export RUSTUP_UPDATE_ROOT=https://rsproxy.cn/rustup
export CARGO_REGISTRIES_CRATES_IO_PROTOCOL=sparse
export CARGO_NET_GIT_FETCH_WITH_CLI=true
```

### Q9: SSE 流式响应中断

**原因**: 可能是 reqwest 自动解压导致的问题（已在 v0.1.0 修复）。

**解决**: 更新到最新版本。如果自行编译，请确保使用最新代码。

---

## 12. 完整示例

### 场景：用 Codex 连接 DeepSeek

**第一步：安装 Rust 和克隆项目**

```bash
# 安装 Rust（如果还没有）
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# 克隆项目
git clone https://github.com/dahai9/response-adapter.git
cd response-adapter
```

**第二步：配置**

```bash
cp .env.example .env
```

编辑 `.env`：

```bash
UPSTREAM_API_KEY=sk-你的deepseek密钥
UPSTREAM_BASE_URL=https://api.deepseek.com/v1
ADAPTER_MODEL_MAP={"gpt-5.4":"deepseek-chat","gpt-5.5":"deepseek-reasoner","gpt-5.2":"deepseek-chat","gpt-5.3-codex":"deepseek-chat"}
ADAPTER_THINKING=enabled
```

**第三步：编译并启动**

```bash
cargo build --release
./target/release/responses-adapter
```

**第四步：配置 Codex**

```bash
mkdir -p ~/.codex
cat > ~/.codex/config.toml << 'EOF'
model = "gpt-5.4"
model_provider = "responses-adapter"

[model_providers.responses-adapter]
name = "Responses Adapter"
base_url = "http://127.0.0.1:8787/v1"
wire_api = "responses"
env_key = "UPSTREAM_API_KEY"
EOF
```

**第五步：使用**

```bash
codex
# 现在 Codex 的所有请求都会通过 adapter 转发到 DeepSeek
```

---

## 附录：API 端点参考

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/v1/responses` | 将 Responses API 请求转换为 Chat Completions 并转发 |
| `GET` | `/v1/models` | 返回配置的模型列表 |
| `GET` | `/health` | 健康检查端点 |

## 附录：环境变量速查表

| 变量名 | 必填 | 默认值 | 说明 |
|---|---|---|---|
| `UPSTREAM_API_KEY` | **是** | - | 上游 API 密钥 |
| `UPSTREAM_BASE_URL` | **是** | - | 上游 Chat Completions 端点基础 URL |
| `ADAPTER_MODEL` | 否 | - | 固定模型覆盖 |
| `ADAPTER_MODEL_MAP` | 否 | `{}` | JSON 模型映射表 |
| `ADAPTER_THINKING` | 否 | - | `enabled` 或 `disabled` |
| `ADAPTER_TIMEOUT` | 否 | `120` | 请求超时（秒） |
| `ADAPTER_HOST` | 否 | `127.0.0.1` | 监听地址 |
| `ADAPTER_PORT` | 否 | `8787` | 监听端口 |
| `ADAPTER_MODELS` | 否 | `[]` | `/v1/models` 返回的模型列表 |
| `ADAPTER_DEBUG_BODY` | 否 | - | 设为 `1` 开启调试日志 |
| `RUST_LOG` | 否 | `info` | 日志级别 |

---

> 如有疑问，欢迎在 [GitHub Issues](https://github.com/dahai9/response-adapter/issues) 提出。
