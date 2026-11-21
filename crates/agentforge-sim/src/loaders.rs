//! Loaders that turn recorded public agent traces into replayable [`Trace`]s.
//!
//! Two public traces are supported, both read as plain JSONL text. The published files are
//! compressed, so decompress them first (`zstd -d` for the `.jsonl.zst` files, `gunzip` for
//! the `.jsonl.gz` file).
//!
//! - [`TraceFormat::FreeInference`]: the FreeInference agentic trace (Harvard, CC BY 4.0,
//!   Hugging Face dataset `harvardMadsys/freeinference_agentic_trace`), one agent session per
//!   line with its LLM requests. Subagent sessions nested in tool calls are ignored.
//! - [`TraceFormat::TraceLab`]: the TraceLab coding trace (University of Washington SyFI, CC BY
//!   4.0, release asset `syfi_coding_trace.jsonl.gz` of `github.com/uw-syfi/TraceLab`), one LLM
//!   round per line, with the rounds of a session possibly interleaved with other sessions.
//!
//! Both are reduced to the same per-round record (start, end of generation, total input tokens,
//! output tokens, first tool called) and converted to sessions the same way:
//!
//! - The simulator grows a session's context by each step's new and output tokens, so a step's
//!   `new_tokens` is the round's total input minus the context the simulator already holds.
//!   The context then equals the recorded input at every round. Neither trace's own append
//!   count fits: FreeInference has none, and TraceLab's `newly_append_tokens` is the provider's
//!   uncached input, which is the whole prompt after a prompt-cache miss. When the recorded
//!   input is below the simulator's context (output such as reasoning that the next prompt
//!   leaves out), the step appends 1 token and the context runs ahead until the input catches
//!   up.
//! - A round whose input is smaller than the previous round's (a compaction or a rewrite), or
//!   a FreeInference request whose `mutation.transition` is not `append`, ends the session and
//!   starts a new one at that round, because the simulator only grows a context.
//! - Every session starts with no initial prefix, so its first step prefills the whole recorded
//!   input; the simulator has no cache shared between sessions.
//! - The wait after a round runs from the end of its generation to the start of the next round,
//!   which covers tool execution and any human turn. It is named after the round's first tool
//!   call, or `user_turn` when the round called no tool.
//!
//! The recorded wait is replayed as `wait_s`. The latency model a policy sees is fitted per
//! tool name over the whole trace by [`with_fitted_latencies`], so the policy knows each tool's
//! distribution but not the realized wait.

use std::collections::BTreeMap;

use agentforge_ir::LatencyModel;
use serde::Deserialize;
use serde_json::Value;

use crate::report::mean;
use crate::trace::{Session, ToolWait, Trace, TraceStep};

/// Tool name under which [`fit_latency_models`] pools rarely seen tools.
pub const POOLED_TOOL: &str = "other";

/// Tool name of a wait after a round that called no tool.
pub const USER_TURN: &str = "user_turn";

const MIN_WAIT_S: f64 = 1e-3;
const MIN_SIGMA: f64 = 0.05;
const MIN_FIT_SAMPLES: usize = 10;

/// The layout of a trace file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraceFormat {
    /// [`Trace::to_jsonl`] output: one [`Session`] per line.
    Native,
    FreeInference,
    TraceLab,
}

/// What a load read and produced.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LoadSummary {
    /// Sessions in the file, before splitting.
    pub sessions_in: usize,
    /// Sessions in the trace, after splitting at context shrinks.
    pub sessions_out: usize,
    pub steps: u64,
    /// Places where a recorded session was split because its context shrank.
    pub splits: u64,
    /// JSON lines without the fields a session or round needs.
    pub skipped_lines: u64,
}

/// Why a trace file could not be read. Lines that are JSON but lack fields are skipped, not
/// errors.
#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("line {line} is not JSON: {source}")]
    Json {
        line: usize,
        source: serde_json::Error,
    },
    #[error("native trace: {0}")]
    Native(#[from] serde_json::Error),
}

/// Parse a trace. FreeInference and TraceLab traces come back with fitted tool latencies
/// ([`with_fitted_latencies`] with at least 10 waits per tool), ready to replay.
pub fn load_trace(text: &str, format: TraceFormat) -> Result<(Trace, LoadSummary), LoadError> {
    let (groups, skipped_lines) = match format {
        TraceFormat::Native => {
            let trace = Trace::from_jsonl(text)?;
            let summary = LoadSummary {
                sessions_in: trace.sessions.len(),
                sessions_out: trace.sessions.len(),
                steps: trace.total_steps(),
                ..LoadSummary::default()
            };
            return Ok((trace, summary));
        }
        TraceFormat::FreeInference => freeinference_rounds(text)?,
        TraceFormat::TraceLab => tracelab_rounds(text)?,
    };
    let sessions = sessions_from_rounds(&groups);
    let trace = with_fitted_latencies(Trace { sessions }, MIN_FIT_SAMPLES);
    let summary = LoadSummary {
        sessions_in: groups.len(),
        sessions_out: trace.sessions.len(),
        steps: trace.total_steps(),
        splits: (trace.sessions.len() - groups.len()) as u64,
        skipped_lines,
    };
    Ok((trace, summary))
}

/// A log-normal latency per tool name, fitted to the trace's waits: the median wait and the
/// standard deviation of the log wait, with waits floored at 1 ms and sigma at 0.05. Tools
/// with fewer than `min_samples` waits are pooled under [`POOLED_TOOL`].
pub fn fit_latency_models(trace: &Trace, min_samples: usize) -> BTreeMap<String, LatencyModel> {
    let mut waits: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    for tool in tool_waits(trace) {
        waits
            .entry(tool.name.as_str())
            .or_default()
            .push(tool.wait_s.max(MIN_WAIT_S));
    }
    let mut models = BTreeMap::new();
    let mut pooled = Vec::new();
    for (name, samples) in waits {
        if samples.len() >= min_samples && name != POOLED_TOOL {
            models.insert(name.to_string(), fit_lognormal(samples));
        } else {
            pooled.extend(samples);
        }
    }
    if !pooled.is_empty() {
        models.insert(POOLED_TOOL.to_string(), fit_lognormal(pooled));
    }
    models
}

/// Replace every tool wait's latency model with the one [`fit_latency_models`] fits for its
/// tool name, keeping the realized waits.
pub fn with_fitted_latencies(mut trace: Trace, min_samples: usize) -> Trace {
    let models = fit_latency_models(&trace, min_samples);
    for step in trace.sessions.iter_mut().flat_map(|s| s.steps.iter_mut()) {
        if let Some(tool) = step.tool.as_mut() {
            if let Some(model) = models.get(&tool.name).or_else(|| models.get(POOLED_TOOL)) {
                tool.latency = model.clone();
            }
        }
    }
    trace
}

fn tool_waits(trace: &Trace) -> impl Iterator<Item = &ToolWait> {
    trace
        .sessions
        .iter()
        .flat_map(|s| s.steps.iter())
        .filter_map(|step| step.tool.as_ref())
}

fn fit_lognormal(mut waits: Vec<f64>) -> LatencyModel {
    waits.sort_by(f64::total_cmp);
    let n = waits.len();
    let median_s = if n % 2 == 1 {
        waits[n / 2]
    } else {
        0.5 * (waits[n / 2 - 1] + waits[n / 2])
    };
    let logs: Vec<f64> = waits.iter().map(|w| w.ln()).collect();
    let log_mean = mean(&logs);
    let variance = if n > 1 {
        logs.iter().map(|l| (l - log_mean).powi(2)).sum::<f64>() / (n - 1) as f64
    } else {
        0.0
    };
    LatencyModel::LogNormal {
        median_s,
        sigma: variance.sqrt().max(MIN_SIGMA),
    }
}

/// One LLM round of a recorded session, in seconds since the Unix epoch.
#[derive(Debug, Clone, PartialEq)]
struct Round {
    start_s: f64,
    end_s: f64,
    input_tokens: u64,
    output_tokens: u64,
    tool: Option<String>,
    rewrites_context: bool,
}

fn continues_session(prev: &Round, next: &Round) -> bool {
    !next.rewrites_context && next.input_tokens >= prev.input_tokens
}

fn sessions_from_rounds(groups: &[Vec<Round>]) -> Vec<Session> {
    let t0 = groups
        .iter()
        .flatten()
        .map(|r| r.start_s)
        .fold(f64::INFINITY, f64::min);
    let mut sessions: Vec<Session> = groups
        .iter()
        .flat_map(|rounds| rounds.chunk_by(continues_session))
        .map(|run| session_from_run(run, t0))
        .collect();
    sessions.sort_by(|a, b| a.arrival_s.total_cmp(&b.arrival_s));
    for (id, s) in sessions.iter_mut().enumerate() {
        s.id = id as u32;
    }
    sessions
}

fn session_from_run(run: &[Round], t0: f64) -> Session {
    let mut context = 0u64;
    let steps = run
        .iter()
        .enumerate()
        .map(|(i, round)| {
            let new_tokens = round.input_tokens.saturating_sub(context).max(1);
            context += new_tokens + round.output_tokens;
            let tool = run.get(i + 1).map(|next| {
                let wait_s = (next.start_s - round.end_s).max(0.0);
                ToolWait {
                    name: round.tool.clone().unwrap_or_else(|| USER_TURN.to_string()),
                    wait_s,
                    latency: LatencyModel::Fixed { seconds: wait_s },
                }
            });
            TraceStep {
                new_tokens,
                output_tokens: round.output_tokens,
                tool,
            }
        })
        .collect();
    Session {
        id: 0,
        arrival_s: run[0].start_s - t0,
        initial_prefix_tokens: 0,
        steps,
    }
}

/// Parse each non-empty line as JSON, failing on a line that is not JSON, and keep the records
/// `parse` accepts. Returns the records and the number of lines `parse` rejected.
fn parse_lines<T>(
    text: &str,
    mut parse: impl FnMut(Value) -> Option<T>,
) -> Result<(Vec<T>, u64), LoadError> {
    let mut records = Vec::new();
    let mut skipped = 0;
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(line).map_err(|source| LoadError::Json {
            line: i + 1,
            source,
        })?;
        match parse(value) {
            Some(record) => records.push(record),
            None => skipped += 1,
        }
    }
    Ok((records, skipped))
}

#[derive(Deserialize)]
struct FiSession {
    requests: Vec<FiRequest>,
}

#[derive(Deserialize)]
struct FiRequest {
    step: u64,
    start_ms: f64,
    end_ms: f64,
    provider_tokens: FiTokens,
    #[serde(default)]
    mutation: Option<FiMutation>,
    #[serde(default)]
    tool_calls: Vec<FiToolCall>,
}

#[derive(Deserialize)]
struct FiTokens {
    prompt: u64,
    completion: u64,
}

#[derive(Deserialize)]
struct FiMutation {
    #[serde(default)]
    transition: Option<String>,
}

#[derive(Deserialize)]
struct FiToolCall {
    name: String,
}

fn freeinference_rounds(text: &str) -> Result<(Vec<Vec<Round>>, u64), LoadError> {
    parse_lines(text, |value| {
        let mut requests = serde_json::from_value::<FiSession>(value).ok()?.requests;
        if requests.is_empty() {
            return None;
        }
        requests.sort_by_key(|r| r.step);
        Some(
            requests
                .into_iter()
                .map(|r| Round {
                    start_s: r.start_ms / 1000.0,
                    end_s: r.end_ms / 1000.0,
                    input_tokens: r.provider_tokens.prompt,
                    output_tokens: r.provider_tokens.completion,
                    tool: r.tool_calls.into_iter().next().map(|c| c.name),
                    rewrites_context: r
                        .mutation
                        .and_then(|m| m.transition)
                        .is_some_and(|t| t != "append"),
                })
                .collect(),
        )
    })
}

#[derive(Deserialize)]
struct TlRound {
    session_id: String,
    round_index: u64,
    input_tokens_total: u64,
    output_tokens: u64,
    timing_events: Vec<TlEvent>,
    #[serde(default)]
    tools: Vec<TlTool>,
}

#[derive(Deserialize)]
struct TlEvent {
    event_type: String,
    timestamp: String,
}

#[derive(Deserialize)]
struct TlTool {
    tool_name: String,
    #[serde(default)]
    emitted_at: Option<String>,
}

/// TraceLab events the model produces; a round's generation ends at the last of them.
const GENERATION_EVENTS: [&str; 4] = ["reasoning", "text", "tool_call", "usage_report"];

fn tracelab_rounds(text: &str) -> Result<(Vec<Vec<Round>>, u64), LoadError> {
    let (rows, skipped) = parse_lines(text, |value| {
        let row = serde_json::from_value::<TlRound>(value).ok()?;
        let mut start_s = f64::INFINITY;
        let mut end_s = f64::NEG_INFINITY;
        for event in &row.timing_events {
            let t = parse_utc_timestamp(&event.timestamp)?;
            start_s = start_s.min(t);
            if GENERATION_EVENTS.contains(&event.event_type.as_str()) {
                end_s = end_s.max(t);
            }
        }
        for emitted in row.tools.iter().filter_map(|t| t.emitted_at.as_deref()) {
            end_s = end_s.max(parse_utc_timestamp(emitted)?);
        }
        if !start_s.is_finite() {
            return None;
        }
        let round = Round {
            start_s,
            end_s: end_s.max(start_s),
            input_tokens: row.input_tokens_total,
            output_tokens: row.output_tokens,
            tool: row.tools.into_iter().next().map(|t| t.tool_name),
            rewrites_context: false,
        };
        Some((row.session_id, row.round_index, round))
    })?;
    let mut sessions: BTreeMap<String, Vec<(u64, Round)>> = BTreeMap::new();
    for (session_id, round_index, round) in rows {
        sessions
            .entry(session_id)
            .or_default()
            .push((round_index, round));
    }
    let groups = sessions
        .into_values()
        .map(|mut rounds| {
            rounds.sort_by_key(|(index, _)| *index);
            rounds.into_iter().map(|(_, round)| round).collect()
        })
        .collect();
    Ok((groups, skipped))
}

/// Seconds since the Unix epoch of a UTC timestamp `YYYY-MM-DDTHH:MM:SS[.fff]Z`.
fn parse_utc_timestamp(s: &str) -> Option<f64> {
    let s = s.strip_suffix('Z')?;
    let (date, time) = s.split_once('T')?;
    let mut date_parts = date.splitn(3, '-');
    let year: i64 = date_parts.next()?.parse().ok()?;
    let month: i64 = date_parts.next()?.parse().ok()?;
    let day: i64 = date_parts.next()?.parse().ok()?;
    let (hms, fraction) = match time.split_once('.') {
        Some((hms, digits)) => (hms, format!("0.{digits}").parse::<f64>().ok()?),
        None => (time, 0.0),
    };
    let mut time_parts = hms.splitn(3, ':');
    let hour: i64 = time_parts.next()?.parse().ok()?;
    let minute: i64 = time_parts.next()?.parse().ok()?;
    let second: i64 = time_parts.next()?.parse().ok()?;
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    let seconds = days_from_civil(year, month, day) * 86_400 + hour * 3_600 + minute * 60 + second;
    Some(seconds as f64 + fraction)
}

/// Days from 1970-01-01 to a proleptic Gregorian date (Howard Hinnant's `days_from_civil`).
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let year = if month <= 2 { year - 1 } else { year };
    let era = year.div_euclid(400);
    let year_of_era = year - era * 400;
    let day_of_year = (153 * ((month + 9) % 12) + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{run, Lru, Plan, Policy, SimConfig};
    use serde_json::json;

    const T0_MS: u64 = 1_780_292_400_000;

    fn jsonl(rows: &[Value]) -> String {
        rows.iter().map(|r| format!("{r}\n")).collect()
    }

    fn fi_request(
        step: u64,
        start_s: f64,
        end_s: f64,
        prompt: u64,
        completion: u64,
        transition: Option<&str>,
        tool: &str,
    ) -> Value {
        json!({
            "step": step,
            "turn_index": 0,
            "step_trigger": if step == 0 { "user" } else { "tool" },
            "start_ms": T0_MS + (start_s * 1000.0) as u64,
            "end_ms": T0_MS + (end_s * 1000.0) as u64,
            "ttft_ms": 400,
            "model": "m",
            "provider": "p",
            "provider_tokens": { "prompt": prompt, "completion": completion, "cached": 0 },
            "role_tokens": {},
            "n_messages": 2 * step + 1,
            "finish_reason": "tool_calls",
            "mutation": transition.map(|t| json!({ "transition": t, "cause": null })),
            "block_ids": [1, 2, 3],
            "tool_calls": [{
                "call_index": 0,
                "name": tool,
                "category": "shell",
                "latency_ms": 1000,
                "result_tokens": 100,
                "outcome": "ok",
                "subagent": null
            }]
        })
    }

    fn fi_session(id: &str, requests: Vec<Value>) -> Value {
        json!({
            "session_id": id,
            "user_id": "u",
            "harness": "claude-code",
            "start_ms": T0_MS,
            "end_ms": T0_MS + 100_000,
            "requests": requests
        })
    }

    fn freeinference_fixture() -> String {
        jsonl(&[
            fi_session(
                "compacted",
                vec![
                    fi_request(1, 2.5, 4.0, 20_800, 200, Some("append"), "Edit"),
                    fi_request(0, 0.0, 2.0, 20_000, 300, None, "Read"),
                    fi_request(2, 10.0, 11.0, 8_000, 150, Some("compact"), "Bash"),
                    fi_request(3, 12.0, 13.0, 8_400, 100, Some("append"), "Edit"),
                ],
            ),
            fi_session(
                "steady",
                vec![
                    fi_request(0, 5.0, 6.0, 15_000, 400, None, "Bash"),
                    fi_request(1, 9.0, 10.5, 16_000, 250, Some("append"), "Bash"),
                    fi_request(2, 70.5, 71.5, 16_900, 120, Some("append"), "Read"),
                ],
            ),
        ])
    }

    fn step_tokens(s: &Session) -> Vec<(u64, u64)> {
        s.steps
            .iter()
            .map(|st| (st.new_tokens, st.output_tokens))
            .collect()
    }

    fn step_waits(s: &Session) -> Vec<Option<(String, f64)>> {
        s.steps
            .iter()
            .map(|st| {
                st.tool
                    .as_ref()
                    .map(|t| (t.name.clone(), (t.wait_s * 1000.0).round() / 1000.0))
            })
            .collect()
    }

    fn wait(name: &str, seconds: f64) -> Option<(String, f64)> {
        Some((name.to_string(), seconds))
    }

    #[test]
    fn freeinference_sessions_split_at_compaction() {
        let (trace, summary) =
            load_trace(&freeinference_fixture(), TraceFormat::FreeInference).unwrap();
        assert_eq!(
            summary,
            LoadSummary {
                sessions_in: 2,
                sessions_out: 3,
                steps: 7,
                splits: 1,
                skipped_lines: 0,
            }
        );
        let arrivals: Vec<f64> = trace.sessions.iter().map(|s| s.arrival_s).collect();
        assert_eq!(arrivals, vec![0.0, 5.0, 10.0]);
        let ids: Vec<u32> = trace.sessions.iter().map(|s| s.id).collect();
        assert_eq!(ids, vec![0, 1, 2]);
        assert!(trace.sessions.iter().all(|s| s.initial_prefix_tokens == 0));

        let [before, steady, after] = &trace.sessions[..] else {
            panic!("three sessions");
        };
        assert_eq!(step_tokens(before), vec![(20_000, 300), (500, 200)]);
        assert_eq!(step_waits(before), vec![wait("Read", 0.5), None]);
        assert_eq!(
            step_tokens(steady),
            vec![(15_000, 400), (600, 250), (650, 120)]
        );
        assert_eq!(
            step_waits(steady),
            vec![wait("Bash", 3.0), wait("Bash", 60.0), None]
        );
        assert_eq!(step_tokens(after), vec![(8_000, 150), (250, 100)]);
        assert_eq!(step_waits(after), vec![wait("Bash", 1.0), None]);
    }

    #[test]
    fn loaded_traces_carry_fitted_latencies_and_realized_waits() {
        let (trace, _) = load_trace(&freeinference_fixture(), TraceFormat::FreeInference).unwrap();
        let waits: Vec<f64> = tool_waits(&trace).map(|t| t.wait_s).collect();
        assert_eq!(waits.len(), 4);
        for tool in tool_waits(&trace) {
            let LatencyModel::LogNormal { median_s, sigma } = tool.latency else {
                panic!("{:?}", tool.latency);
            };
            assert!((median_s - 2.0).abs() < 1e-9, "{median_s}");
            assert!((sigma - 2.112_580_381_242_256).abs() < 1e-9, "{sigma}");
        }
    }

    #[test]
    fn fitted_latencies_pool_tools_below_min_samples() {
        let (trace, _) = load_trace(&freeinference_fixture(), TraceFormat::FreeInference).unwrap();
        let models = fit_latency_models(&trace, 3);
        assert_eq!(models.keys().collect::<Vec<_>>(), vec!["Bash", "other"]);
        let LatencyModel::LogNormal { median_s, sigma } = models["Bash"] else {
            panic!("{:?}", models["Bash"]);
        };
        assert!((median_s - 3.0).abs() < 1e-9, "{median_s}");
        assert!((sigma - 2.119_159_354_155_291_6).abs() < 1e-9, "{sigma}");
        assert_eq!(
            models["other"],
            LatencyModel::LogNormal {
                median_s: 0.5,
                sigma: 0.05
            }
        );

        let fitted = with_fitted_latencies(trace, 3);
        for tool in tool_waits(&fitted) {
            let expected = if tool.name == "Bash" {
                &models["Bash"]
            } else {
                &models["other"]
            };
            assert_eq!(&tool.latency, expected, "{}", tool.name);
        }
    }

    fn tl_round(
        session: &str,
        round: u64,
        input: u64,
        newly_append: u64,
        output: u64,
        events: &[(&str, &str)],
        tool: Option<(&str, &str, &str)>,
    ) -> Value {
        json!({
            "provider": "claude",
            "session_id": session,
            "round_index": round,
            "round_id": format!("{session}-{round}"),
            "model": "m",
            "input_tokens_total": input,
            "prefix_tokens": input - newly_append,
            "newly_append_tokens": newly_append,
            "output_tokens": output,
            "reasoning_output_tokens": null,
            "timing_events": events
                .iter()
                .map(|(kind, at)| json!({
                    "event_type": kind,
                    "timestamp": format!("2026-06-01T{at}Z"),
                    "source": "trace"
                }))
                .collect::<Vec<_>>(),
            "tools": tool
                .map(|(name, emitted, result)| vec![json!({
                    "tool_index": 0,
                    "tool_name": name,
                    "tool_call_id": "c",
                    "emitted_at": format!("2026-06-01T{emitted}Z"),
                    "result_at": format!("2026-06-01T{result}Z"),
                    "tool_wall_latency_ms": 0,
                    "tool_internal_latency_ms": null,
                    "is_error": false,
                    "input_chars": 10,
                    "result_chars": 10
                })])
                .unwrap_or_default(),
            "store": "s",
            "trace_key": "k",
            "first_input_event_type": events[0].0
        })
    }

    fn tracelab_fixture() -> String {
        jsonl(&[
            tl_round(
                "s1",
                1,
                121_500,
                1_500,
                200,
                &[("tool_result", "05:40:16.500"), ("text", "05:40:18.000")],
                None,
            ),
            tl_round(
                "s2",
                0,
                50_000,
                50_000,
                100,
                &[
                    ("user_message", "05:40:10.000"),
                    ("tool_call", "05:40:11.000"),
                ],
                Some(("Read", "05:40:11.000", "05:40:11.250")),
            ),
            tl_round(
                "s1",
                0,
                120_000,
                120_000,
                300,
                &[
                    ("user_message", "05:40:00.000"),
                    ("reasoning", "05:40:02.000"),
                    ("tool_call", "05:40:04.500"),
                ],
                Some(("Bash", "05:40:04.500", "05:40:16.500")),
            ),
            tl_round(
                "s2",
                1,
                20_000,
                20_000,
                400,
                &[("tool_result", "05:40:11.250"), ("text", "05:40:13.000")],
                None,
            ),
            tl_round(
                "s1",
                2,
                122_000,
                122_000,
                50,
                &[("user_message", "05:41:48.000"), ("text", "05:41:50.000")],
                None,
            ),
        ])
    }

    #[test]
    fn tracelab_rounds_group_by_session_and_split_at_compaction() {
        let (trace, summary) = load_trace(&tracelab_fixture(), TraceFormat::TraceLab).unwrap();
        assert_eq!(
            summary,
            LoadSummary {
                sessions_in: 2,
                sessions_out: 3,
                steps: 5,
                splits: 1,
                skipped_lines: 0,
            }
        );
        let arrivals: Vec<f64> = trace
            .sessions
            .iter()
            .map(|s| (s.arrival_s * 1000.0).round() / 1000.0)
            .collect();
        assert_eq!(arrivals, vec![0.0, 10.0, 11.25]);

        let [s1, s2_before, s2_after] = &trace.sessions[..] else {
            panic!("three sessions");
        };
        assert_eq!(
            step_tokens(s1),
            vec![(120_000, 300), (1_200, 200), (300, 50)]
        );
        assert_eq!(
            step_waits(s1),
            vec![wait("Bash", 12.0), wait("user_turn", 90.0), None]
        );
        assert_eq!(step_tokens(s2_before), vec![(50_000, 100)]);
        assert_eq!(step_waits(s2_before), vec![None]);
        assert_eq!(step_tokens(s2_after), vec![(20_000, 400)]);
    }

    #[test]
    fn utc_timestamps_parse_to_unix_seconds() {
        assert_eq!(parse_utc_timestamp("1970-01-01T00:00:01.500Z"), Some(1.5));
        let t = parse_utc_timestamp("2026-06-01T05:40:20.151Z").unwrap();
        assert!((t - 1_780_292_420.151).abs() < 1e-6, "{t}");
        assert_eq!(
            parse_utc_timestamp("2026-06-01T05:40:20Z"),
            Some(1_780_292_420.0)
        );
        assert_eq!(parse_utc_timestamp("2026-06-01 05:40:20Z"), None);
    }

    #[test]
    fn lines_without_session_fields_are_skipped_and_non_json_fails() {
        let text = format!("{}{{\"session_id\": \"x\"}}\n\n", freeinference_fixture());
        let (_, summary) = load_trace(&text, TraceFormat::FreeInference).unwrap();
        assert_eq!(summary.skipped_lines, 1);
        assert_eq!(summary.sessions_in, 2);

        let broken = format!("{}not json\n", tracelab_fixture());
        match load_trace(&broken, TraceFormat::TraceLab) {
            Err(LoadError::Json { line, .. }) => assert_eq!(line, 6),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn native_traces_load_unchanged() {
        let (trace, _) = load_trace(&freeinference_fixture(), TraceFormat::FreeInference).unwrap();
        let (back, summary) = load_trace(&trace.to_jsonl().unwrap(), TraceFormat::Native).unwrap();
        assert_eq!(back, trace);
        assert_eq!(summary.sessions_out, 3);
        assert_eq!(summary.steps, 7);
    }

    #[test]
    fn loaded_traces_replay_to_completion() {
        let machine = agentforge_machine::Machine::builtin("h200").unwrap();
        let model = agentforge_ir::ModelSpec::preset("llama-3.1-70b").unwrap();
        let cfg = SimConfig::new(model, machine.devices[0].clone(), machine.tiers.clone(), 8);
        for (text, format) in [
            (freeinference_fixture(), TraceFormat::FreeInference),
            (tracelab_fixture(), TraceFormat::TraceLab),
        ] {
            let (trace, _) = load_trace(&text, format).unwrap();
            let new_tokens: u64 = trace
                .sessions
                .iter()
                .flat_map(|s| &s.steps)
                .map(|st| st.new_tokens)
                .sum();
            for policy in [&Lru as &dyn Policy, &Plan] {
                let report = run(&cfg, &trace, policy).unwrap();
                assert_eq!(report.sessions_done as usize, trace.sessions.len());
                assert_eq!(report.steps, trace.total_steps());
                assert!(report.prefill_tokens >= new_tokens);
            }
            assert_eq!(run(&cfg, &trace, &Lru).unwrap().prefill_tokens, new_tokens);
        }
    }
}
