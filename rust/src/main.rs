use std::env;
use std::io::{self, Read, Write};
use std::thread;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

mod behavior_model;
mod fsrs_adr;
mod fsrs_v6;
mod search;

use behavior_model::BehaviorModel;
use fsrs_adr::FSRSADR;
use fsrs_v6::FSRSv6;
use search::{dr_summary_by_weight, safety_summary, simulate};

const CMD_CONFIGURE: u8 = 1;
const CMD_EVALUATE_MANY: u8 = 2;
const CMD_SAFETY_MANY: u8 = 3;
const CMD_DR_SUMMARY_MANY: u8 = 4;
const CMD_CLOSE: u8 = 255;

const STATUS_OK: u8 = 0;
const STATUS_ERR: u8 = 1;

#[derive(Clone, Copy)]
struct EvalState {
    fsrs: FSRSv6,
    behavior_model: BehaviorModel,
    days: i32,
    deck_size: i32,
    new_cards_per_day: i32,
    threads: usize,
}

#[derive(Clone, Copy)]
struct Candidate {
    flat: f32,
    s_multi: f32,
    d_multi: f32,
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.iter().any(|arg| arg == "--version" || arg == "version") {
        println!("adr-simulator-helper {}", env!("CARGO_PKG_VERSION"));
        return;
    }
    if args.iter().any(|arg| arg == "--help" || arg == "-h" || arg == "help") {
        print_help();
        return;
    }
    if let Err(err) = serve_binary_worker() {
        let _ = writeln!(io::stderr(), "{err}");
        std::process::exit(1);
    }
}

fn print_help() {
    println!(
        "adr-simulator-helper {}\n\n\
         Binary worker for linear ADR optimization.\n\
         It reads length-prefixed binary commands from stdin and writes \
         length-prefixed binary responses to stdout. This rewrite intentionally \
         has no JSON request/response mode.",
        env!("CARGO_PKG_VERSION")
    );
}

fn serve_binary_worker() -> Result<(), String> {
    let mut stdin = io::stdin().lock();
    let mut stdout = io::stdout().lock();
    let mut state: Option<EvalState> = None;

    loop {
        let frame = match read_frame(&mut stdin)? {
            Some(frame) => frame,
            None => return Ok(()),
        };
        let mut reader = BinReader::new(&frame);
        let cmd = match reader.u8() {
            Ok(cmd) => cmd,
            Err(err) => {
                write_error(&mut stdout, err)?;
                continue;
            }
        };

        if cmd == CMD_CLOSE {
            write_ok(&mut stdout, &[])?;
            return Ok(());
        }

        let result = match cmd {
            CMD_CONFIGURE => handle_configure(&mut reader, &mut state),
            CMD_EVALUATE_MANY => with_state(state, |st| handle_evaluate_many(&mut reader, st)),
            CMD_SAFETY_MANY => with_state(state, |st| handle_safety_many(&mut reader, st)),
            CMD_DR_SUMMARY_MANY => with_state(state, |st| handle_dr_summary_many(&mut reader, st)),
            _ => Err(format!("unknown command byte {cmd}")),
        };

        match result {
            Ok(payload) => write_ok(&mut stdout, &payload)?,
            Err(err) => write_error(&mut stdout, err)?,
        }
    }
}

fn with_state<F>(state: Option<EvalState>, f: F) -> Result<Vec<u8>, String>
where
    F: FnOnce(EvalState) -> Result<Vec<u8>, String>,
{
    match state {
        Some(state) => f(state),
        None => Err("worker is not configured".to_string()),
    }
}

fn handle_configure(
    reader: &mut BinReader<'_>,
    state: &mut Option<EvalState>,
) -> Result<Vec<u8>, String> {
    let weights = reader.array_f32::<21>()?;
    let days = reader.i32()?;
    let deck_size = reader.i32()?;
    let new_cards_per_day = reader.i32()?;
    let initial_rating_prob = normalize_array(reader.array_f32::<4>()?, "initial_rating_prob")?;
    let initial_cost = reader.array_f32::<4>()?;
    let review_rating_prob_given_success =
        normalize_array(reader.array_f32::<3>()?, "review_rating_prob_given_success")?;
    let review_cost = reader.array_f32::<4>()?;
    let requested_threads = reader.u32()? as usize;
    reader.finish()?;

    let threads = configured_threads(requested_threads);
    *state = Some(EvalState {
        fsrs: FSRSv6::new(weights),
        behavior_model: BehaviorModel::new(
            initial_rating_prob,
            initial_cost,
            review_rating_prob_given_success,
            review_cost,
        ),
        days,
        deck_size,
        new_cards_per_day,
        threads,
    });

    let mut out = Vec::new();
    push_u32(&mut out, threads as u32);
    Ok(out)
}

fn handle_evaluate_many(reader: &mut BinReader<'_>, state: EvalState) -> Result<Vec<u8>, String> {
    let weight = reader.f32()?;
    let seed = reader.u64()?;
    let candidates = reader.candidates()?;
    reader.finish()?;

    let rows = parallel_collect(candidates, state.threads, |idx, candidate| {
        let adr = FSRSADR::linear(candidate.flat, candidate.s_multi, candidate.d_multi);
        let mut rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(idx as u64 * 1_000_003));
        let result = simulate(
            weight,
            state.deck_size,
            state.new_cards_per_day,
            state.days as f32,
            &state.fsrs,
            &adr,
            &state.behavior_model,
            &mut rng,
        );
        let memorized_fraction = result.memorized();
        let memorized_cards = memorized_fraction * state.deck_size as f64;
        let memorized_per_minute = 60.0 * result.efficiency();
        (
            candidate,
            result.total_average_memorized,
            result.total_cost,
            result.total_iters,
            memorized_fraction,
            memorized_cards,
            memorized_per_minute,
        )
    })?;

    let mut out = Vec::new();
    push_u32(&mut out, rows.len() as u32);
    for (
        candidate,
        total_average_memorized,
        total_cost,
        total_iters,
        memorized_fraction,
        memorized_cards,
        memorized_per_minute,
    ) in rows
    {
        push_f32(&mut out, candidate.flat);
        push_f32(&mut out, candidate.s_multi);
        push_f32(&mut out, candidate.d_multi);
        push_f64(&mut out, total_average_memorized);
        push_f64(&mut out, total_cost);
        push_i32(&mut out, total_iters);
        push_f64(&mut out, memorized_fraction);
        push_f64(&mut out, memorized_cards);
        push_f64(&mut out, memorized_per_minute);
    }
    Ok(out)
}

fn handle_safety_many(reader: &mut BinReader<'_>, state: EvalState) -> Result<Vec<u8>, String> {
    let s_max = reader.f32()?;
    let max_checks = reader.i32()?;
    let candidates = reader.candidates()?;
    reader.finish()?;

    let rows = parallel_collect(candidates, state.threads, |_idx, candidate| {
        let adr = FSRSADR::linear(candidate.flat, candidate.s_multi, candidate.d_multi);
        let summary = safety_summary(
            &state.fsrs,
            &adr,
            &state.behavior_model,
            state.days as f32,
            s_max,
            max_checks,
        );
        (candidate, summary)
    })?;

    let mut out = Vec::new();
    push_u32(&mut out, rows.len() as u32);
    for (candidate, summary) in rows {
        push_f32(&mut out, candidate.flat);
        push_f32(&mut out, candidate.s_multi);
        push_f32(&mut out, candidate.d_multi);
        push_i32(&mut out, summary.checks);
        push_i32(&mut out, summary.interval_flips);
        push_i32(&mut out, summary.hard_shortens);
        push_f32(&mut out, summary.dr_p10);
        push_f32(&mut out, summary.dr_mean);
        push_f32(&mut out, summary.dr_p90);
        push_f32(&mut out, summary.aggression);
    }
    Ok(out)
}

fn handle_dr_summary_many(
    reader: &mut BinReader<'_>,
    state: EvalState,
) -> Result<Vec<u8>, String> {
    let start_weight = reader.f32()?;
    let prune_weight = reader.f32()?;
    let candidates = reader.candidates()?;
    reader.finish()?;

    let rows = parallel_collect(candidates, state.threads, |_idx, candidate| {
        let adr = FSRSADR::linear(candidate.flat, candidate.s_multi, candidate.d_multi);
        let summary = dr_summary_by_weight(
            &state.fsrs,
            &adr,
            &state.behavior_model,
            state.days as f32,
            start_weight,
            prune_weight,
        );
        (candidate, summary)
    })?;

    let mut out = Vec::new();
    push_u32(&mut out, rows.len() as u32);
    for (candidate, summary) in rows {
        push_f32(&mut out, candidate.flat);
        push_f32(&mut out, candidate.s_multi);
        push_f32(&mut out, candidate.d_multi);
        push_i64(&mut out, summary.samples);
        push_f32(&mut out, summary.dr_p10);
        push_f32(&mut out, summary.dr_mean);
        push_f32(&mut out, summary.dr_p90);
        push_f32(&mut out, summary.aggression);
    }
    Ok(out)
}

fn parallel_collect<T, U, F>(items: Vec<T>, threads: usize, func: F) -> Result<Vec<U>, String>
where
    T: Copy + Send + Sync,
    U: Send,
    F: Fn(usize, T) -> U + Sync,
{
    let len = items.len();
    if len == 0 {
        return Ok(Vec::new());
    }
    let workers = threads.clamp(1, len);
    if workers == 1 {
        return Ok(items
            .into_iter()
            .enumerate()
            .map(|(idx, item)| func(idx, item))
            .collect());
    }

    let chunk_size = (len + workers - 1) / workers;
    let mut indexed = Vec::with_capacity(len);
    thread::scope(|scope| {
        let mut handles = Vec::new();
        for (chunk_idx, chunk) in items.chunks(chunk_size).enumerate() {
            let start = chunk_idx * chunk_size;
            let func = &func;
            handles.push(scope.spawn(move || {
                chunk
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(idx, item)| (start + idx, func(start + idx, item)))
                    .collect::<Vec<_>>()
            }));
        }
        for handle in handles {
            match handle.join() {
                Ok(mut values) => indexed.append(&mut values),
                Err(_) => return Err("worker thread panicked".to_string()),
            }
        }
        Ok(())
    })?;

    indexed.sort_by_key(|(idx, _)| *idx);
    Ok(indexed.into_iter().map(|(_, value)| value).collect())
}

fn configured_threads(requested: usize) -> usize {
    if requested > 0 {
        return requested;
    }
    if let Ok(raw) = env::var("ADR_SIMULATOR_THREADS") {
        if let Ok(value) = raw.parse::<usize>() {
            if value > 0 {
                return value;
            }
        }
    }
    thread::available_parallelism().map_or(1, usize::from)
}

fn normalize_array<const N: usize>(mut values: [f32; N], name: &str) -> Result<[f32; N], String> {
    let total: f32 = values.iter().sum();
    if total <= 0.0 {
        return Err(format!("{name} must have a positive sum"));
    }
    for value in &mut values {
        *value /= total;
    }
    Ok(values)
}

fn read_frame<R: Read>(reader: &mut R) -> Result<Option<Vec<u8>>, String> {
    let mut len_bytes = [0_u8; 4];
    let mut filled = 0;
    while filled < len_bytes.len() {
        match reader.read(&mut len_bytes[filled..]) {
            Ok(0) if filled == 0 => return Ok(None),
            Ok(0) => return Err("unexpected EOF while reading frame length".to_string()),
            Ok(read) => filled += read,
            Err(err) => return Err(format!("failed to read frame length: {err}")),
        }
    }

    let len = u32::from_le_bytes(len_bytes) as usize;
    let mut payload = vec![0_u8; len];
    reader
        .read_exact(&mut payload)
        .map_err(|err| format!("failed to read frame payload: {err}"))?;
    Ok(Some(payload))
}

fn write_ok<W: Write>(writer: &mut W, payload: &[u8]) -> Result<(), String> {
    let mut frame = Vec::with_capacity(payload.len() + 1);
    frame.push(STATUS_OK);
    frame.extend_from_slice(payload);
    write_frame(writer, &frame)
}

fn write_error<W: Write>(writer: &mut W, error: String) -> Result<(), String> {
    let bytes = error.as_bytes();
    let mut frame = Vec::with_capacity(bytes.len() + 5);
    frame.push(STATUS_ERR);
    push_u32(&mut frame, bytes.len() as u32);
    frame.extend_from_slice(bytes);
    write_frame(writer, &frame)
}

fn write_frame<W: Write>(writer: &mut W, payload: &[u8]) -> Result<(), String> {
    writer
        .write_all(&(payload.len() as u32).to_le_bytes())
        .and_then(|_| writer.write_all(payload))
        .and_then(|_| writer.flush())
        .map_err(|err| format!("failed to write frame: {err}"))
}

struct BinReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> BinReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn finish(&self) -> Result<(), String> {
        if self.pos == self.bytes.len() {
            Ok(())
        } else {
            Err(format!(
                "command had {} trailing bytes",
                self.bytes.len() - self.pos
            ))
        }
    }

    fn take<const N: usize>(&mut self) -> Result<[u8; N], String> {
        if self.pos + N > self.bytes.len() {
            return Err("command payload ended early".to_string());
        }
        let mut out = [0_u8; N];
        out.copy_from_slice(&self.bytes[self.pos..self.pos + N]);
        self.pos += N;
        Ok(out)
    }

    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take::<1>()?[0])
    }

    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take()?))
    }

    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take()?))
    }

    fn i32(&mut self) -> Result<i32, String> {
        Ok(i32::from_le_bytes(self.take()?))
    }

    fn f32(&mut self) -> Result<f32, String> {
        Ok(f32::from_le_bytes(self.take()?))
    }

    fn array_f32<const N: usize>(&mut self) -> Result<[f32; N], String> {
        let mut out = [0.0_f32; N];
        for item in &mut out {
            *item = self.f32()?;
        }
        Ok(out)
    }

    fn candidates(&mut self) -> Result<Vec<Candidate>, String> {
        let len = self.u32()? as usize;
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            out.push(Candidate {
                flat: self.f32()?,
                s_multi: self.f32()?,
                d_multi: self.f32()?,
            });
        }
        Ok(out)
    }
}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_i32(out: &mut Vec<u8>, value: i32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_i64(out: &mut Vec<u8>, value: i64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_f32(out: &mut Vec<u8>, value: f32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_f64(out: &mut Vec<u8>, value: f64) {
    out.extend_from_slice(&value.to_le_bytes());
}
