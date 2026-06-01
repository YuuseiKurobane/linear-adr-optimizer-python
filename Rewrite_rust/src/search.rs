use rand::Rng;

use crate::{
    behavior_model::BehaviorModel,
    fsrs_adr::FSRSADR,
    fsrs_v6::{FSRSv6, FSRSv6State},
};

#[derive(Debug, Clone)]
pub struct SimResult {
    pub total_average_memorized: f64,
    pub total_cost: f64,
    pub weight: f32,
    pub days: f32,
    pub total_iters: i32,
}
impl SimResult {
    pub fn efficiency(&self) -> f64 {
        self.total_average_memorized / self.total_cost
    }
    pub fn memorized(&self) -> f64 {
        self.total_average_memorized / (self.weight as f64 * self.days as f64)
    }
}

fn get_proportion(t: f32, limit_t: f32, end_t: f32) -> f32 {
    if t < limit_t {
        1.0
    } else if t == end_t {
        0.0
    } else {
        1.0 - (t - limit_t) / (end_t - limit_t)
    }
}

fn simulate_review_card<T: Rng>(
    start_weight: f32,
    start_t: f32,
    limit_t: f32,
    end_t: f32,
    start_state: FSRSv6State,
    predictor: &FSRSv6,
    adr_model: &FSRSADR,
    behavior_model: &BehaviorModel,
    rng: &mut T,
) -> SimResult {
    let mut weight = start_weight;
    let mut t = start_t;
    let mut state = start_state;
    let mut accum_sim_result = SimResult { total_average_memorized: 0.0, total_cost: 0.0, weight: start_weight, days: end_t - start_t, total_iters: 0 };
    let mut split = weight > 1.0;
    let mut monte_carlo_len = 0;
    let mut monte_carlo_prune_len = 0;
    let mut monte_carlo_start_t = 0.0;
    let mut monte_carlo_memorized = 0.0;
    let mut monte_carlo_cost = 0.0;
    let mut can_prune = false;

    loop {
        let dr = adr_model.get_dr(state.s, state.d);
        let (interval, r) = predictor.schedule(&state, dr);
        let review_day = f32::min(t + interval, end_t);
        let time_existing_in_memory = review_day - t;
        let review_day_proportion = get_proportion(review_day, limit_t, end_t);
        if !split {
            // Early return of random simulations
            let monte_carlo_elapsed_t = t - monte_carlo_start_t;
            if can_prune && monte_carlo_len > monte_carlo_prune_len && monte_carlo_start_t + 2.0 * monte_carlo_elapsed_t < end_t {
                let memorized_vol_per_day = monte_carlo_memorized / monte_carlo_elapsed_t;
                let cost_per_day = monte_carlo_cost / monte_carlo_elapsed_t;
                let remaining_day_volume = {
                    let rect = (limit_t - t).max(0.0);
                    let triangle = 0.5 * get_proportion(t, limit_t, end_t) * (end_t - f32::max(limit_t, t));
                    rect + triangle
                };
                let est_memorized = weight * memorized_vol_per_day * remaining_day_volume;
                let est_cost = weight * cost_per_day * remaining_day_volume;
                accum_sim_result.total_average_memorized += est_memorized as f64;
                accum_sim_result.total_cost += est_cost as f64;

                return accum_sim_result;
            }
        }

        // total memorized is the same between all rating options
        accum_sim_result.total_average_memorized += {
            if t < limit_t && limit_t < review_day {
                let volume = 
                    predictor.forgetting_curve_volume(&state, limit_t - t)
                    + predictor.forgetting_curve_volume_weighted(
                        &state, 
                        limit_t - t, 
                        review_day - t,
                        1.0,
                        get_proportion(review_day, limit_t, end_t));

                weight * volume
            } else if t < limit_t {
                weight * predictor.forgetting_curve_volume(&state, time_existing_in_memory)
            } else {
                weight * predictor.forgetting_curve_volume_weighted(
                    &state, 
                    0.0, 
                    time_existing_in_memory, 
                    get_proportion(t, limit_t, end_t),
                    review_day_proportion,
                )
            }
        } as f64;

        if review_day >= end_t {
            break
        }
        if split && weight <= 1.0 {
            // Setup monte carlo
            monte_carlo_start_t = t;
            split = false;
            can_prune = rng.random_bool(0.99);
            monte_carlo_prune_len = rng.random_range(32..64);
        }
        let probs = behavior_model.review_rating_prob_dist(r);
        let (cont_rating_idx, cont_prob) = 
            if split {
                let mut max_idx = 0usize;
                let mut max_value = probs[0];
                for i in 1..probs.len() {
                    if probs[i] > max_value {
                        max_value = probs[i];
                        max_idx = i;
                    }
                }
                (max_idx, max_value)
            } else {
                (behavior_model.sample_review_rating_idx(r, rng), 1.0)
            };
        if !split && can_prune {
            monte_carlo_len += 1;
            monte_carlo_memorized += predictor.forgetting_curve_volume(&state, time_existing_in_memory);
            monte_carlo_cost += behavior_model.review_cost(cont_rating_idx);
        }
        if split {
            for i in 0..probs.len() as i32 {
                if i as usize == cont_rating_idx {
                    continue
                }
                let rating_idx = i as usize;
                let rating: i32 = i + 1;
                let next_weight = weight * probs[rating_idx];
                accum_sim_result.total_cost += (next_weight * review_day_proportion * behavior_model.review_cost(rating_idx)) as f64;

                let split_result = simulate_review_card(
                    next_weight,
                    review_day,
                    limit_t,
                    end_t,
                    predictor.transition(&state, rating, interval),
                    &predictor,
                    &adr_model,
                    &behavior_model,
                    rng,
                );
                accum_sim_result.total_average_memorized += split_result.total_average_memorized;
                accum_sim_result.total_cost += split_result.total_cost;
                accum_sim_result.total_iters += split_result.total_iters;
            }
        }

        let next_weight = weight * cont_prob;
        accum_sim_result.total_cost += (weight * review_day_proportion * cont_prob * behavior_model.review_cost(cont_rating_idx)) as f64;
        accum_sim_result.total_iters += 1;

        // Prepare state for the next iteration
        weight = next_weight;
        t = review_day;
        state = predictor.transition(&state, cont_rating_idx as i32 + 1, interval);
    }
    // if l > 1000 {
    //     println!("start {} end {} t = {} len = {}", start_weight, weight, start_t, l);
    // }
    accum_sim_result
}

pub fn simulate<T: Rng>(
    weight: f32,
    deck_size: i32,
    new_cards_per_day: i32,
    end_t: f32,
    predictor: &FSRSv6,
    adr_model: &FSRSADR,
    behavior_model: &BehaviorModel,
    rng: &mut T,
) -> SimResult {
    let learn_days = deck_size as f32 / new_cards_per_day as f32;
    let limit_t = f32::max(0.0, end_t - learn_days);
    let mut accum_sim_result = SimResult { total_average_memorized: 0.0, total_cost: 0.0, weight: weight, days: end_t, total_iters: 0 };
    for rating_idx in 0..4 {
        let rating = rating_idx as i32 + 1;
        let p = behavior_model.initial_rating_prob(rating_idx);
        accum_sim_result.total_cost += (p * weight * behavior_model.initial_cost(rating_idx)) as f64;
        let init_state = predictor.first_review(rating);
        let split_result = simulate_review_card(p * weight, 0.0, limit_t, end_t, init_state, &predictor, &adr_model, &behavior_model, rng);
        accum_sim_result.total_average_memorized += split_result.total_average_memorized;
        accum_sim_result.total_cost += split_result.total_cost;
        accum_sim_result.total_iters += split_result.total_iters;
    }
    accum_sim_result
}

#[derive(Debug, Clone)]
pub struct SafetySummary {
    pub checks: i32,
    pub interval_flips: i32,
    pub hard_shortens: i32,
    pub dr_p10: f32,
    pub dr_mean: f32,
    pub dr_p90: f32,
    pub aggression: f32,
}

#[derive(Debug, Clone)]
pub struct DrSummary {
    pub samples: i64,
    pub dr_p10: f32,
    pub dr_mean: f32,
    pub dr_p90: f32,
    pub aggression: f32,
}

#[derive(Debug, Clone, Copy)]
struct SafetyNode {
    weight: f32,
    t: f32,
    state: FSRSv6State,
    elapsed: f32,
}

fn weighted_percentile(values: &[(f32, f32)], percentile: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
    let total_weight: f32 = sorted.iter().map(|(_, w)| *w).sum();
    if total_weight <= 0.0 {
        return sorted[sorted.len() / 2].0;
    }
    let target = total_weight * percentile.clamp(0.0, 1.0);
    let mut accum = 0.0;
    for (value, weight) in sorted {
        accum += weight;
        if accum >= target {
            return value;
        }
    }
    values[values.len() - 1].0
}

pub fn dr_summary_by_weight(
    predictor: &FSRSv6,
    adr_model: &FSRSADR,
    behavior_model: &BehaviorModel,
    days: f32,
    start_weight: f32,
    prune_weight: f32,
) -> DrSummary {
    let mut stack = Vec::new();
    let mut samples = 0i64;
    let mut dr_values: Vec<(f32, f32)> = Vec::new();
    let mut dr_weight_sum = 0.0f32;
    let mut dr_weighted_sum = 0.0f32;

    for rating_idx in 0..4 {
        let rating = rating_idx as i32 + 1;
        let weight = start_weight * behavior_model.initial_rating_prob(rating_idx);
        if weight <= 0.0 {
            continue;
        }

        let state = predictor.first_review(rating);
        let dr = adr_model.get_dr(state.s, state.d);
        if dr.is_finite() {
            dr_values.push((dr, weight));
            dr_weight_sum += weight;
            dr_weighted_sum += dr * weight;
            samples += 1;
        }

        let interval = predictor.get_interval(&state, dr).round().max(1.0);
        if weight >= prune_weight && interval.is_finite() && interval > 0.0 {
            stack.push(SafetyNode {
                weight,
                t: 0.0,
                state,
                elapsed: interval,
            });
        }
    }

    while let Some(node) = stack.pop() {
        let review_t = node.t + node.elapsed;
        if review_t >= days {
            continue;
        }

        let r = predictor.forgetting_curve(&node.state, node.elapsed);
        let probs = behavior_model.review_rating_prob_dist(r);

        for rating_idx in 0..4 {
            let rating = rating_idx as i32 + 1;
            let child_weight = node.weight * probs[rating_idx].max(0.0);
            if child_weight <= 0.0 {
                continue;
            }

            let post = predictor.transition(&node.state, rating, node.elapsed);
            let dr = adr_model.get_dr(post.s, post.d);
            if dr.is_finite() {
                dr_values.push((dr, child_weight));
                dr_weight_sum += child_weight;
                dr_weighted_sum += dr * child_weight;
                samples += 1;
            }

            if child_weight < prune_weight {
                continue;
            }

            let interval = predictor.get_interval(&post, dr).round().max(1.0);
            if interval.is_finite() && interval > 0.0 {
                stack.push(SafetyNode {
                    weight: child_weight,
                    t: review_t,
                    state: post,
                    elapsed: interval,
                });
            }
        }
    }

    let dr_p10 = weighted_percentile(&dr_values, 0.10);
    let dr_p90 = weighted_percentile(&dr_values, 0.90);
    let dr_mean = if dr_weight_sum > 0.0 {
        dr_weighted_sum / dr_weight_sum
    } else {
        0.0
    };

    DrSummary {
        samples,
        dr_p10,
        dr_mean,
        dr_p90,
        aggression: dr_p90 - dr_p10,
    }
}

pub fn safety_summary(
    predictor: &FSRSv6,
    adr_model: &FSRSADR,
    behavior_model: &BehaviorModel,
    days: f32,
    s_max: f32,
    max_checks: i32,
) -> SafetySummary {
    let mut stack = Vec::new();
    for rating_idx in 0..4 {
        let rating = rating_idx as i32 + 1;
        let weight = behavior_model.initial_rating_prob(rating_idx);
        if weight <= 0.0 {
            continue;
        }
        let state = predictor.first_review(rating);
        let dr = adr_model.get_dr(state.s, state.d);
        let interval = predictor.get_interval(&state, dr).round().max(1.0);
        stack.push(SafetyNode {
            weight,
            t: 0.0,
            state,
            elapsed: interval,
        });
    }

    let mut checks = 0;
    let mut interval_flips = 0;
    let mut hard_shortens = 0;
    let mut dr_values: Vec<(f32, f32)> = Vec::new();
    let mut dr_weight_sum = 0.0f32;
    let mut dr_weighted_sum = 0.0f32;

    while let Some(node) = stack.pop() {
        if checks >= max_checks {
            break;
        }
        let review_t = node.t + node.elapsed;
        if review_t >= days {
            continue;
        }

        let r = predictor.forgetting_curve(&node.state, node.elapsed);
        let probs = behavior_model.review_rating_prob_dist(r);
        let mut intervals = [0.0f32; 4];
        let mut post_states = [FSRSv6State { s: 0.0, d: 0.0 }; 4];

        for rating_idx in 0..4 {
            let rating = rating_idx as i32 + 1;
            let post = predictor.transition(&node.state, rating, node.elapsed);
            let dr = adr_model.get_dr(post.s, post.d);
            intervals[rating_idx] = predictor.get_interval(&post, dr);
            post_states[rating_idx] = post;

            let dr_weight = node.weight * probs[rating_idx].max(0.0);
            if dr_weight > 0.0 {
                dr_values.push((dr, dr_weight));
                dr_weight_sum += dr_weight;
                dr_weighted_sum += dr * dr_weight;
            }
        }

        if node.state.s < s_max {
            checks += 1;
            let eps = 1e-4;
            if intervals[0] > intervals[1] + eps
                || intervals[1] > intervals[2] + eps
                || intervals[2] > intervals[3] + eps
            {
                interval_flips += 1;
            }
            if intervals[1] + eps < node.elapsed {
                hard_shortens += 1;
            }
        }

        for rating_idx in (0..4).rev() {
            let child_weight = node.weight * probs[rating_idx].max(0.0);
            if child_weight < 1e-6 {
                continue;
            }
            let interval = intervals[rating_idx].round().max(1.0);
            if interval.is_finite() && interval > 0.0 {
                stack.push(SafetyNode {
                    weight: child_weight,
                    t: review_t,
                    state: post_states[rating_idx],
                    elapsed: interval,
                });
            }
        }
    }

    let dr_p10 = weighted_percentile(&dr_values, 0.10);
    let dr_p90 = weighted_percentile(&dr_values, 0.90);
    let dr_mean = if dr_weight_sum > 0.0 {
        dr_weighted_sum / dr_weight_sum
    } else {
        0.0
    };

    SafetySummary {
        checks,
        interval_flips,
        hard_shortens,
        dr_p10,
        dr_mean,
        dr_p90,
        aggression: dr_p90 - dr_p10,
    }
}
