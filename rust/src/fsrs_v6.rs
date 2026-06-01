type FSRSv6Params = [f32; 21];
#[derive(Clone, Copy, Debug)]
pub struct FSRSv6State {
    pub s: f32,
    pub d: f32,
}

#[derive(Clone, Copy)]
pub struct FSRSv6 {
    w: FSRSv6Params,
}

// \int_a^b {interp(a, b, x) * (1+zx)^d dx} where interp(a, b, x) is a linear interpolation satisfying interp(a, b, a) = l and interp(a, b, b) = u for some l, u
#[inline]
pub fn linear_weighted_forgetting_curve_volume(
    a: f32,
    b: f32,
    l: f32,
    u: f32,
    z: f32,
    d: f32,
) -> f32 {
    // z == 0: (1 + z x)^d == 1
    // if z == 0.0 {
    //     return 0.5 * (l + u) * (b - a);
    // }

    // Generic case: z != 0, d != -1, -2
    let a_term = 1.0 + z * a;
    let b_term = 1.0 + z * b;

    let d1 = d + 1.0;
    let d2 = d + 2.0;

    let a_d1 = a_term.powf(d1);
    let b_d1 = b_term.powf(d1);
    let a_d2 = a_term.powf(d2);
    let b_d2 = b_term.powf(d2);

    // ∫_a^b (1 + z x)^d dx
    let i0 = (b_d1 - a_d1) / (z * d1);

    // ∫_a^b (x - a)(1 + z x)^d dx
    let i1 = (1.0 / (z * z))
        * ((b_d2 - a_d2) / d2 - (b_d1 - a_d1) / d1)
        - (a / (z * d1)) * (b_d1 - a_d1);

    l * i0 + (u - l) / (b - a) * i1
}

impl FSRSv6 {
    const MIN_STABILITY: f32 = 0.1;
    const MAX_STABILITY: f32 = 365.0 * 25.0;

    pub fn new(params: FSRSv6Params) -> Self {
        Self { w: params }
    }
    fn init_d(&self, rating: i32) -> f32 {
        (self.w[4] - (self.w[5] * (rating - 1) as f32).exp() + 1.0).clamp(1.0, 10.0)
    }
    pub fn first_review(&self, rating: i32) -> FSRSv6State {
        let s = self.w[rating as usize - 1];
        let d = self.init_d(rating);
        FSRSv6State { s: s, d: d }
    }
    pub fn forgetting_curve(&self, state: &FSRSv6State, elapsed: f32) -> f32 {
        let decay = -self.w[20];
        let factor = 0.9_f32.powf(1.0 / decay) - 1.0;
        (1.0 + factor * elapsed / state.s).powf(decay)
    }
    pub fn schedule(&self, state: &FSRSv6State, dr: f32) -> (f32, f32) {
        let interval = self.get_interval(state, dr).round();
        (interval, self.forgetting_curve(state, interval))
    }
    pub fn forgetting_curve_volume(&self, state: &FSRSv6State, endpoint: f32) -> f32 {
        let decay = -self.w[20];
        let factor = 0.9_f32.powf(1.0 / decay) - 1.0;
        let a = factor / state.s;
        let denom = a * (decay + 1.0);
        ((1.0 + a * endpoint).powf(decay + 1.0) - 1.0) / denom
    }
    pub fn forgetting_curve_volume_weighted(
        &self,
        state: &FSRSv6State,
        a: f32,
        b: f32,
        l: f32,
        u: f32,
    ) -> f32 {
        let decay = -self.w[20];
        let factor = 0.9_f32.powf(1.0 / decay) - 1.0;
        linear_weighted_forgetting_curve_volume(a, b, l, u, factor / state.s, decay)
    }
    fn stability_short_term(&self, s: f32, rating: i32) -> f32 {
        let sinc = (self.w[17] * (rating as f32 - 3.0 + self.w[18])).exp()
            * s.powf(-self.w[19]);
        if rating >= 2 {
            s * f32::max(1.0, sinc)
        } else {
            s * sinc
        }
    }
    fn stability_after_success(&self, s: f32, r: f32, d: f32, rating: i32) -> f32 {
        let hard_penalty = if rating == 2 { self.w[15] } else { 1.0 };
        let easy_bonus = if rating == 4 { self.w[16] } else { 1.0 };
        let sinc = 
            self.w[8].exp()
            * (11.0 - d)
            * (s.powf(-self.w[9]))
            * (((1.0 - r) * self.w[10]).exp() - 1.0);
        s * (1.0 + sinc * hard_penalty * easy_bonus)
    }
    fn stability_after_failure(&self, s: f32, r: f32, d: f32) -> f32 {
        let new_s = 
            self.w[11]
            * (d.powf(-self.w[12]))
            * ((s + 1.0).powf(self.w[13]) - 1.0)
            * ((1.0 - r) * self.w[14]).exp();
        let new_min = s / (self.w[17] * self.w[18]).exp();
        return f32::min(new_s, new_min)
    }
    fn linear_damping(&self, delta_d: f32, old_d: f32) -> f32 {
        return delta_d * (10.0 - old_d) / 9.0
    }
    fn mean_reversion(&self, weight: f32, init: f32, current: f32) -> f32 {
        return weight * init + (1.0 - weight) * current
    }
    fn next_d(&self, d: f32, rating: i32) -> f32 {
        let delta_d = -self.w[6] * (rating - 3) as f32;
        let new_d = d + self.linear_damping(delta_d, d);
        let new_d = self.mean_reversion(self.w[7], self.init_d(4), new_d);
        new_d.clamp(1.0, 10.0)
    }
    pub fn transition(&self, state: &FSRSv6State, rating: i32, elapsed: f32) -> FSRSv6State {
        let r = self.forgetting_curve(&state, elapsed);
        let s = if elapsed < 1.0 {
            self.stability_short_term(state.s, rating)
        } else if rating > 1 {
            self.stability_after_success(state.s, r, state.d, rating)
        } else {
            self.stability_after_failure(state.s, r, state.d)
        };
        let s = s.clamp(Self::MIN_STABILITY, Self::MAX_STABILITY);
        let d = self.next_d(state.d, rating);
        FSRSv6State { s: s, d: d }
    }
    pub fn get_interval(&self, state: &FSRSv6State, dr: f32) -> f32 {
        let decay = -self.w[20];
        let factor = 0.9_f32.powf(1.0 / decay) - 1.0;
        f32::max(1.0, state.s / factor * (dr.powf(1.0 / decay) - 1.0))
    }
}
