#[derive(Clone, Debug)]
pub struct FSRSADR {
    pub flat: f32,
    pub s_multi: f32,
    pub d_multi: f32,
}

impl FSRSADR {
    pub fn linear(flat: f32, s_multi: f32, d_multi: f32) -> Self {
        Self {
            flat,
            s_multi,
            d_multi,
        }
    }

    pub fn get_dr(&self, s: f32, d: f32) -> f32 {
        let log_s = s.ln();
        let logit = self.flat + self.s_multi * log_s + self.d_multi * d;
        sigmoid(logit).clamp(0.0, 0.995)
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    let x = x.clamp(-10.0, 10.0);
    1.0 / (1.0 + (-x).exp())
}
