use rand::Rng;

#[derive(Clone, Copy)]
pub struct BehaviorModel {
    initial_rating_prob: [f32; 4],
    initial_cost: [f32; 4],
    review_rating_prob_given_success: [f32; 3],
    review_cost: [f32; 4],
}

impl BehaviorModel {
    pub fn new(
        initial_rating_prob: [f32; 4],
        initial_cost: [f32; 4],
        review_rating_prob_given_success: [f32; 3],
        review_cost: [f32; 4],
    ) -> Self {
        Self {
            initial_rating_prob,
            initial_cost,
            review_rating_prob_given_success,
            review_cost,
        }
    }
    #[inline]
    pub fn initial_rating_prob(&self, rating_idx: usize) -> f32 {
        self.initial_rating_prob[rating_idx]
    }
    #[inline]
    pub fn initial_cost(&self, rating_idx: usize) -> f32 {
        self.initial_cost[rating_idx]
    }
    #[inline]
    pub fn review_rating_prob_dist(&self, r: f32) -> [f32; 4] {
        let mut out = [0.0; 4];
        out[0] = 1.0 - r;
        for i in 1..4 {
            out[i] = r * self.review_rating_prob_given_success[i - 1];
        }
        out
    }
    #[inline]
    pub fn review_cost(&self, rating: usize) -> f32 {
        self.review_cost[rating]
    }
    pub fn sample_review_rating_idx<R: Rng>(
        &self,
        r: f32,
        rng: &mut R,
    ) -> usize {
        let u: f32 = rng.random();

        // p(0)
        let mut cum = 1.0 - r;
        if u < cum {
            return 0;
        }

        // p(1), p(2), p(3)
        for (i, &p) in self.review_rating_prob_given_success.iter().enumerate() {
            cum += r * p;
            if u < cum {
                return i + 1;
            }
        }

        // numerical fallback, return 'Good'
        2
    }
}
