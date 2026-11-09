#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampling {
    pub top_k: usize,
    pub top_p: f32,
    pub temperature: f32,
    pub seed: u64,
}

impl Sampling {
    pub fn greedy() -> Sampling {
        Sampling { top_k: 1, top_p: 0.0, temperature: 1.0, seed: 0 }
    }

    pub fn draw(&self, row: &[f32], state: &mut u64) -> usize {
        let clean = |x: f32| if x.is_finite() { x } else { 0.0 };
        if self.top_k == 1 {
            return (0..row.len()).fold(0, |b, i| if clean(row[i]) > clean(row[b]) { i } else { b });
        }
        let mut top: Vec<usize> = (0..row.len()).collect();
        top.sort_by(|&a, &b| clean(row[b]).total_cmp(&clean(row[a])));
        if self.top_k > 0 {
            top.truncate(self.top_k.min(row.len()));
        }
        let scaled: Vec<f64> =
            top.iter().map(|&i| f64::from(clean(row[i])) / f64::from(self.temperature)).collect();
        let most = scaled.iter().copied().fold(f64::MIN, f64::max);
        let mut probs: Vec<f64> = scaled.iter().map(|x| (x - most).exp()).collect();
        let total: f64 = probs.iter().sum();
        probs.iter_mut().for_each(|x| *x /= total);
        if self.top_p > 0.0 && self.top_p < 1.0 {
            let mut below = 0.0;
            for x in probs.iter_mut().rev() {
                below += *x;
                if below <= 1.0 - f64::from(self.top_p) {
                    *x = 0.0;
                }
            }
        }
        let kept: f64 = probs.iter().sum();
        let u = splitmix(state) as f64 / (1u64 << 53) as f64 * kept;
        let mut seen = 0.0;
        for (i, x) in probs.iter().enumerate() {
            seen += x;
            if u < seen {
                return top[i];
            }
        }
        top[probs.iter().rposition(|x| *x > 0.0).unwrap_or(0)]
    }
}

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (z ^ (z >> 31)) >> 11
}

#[cfg(test)]
mod tests {
    use super::Sampling;

    #[test]
    fn greedy_is_the_first_greatest_past_a_nan_or_an_infinity() {
        let mut state = 0;
        let g = Sampling::greedy();
        assert_eq!(g.draw(&[0.0, 3.0, 3.0, -1.0], &mut state), 1);
        assert_eq!(g.draw(&[f32::NAN, f32::INFINITY, 2.0], &mut state), 2);
        assert_eq!(state, 0);
    }

    #[test]
    fn top_k_draws_follow_their_softmax_and_repeat_by_seed() {
        let s = Sampling { top_k: 2, top_p: 0.0, temperature: 1.0, seed: 7 };
        let row = [0.0, 2.0_f32.ln(), -5.0, -9.0];
        let draws = |seed: u64| {
            let mut state = seed;
            (0..30_000).map(|_| s.draw(&row, &mut state)).collect::<Vec<_>>()
        };
        let d = draws(7);
        assert_eq!(d, draws(7));
        assert_ne!(d, draws(8));
        assert!(d.iter().all(|&t| t < 2));
        let share = d.iter().filter(|&&t| t == 1).count() as f64 / d.len() as f64;
        assert!((share - 2.0 / 3.0).abs() < 0.02, "share {share}");
    }

    #[test]
    fn temperature_sharpens_and_top_p_drops_the_tail() {
        let row = [0.0, 1.0, 2.0, 3.0];
        let share = |s: Sampling, t: usize| {
            let mut state = s.seed;
            (0..20_000).filter(|_| s.draw(&row, &mut state) == t).count() as f64 / 20_000.0
        };
        let base = Sampling { top_k: 0, top_p: 0.0, temperature: 1.0, seed: 3 };
        assert!(share(Sampling { temperature: 0.25, ..base }, 3) > share(base, 3));
        assert!(share(Sampling { temperature: 4.0, ..base }, 3) < share(base, 3));
        assert_eq!(share(Sampling { top_p: 0.5, ..base }, 0), 0.0);
        assert!(share(Sampling { top_p: 0.5, ..base }, 3) > 0.99);
    }
}
