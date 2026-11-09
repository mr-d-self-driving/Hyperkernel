use super::spec::Spec;

pub fn dit(s: &Spec) -> Vec<u8> {
    let pairs = s.head / 2;
    let (t, hw) = (pairs - 2 * (pairs / 3), pairs / 3);
    let mut out = Vec::with_capacity(s.heads * s.turns * 2 * pairs * 4);
    for h in 0..s.heads {
        let base = 1e4 * (1.0 + s.sigma * linspace(h, s.heads));
        let omega: Vec<f64> = [t, hw, hw]
            .iter()
            .flat_map(|&n| (0..n).map(move |k| 1.0 / base.powf(k as f64 / n as f64)))
            .collect();
        for p in 0..s.turns {
            out.extend(omega.iter().flat_map(|w| ((p as f64 * w).cos() as f32).to_le_bytes()));
            out.extend(omega.iter().flat_map(|w| ((p as f64 * w).sin() as f32).to_le_bytes()));
        }
    }
    out
}

fn linspace(i: usize, n: usize) -> f64 {
    if n == 1 {
        return -1.0;
    }
    let step = 2.0 / (n - 1) as f64;
    if i < n / 2 { -1.0 + step * i as f64 } else { 1.0 - step * (n - 1 - i) as f64 }
}

#[cfg(test)]
mod tests {
    use super::dit;
    use crate::mg3::spec;

    #[test]
    fn the_dit_table_turns_a_position_a_row() {
        let s = spec::chunk();
        let d: Vec<f32> = dit(&s).chunks(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        assert_eq!(d.len(), s.heads * s.turns * s.head);
        assert_eq!((d[s.head], d[s.head + s.head / 2]), (1f64.cos() as f32, 1f64.sin() as f32));
    }
}
