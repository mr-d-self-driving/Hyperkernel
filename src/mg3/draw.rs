pub fn normal_bf16(name: &str, stream: usize, bytes: usize) -> Vec<u8> {
    let (basis, prime, gamma) =
        (0xcbf2_9ce4_8422_2325u64, 0x0000_0100_0000_01b3u64, 0x9e37_79b9_7f4a_7c15u64);
    let h = name.bytes().fold(basis, |h, b| (h ^ u64::from(b)).wrapping_mul(prime));
    let seed = (h ^ (stream as u64).wrapping_mul(gamma)) | 1;
    let unit = |c: u64| {
        ((mix(seed.wrapping_add(c.wrapping_add(1).wrapping_mul(gamma))) >> 11) + 1) as f64
            / (1u64 << 53) as f64
    };
    let normal = |i: u64| (-2.0 * unit(2 * i).ln()).sqrt() * (std::f64::consts::TAU * unit(2 * i + 1)).cos();
    (0..bytes as u64 / 2).flat_map(|i| bf16(normal(i) as f32).to_le_bytes()).collect()
}

fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

pub fn bf16(f: f32) -> u16 {
    let f = if f.is_nan() { f } else { f.clamp(-6e4, 6e4) };
    let bits = f.to_bits();
    if f.is_nan() {
        return ((bits >> 16) | 0x40) as u16;
    }
    ((bits + (0x7fff + ((bits >> 16) & 1))) >> 16) as u16
}

#[cfg(test)]
mod tests {
    use super::{bf16, normal_bf16};

    #[test]
    fn a_draw_is_its_names_and_streams() {
        assert_eq!(normal_bf16("", 0, 8), [46, 63, 48, 63, 171, 63, 130, 191]);
        assert_eq!(normal_bf16("noise", 1, 64), normal_bf16("noise", 1, 64));
        assert_ne!(normal_bf16("noise", 1, 64), normal_bf16("noise", 2, 64));
        let v: Vec<f32> = normal_bf16("noise", 3, 2 << 16)
            .chunks(2)
            .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
            .collect();
        let mean = v.iter().sum::<f32>() / v.len() as f32;
        let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / v.len() as f32;
        assert!(mean.abs() < 0.02 && (var - 1.0).abs() < 0.03, "{mean} {var}");
    }

    #[test]
    fn bf16_rounds_to_nearest_even() {
        assert_eq!(
            (bf16(1.0), bf16(-2.0), bf16(f32::from_bits(0x3f80_8000)), bf16(f32::from_bits(0x3f81_8000))),
            (0x3f80, 0xc000, 0x3f80, 0x3f82)
        );
    }
}
