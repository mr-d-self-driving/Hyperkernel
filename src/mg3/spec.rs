use serde::Deserialize;

#[derive(Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub struct Clip {
    pub memory: usize,
    pub past: usize,
    pub latents: usize,
}

#[derive(Deserialize, Clone, Debug, PartialEq)]
pub struct Spec {
    pub name: String,
    pub ranks: usize,
    pub tokens: usize,
    pub grid: [usize; 2],
    pub patch_cols: usize,
    pub heads: usize,
    pub head: usize,
    pub turns: usize,
    pub sigma: f64,
    pub text: usize,
    pub text_dim: usize,
    pub frames: usize,
    pub windows: usize,
    pub key_in: usize,
    pub mouse_in: usize,
    pub key_cols: usize,
    pub mouse_cols: usize,
    pub clip: Clip,
}

impl Spec {
    pub fn latents(&self) -> usize {
        self.clip.memory + self.clip.latents
    }

    pub fn memory(&self) -> usize {
        self.clip.memory
    }

    pub fn past(&self) -> usize {
        self.clip.past
    }

    pub fn rows(&self) -> usize {
        self.latents() * self.tokens
    }

    pub fn split(&self) -> usize {
        (self.clip.memory + self.clip.past) * self.tokens
    }

    pub fn window(&self) -> usize {
        self.frames * self.windows
    }

    pub fn latent_bytes(&self) -> usize {
        self.tokens * self.patch_cols * 2
    }

    pub fn row_bytes(&self) -> usize {
        self.patch_cols * 2
    }
}

#[cfg(test)]
pub fn chunk() -> Spec {
    Spec {
        name: "mg3chunk".into(),
        ranks: 1,
        tokens: 880,
        grid: [22, 40],
        patch_cols: 192,
        heads: 24,
        head: 128,
        turns: 2048,
        sigma: 0.8,
        text: 512,
        text_dim: 4096,
        frames: 4,
        windows: 3,
        key_in: 6,
        mouse_in: 2,
        key_cols: 8,
        mouse_cols: 24,
        clip: Clip { memory: 5, past: 4, latents: 14 },
    }
}

#[cfg(test)]
pub fn first() -> Spec {
    Spec { name: "mg3first".into(), clip: Clip { memory: 0, past: 1, latents: 15 }, ..chunk() }
}

#[cfg(test)]
mod tests {
    use super::chunk;

    #[test]
    fn the_chunks_rows() {
        let s = chunk();
        assert_eq!((s.rows(), s.split(), s.window(), s.latent_bytes()), (16720, 7920, 12, 880 * 192 * 2));
    }
}
