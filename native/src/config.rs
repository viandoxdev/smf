use std::sync::Arc;

use jni::{
    objects::{JObject, JValueGen},
    sys::jobject,
    JNIEnv,
};
use parking_lot::Mutex;
use rustybuzz::Language;

use crate::{error::SMFError, font::RasterKind, JNI};

#[derive(Clone)]
pub struct GlobalConfig {
    /// Atlas size, must be a power of 2
    pub atlas_size: u32,
    /// Padding for glyphs in atlasses
    pub glyph_padding: u32,
    /// Sin alpha parameter used in MSDF edge coloring
    pub sin_alpha: f64,
    /// Coloring seed used in MSDF generation
    pub coloring_seed: u64,
    pub set: bool,
}

impl Default for GlobalConfig {
    fn default() -> Self {
        Self::default_config()
    }
}

static GLOBAL_CONFIG: Mutex<GlobalConfig> = Mutex::new(GlobalConfig::default_config());

impl GlobalConfig {
    // Workaround for no const default yet
    pub const fn default_config() -> Self {
        // Non sensical value, GlobalConfig will be set on the java side
        Self {
            atlas_size: 0,
            glyph_padding: 0,
            sin_alpha: 0.0,
            coloring_seed: 0,
            set: false,
        }
    }

    pub fn update(&self) {
        let mut cfg = GLOBAL_CONFIG.lock();
        cfg.atlas_size = self.atlas_size;
        cfg.glyph_padding = self.glyph_padding;
        cfg.sin_alpha = self.sin_alpha;
        cfg.coloring_seed = self.coloring_seed;
        cfg.set = true;
    }

    pub fn is_set(&self) -> bool {
        self.set
    }

    pub fn get_copy() -> GlobalConfig {
        GLOBAL_CONFIG.lock().clone()
    }
}

#[derive(Clone)]
pub struct FontConfig {
    pub raster_kind: RasterKind,
    pub scale: f32,
    pub line_height: f32,
    pub language: Arc<Language>,
}

