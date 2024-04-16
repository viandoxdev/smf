use jni::{
    objects::{JObject, JValueGen},
    sys::jobject,
    JNIEnv,
};
use parking_lot::Mutex;

use crate::{error::SMFError, font::RasterKind};

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
    set: bool,
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

    pub fn from_java(env: &mut JNIEnv, obj: JObject) -> Result<Self, SMFError> {
        Ok(Self {
            atlas_size: env.get_field(&obj, "atlasSize", "I")?.i()? as u32,
            glyph_padding: env.get_field(&obj, "glyphPadding", "I")?.i()? as u32,
            sin_alpha: env.get_field(&obj, "sinAlpha", "D")?.d()?,
            coloring_seed: env.get_field(&obj, "coloringSeed", "J")?.j()? as u64,
            set: false,
        })
    }

    pub fn to_java(&self, env: &mut JNIEnv) -> Result<jobject, SMFError> {
        let class = env.find_class("dev/vndx/bindings/GlobalConfig")?;
        let obj = env.new_object(
            &class,
            "(IIDJ)V",
            &[
                JValueGen::Int(self.atlas_size as i32),
                JValueGen::Int(self.glyph_padding as i32),
                JValueGen::Double(self.sin_alpha),
                JValueGen::Long(self.coloring_seed as i64),
            ],
        )?;
        Ok(obj.as_raw())
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
}

impl FontConfig {
    pub fn from_java(env: &mut JNIEnv, obj: JObject) -> Result<Self, SMFError> {
        Ok(Self {
            raster_kind: RasterKind::try_from(env.get_field(&obj, "rasterKind", "I")?.i()?)?,
            scale: env.get_field(&obj, "scale", "F")?.f()?,
        })
    }

    pub fn to_java(&self, env: &mut JNIEnv) -> Result<jobject, SMFError> {
        let class = env.find_class("dev/vndx/bindings/FontConfig")?;
        let obj = env.new_object(
            &class,
            "(IF)V",
            &[
                JValueGen::Int(self.raster_kind as i32),
                JValueGen::Float(self.scale),
            ],
        )?;
        Ok(obj.as_raw())
    }
}
