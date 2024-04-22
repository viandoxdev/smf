use jni::{JNIEnv, errors::Error as JniError};
use thiserror::Error;
use ttf_parser::FaceParsingError;

#[derive(Error, Debug)]
pub enum SMFError {
    #[error("Failed to parse font")]
    FailedParsing(Option<FaceParsingError>),
    #[error("Raster kind is outside the 0-2 range")]
    MalformedRasterKind,
    #[error("Global config hasn't been set yet, but one is required")]
    GlobalConfigNotSet,
    #[error("The glyph is an SVG or raster image, which aren't supported as of yet")]
    UnsupportedGlyphFormat,
    #[error("Couldn't fit glyph in atlas")]
    PackingError,
    #[error("Couldn't parse language string")]
    LanguageParsingError(&'static str),
    #[error("An unexpected (unreachable) error happened: {0}. Please report this")]
    ExtraError(String),
    #[error("Error with JNI")]
    JNI(JniError)
}

impl From<JniError> for SMFError {
    fn from(value: JniError) -> Self {
        Self::JNI(value)
    }
}

impl From<FaceParsingError> for SMFError {
    fn from(value: FaceParsingError) -> Self {
        Self::FailedParsing(Some(value))
    }
}

impl SMFError {
    pub fn throw(&self, env: &mut JNIEnv) -> jni::errors::Result<()> {
        match self {
            Self::JNI(e) => env.throw_new("java/lang/RuntimeException", e.to_string()),
            _ => env.throw_new("dev/vndx/bindings/NativeException", self.to_string()),
        }
    }
}
