use std::{mem::MaybeUninit, str::FromStr, sync::Arc};

use jni::{
    objects::{JObject, JString, JValueGen, JClass},
    sys::{jboolean, jobject, jobjectArray},
    JNIEnv, descriptors::Desc,
};
use rustybuzz::Language;

use crate::{
    config::{FontConfig, GlobalConfig},
    error::SMFError,
    font::{BatchedCommands, BatchedResults, Command, Mesh, RasterKind}, textures::Diff,
};

// TODO: use unchecked variants (maybe with a cfg ?) to avoid unnecessary checks

pub fn throw_if_err<T, F: FnOnce(&mut JNIEnv) -> Result<T, SMFError>>(env: &mut JNIEnv, f: F) -> T {
    match (f)(env) {
        Ok(v) => v,
        Err(e) => {
            let _ = e.throw(env);
            // SAFETY: We threw an exception above so the returned value shouldn't be read.
            // > When throwing Java exceptions, the exception isn't handled until the native code
            // > gives back the control to Java, so we return as quickly as possible
            unsafe { MaybeUninit::uninit().assume_init() }
        }
    }
}

// TODO: Split trait into from/to variants and remove error associated type and just hardcode SMFError

pub trait JNI
where
    Self: Sized,
{
    type Error;
    fn from_jni(env: &mut JNIEnv, obj: JObject) -> Result<Self, Self::Error>;
    fn to_jni(self, env: &mut JNIEnv) -> Result<jobject, Self::Error>;
}

impl JNI for FontConfig {
    type Error = SMFError;
    fn from_jni(env: &mut JNIEnv, obj: JObject) -> Result<Self, Self::Error> {
        let lang = JString::from(env.get_field(&obj, "language", "Ljava/lang/String;")?.l()?);
        let lang_string: String = env.get_string(&lang)?.into();
        Ok(Self {
            raster_kind: RasterKind::try_from(env.get_field(&obj, "rasterKind", "I")?.i()?)?,
            scale: env.get_field(&obj, "scale", "F")?.f()?,
            raster_scale: env.get_field(&obj, "rasterScale", "F")?.f()?,
            line_height: env.get_field(&obj, "lineHeight", "F")?.f()?,
            language: Arc::from(
                Language::from_str(&lang_string).map_err(|e| SMFError::LanguageParsingError(e))?,
            ),
        })
    }

    fn to_jni(self, env: &mut JNIEnv) -> Result<jobject, Self::Error> {
        let class = env.find_class("dev/vndx/bindings/FontConfig")?;
        let lang = env.new_string(self.language.as_str())?;
        let obj = env.new_object(
            &class,
            "(IFFFLjava/lang/String;)V",
            &[
                JValueGen::Int(self.raster_kind as i32),
                JValueGen::Float(self.scale),
                JValueGen::Float(self.raster_scale),
                JValueGen::Float(self.line_height),
                JValueGen::Object(&lang),
            ],
        )?;
        Ok(obj.as_raw())
    }
}

impl JNI for GlobalConfig {
    type Error = SMFError;

    fn from_jni(env: &mut JNIEnv, obj: JObject) -> Result<Self, Self::Error> {
        Ok(Self {
            atlas_size: env.get_field(&obj, "atlasSize", "I")?.i()? as u32,
            glyph_padding: env.get_field(&obj, "glyphPadding", "I")?.i()? as u32,
            sin_alpha: env.get_field(&obj, "sinAlpha", "D")?.d()?,
            coloring_seed: env.get_field(&obj, "coloringSeed", "J")?.j()? as u64,
            set: false,
        })
    }

    fn to_jni(self, env: &mut JNIEnv) -> Result<jobject, Self::Error> {
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
}

impl JNI for Command {
    type Error = SMFError;
    fn from_jni(env: &mut JNIEnv, obj: JObject) -> Result<Self, Self::Error> {
        let str_obj = env.get_field(&obj, "string", "Ljava/lang/String;")?.l()?;
        let float_obj = env.get_field(&obj, "maxLength", "Ljava/lang/Float;")?.l()?;
        Ok(Self {
            str: env.get_string(&str_obj.into())?.into(),
            multiline: env.get_field(&obj, "multiLine", "Z")?.z()?,
            max_length: if float_obj.is_null() {
                None
            } else {
                Some(env.call_method(&float_obj, "floatValue", "()F", &[])?.f()?)
            },
        })
    }

    fn to_jni(self, env: &mut JNIEnv) -> Result<jobject, Self::Error> {
        unimplemented!()
    }
}

impl JNI for Mesh {
    type Error = SMFError;

    fn from_jni(env: &mut JNIEnv, obj: JObject) -> Result<Self, Self::Error> {
        unimplemented!()
    }

    fn to_jni(self, env: &mut JNIEnv) -> Result<jobject, Self::Error> {
        let class = env.find_class("dev/vndx/bindings/Mesh")?;

        let (vertices_addr, vertices_len) = (self.vertices.as_ptr() as usize, self.vertices.len());
        let (indices_addr, indices_len) = (self.indices.as_ptr() as usize, self.indices.len());

        // We need to forget these or they would get dropped at the end of this function
        std::mem::forget(self.vertices);
        std::mem::forget(self.indices);

        let obj = env.new_object(
            &class,
            "(IJJJJ)V",
            &[
                JValueGen::Int(self.texture as i32),
                JValueGen::Long(vertices_addr as i64),
                JValueGen::Long(vertices_len as i64),
                JValueGen::Long(indices_addr as i64),
                JValueGen::Long(indices_len as i64),
            ],
        )?;

        Ok(obj.as_raw())
    }
}

impl JNI for Diff {
    type Error = SMFError;
    fn from_jni(env: &mut JNIEnv, obj: JObject) -> Result<Self, Self::Error> {
        unimplemented!()
    }

    fn to_jni(self, env: &mut JNIEnv) -> Result<jobject, Self::Error> {
        match self {
            Diff::textureUpdate { tex, channel_count, damage, size, data } => {
                let class = env.find_class("dev/vndx/bindings/Diff$TextureUpdate")?;

                let obj = env.new_object(&class, "(IIIIIIIJ)V", &[
                    JValueGen::Int(tex as i32),
                    JValueGen::Int(channel_count as i32),
                    JValueGen::Int(damage.x1 as i32),
                    JValueGen::Int(damage.y1 as i32),
                    JValueGen::Int(damage.x2 as i32),
                    JValueGen::Int(damage.y2 as i32),
                    JValueGen::Int(size as i32),
                    JValueGen::Long(data as usize as i64),
                ])?;

                Ok(obj.as_raw())
            }
            Diff::textureCreation { tex, channel_count, size, data } => {
                let class = env.find_class("dev/vndx/bindings/Diff$TextureCreation")?;

                let obj = env.new_object(&class, "(IIIJ)V", &[
                    JValueGen::Int(tex as i32),
                    JValueGen::Int(channel_count as i32),
                    JValueGen::Int(size as i32),
                    JValueGen::Long(data as usize as i64),
                ])?;

                Ok(obj.as_raw())
            }
        }
    }
}

impl JNI for BatchedResults {
    type Error = SMFError;

    fn from_jni(env: &mut JNIEnv, obj: JObject) -> Result<Self, Self::Error> {
        unimplemented!()
    }

    fn to_jni(self, env: &mut JNIEnv) -> Result<jobject, Self::Error> {
        let class = env.find_class("dev/vndx/bindings/BatchedResults")?;
        let mesharr_class = env.find_class("[Ldev/vndx/bindings/Mesh;")?;
        let mesh_class = env.find_class("dev/vndx/bindings/Mesh")?;
        let diff_class = env.find_class("dev/vndx/bindings/Diff")?;

        let meshes_list = env.new_object_array(self.meshes.len() as i32, &mesharr_class, JObject::null())?;
        for (i, res) in self.meshes.into_iter().enumerate() {
            let vec = res?;
            let meshes = env.new_object_array(vec.len() as i32, &mesh_class, JObject::null())?;
            for (i, mesh) in vec.into_iter().enumerate() {
                let raw = mesh.to_jni(env)?;
                env.set_object_array_element(&meshes, i as i32, unsafe { JObject::from_raw(raw) })?;
            }

            env.set_object_array_element(&meshes_list, i as i32, meshes)?;
        }

        let diffs = env.new_object_array(self.diffs.len() as i32, &diff_class, JObject::null())?;
        for (i, diff) in self.diffs.into_iter().enumerate() {
            let raw = diff.to_jni(env)?;
            env.set_object_array_element(&diffs, i as i32, unsafe { JObject::from_raw(raw) })?;
        }

        let obj = env.new_object(&class, "([[Ldev/vndx/bindings/Mesh;[Ldev/vndx/bindings/Diff;)V", &[
            JValueGen::Object(&meshes_list),
            JValueGen::Object(&diffs),
        ])?;

        Ok(obj.as_raw())
    }
}
