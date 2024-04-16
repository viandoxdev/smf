use std::{
    mem::{self, MaybeUninit},
    slice,
};

use config::{FontConfig, GlobalConfig};
use error::SMFError;
use font::{Font, RasterKind};
use jni::{
    objects::{JByteArray, JByteBuffer, JClass, JObject, JValueGen},
    sys::{jint, jlong, jobjectArray, jvalue, jobject},
    JNIEnv,
};

mod textures;
mod config;
mod error;
mod font;
mod packing;

fn throw_if_err<T, F: FnOnce(&mut JNIEnv) -> Result<T, SMFError>>(env: &mut JNIEnv, f: F) -> T {
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

/// Load Fonts from a direct ByteBuffer, assumes the ByteBuffer will live for at least as long as
/// the corresponding Font structs. Returns an array of Java Font Objects.
#[no_mangle]
pub unsafe extern "system" fn Java_dev_vndx_bindings_NativeKt_loadFonts<'l>(
    mut env: JNIEnv<'l>,
    _class: JClass<'l>,
    data: JByteBuffer<'l>,
    config: JObject<'l>,
) -> jobjectArray {
    throw_if_err(&mut env, |env| {
        let bytes = 
            slice::from_raw_parts(
                env.get_direct_buffer_address(&data)?,
                env.get_direct_buffer_capacity(&data)?,
            );

        let config = FontConfig::from_java(env, config)?;
        let fonts = Font::from_bytes(&bytes, config)?;
        let class = env.find_class("dev/vndx/bindings/Font")?;

        let mut array = env.new_object_array(fonts.len() as i32, &class, JObject::null())?;
        for (i, font) in fonts.into_iter().enumerate() {
            let config = JObject::from_raw(font.config.to_java(env)?);
            let name = env.new_string(font.name())?;
            let addr = JValueGen::Long(Box::into_raw(Box::new(font)) as usize as i64);
            let res = env.new_object(
                &class,
                "(JLjava/lang/String;Ljava/nio/ByteBuffer;Ldev/vndx/bindings/FontConfig;)V",
                &[addr, JValueGen::Object(&name), JValueGen::Object(&data), JValueGen::Object(&config)],
            )?;
            env.set_object_array_element(&mut array, i as i32, res)?;
        }

        Ok(array.into_raw())
    })
}

#[no_mangle]
pub unsafe extern "system" fn Java_dev_vndx_bindings_NativeKt_destroyFont<'l>(
    mut _env: JNIEnv<'l>,
    _class: JClass<'l>,
    addr: jlong,
) {
    mem::drop(Box::from_raw(addr as usize as *mut Font));
}

#[no_mangle]
pub unsafe extern "system" fn Java_dev_vndx_bindings_NativeKt_setGlobalConfig<'l>(
    mut env: JNIEnv<'l>,
    _class: JClass<'l>,
    config: JObject<'l>
) {
    throw_if_err(&mut env, |env| {
        let config = GlobalConfig::from_java(env, config)?;
        config.update();
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "system" fn Java_dev_vndx_bindings_NativeKt_getGlobalConfig<'l>(
    mut env: JNIEnv<'l>,
    _class: JClass<'l>,
) -> jobject {
    throw_if_err(&mut env, |env| {
        Ok(GlobalConfig::get_copy().to_java(env)?)
    })
}
