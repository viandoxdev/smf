use std::{
    mem::{self, MaybeUninit},
    slice,
};

use config::{FontConfig, GlobalConfig};
use error::SMFError;
use font::{BatchedCommands, Font, RasterKind, Command};
use java::{throw_if_err, JNI};
use jni::{
    objects::{JByteArray, JByteBuffer, JClass, JObject, JObjectArray, JValueGen},
    sys::{jint, jlong, jobject, jobjectArray, jvalue},
    JNIEnv,
};

mod config;
mod error;
mod font;
mod java;
mod packing;
mod textures;

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
        let bytes = slice::from_raw_parts(
            env.get_direct_buffer_address(&data)?,
            env.get_direct_buffer_capacity(&data)?,
        );

        let config = FontConfig::from_jni(env, config)?;
        let fonts = Font::from_bytes(&bytes, config)?;
        let class = env.find_class("dev/vndx/bindings/Font")?;

        let array = env.new_object_array(fonts.len() as i32, &class, JObject::null())?;
        for (i, font) in fonts.into_iter().enumerate() {
            let config = JObject::from_raw(font.config.clone().to_jni(env)?);
            let name = env.new_string(font.name())?;
            let addr = JValueGen::Long(Box::into_raw(Box::new(font)) as usize as i64);
            let el = env.new_object(
                &class,
                "(JLjava/lang/String;Ljava/nio/ByteBuffer;Ldev/vndx/bindings/FontConfig;)V",
                &[
                    addr,
                    JValueGen::Object(&name),
                    JValueGen::Object(&data),
                    JValueGen::Object(&config),
                ],
            )?;

            env.set_object_array_element(&array, i as i32, el)?;
        }

        Ok(array.as_raw())
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
    config: JObject<'l>,
) {
    throw_if_err(&mut env, |env| {
        let config = GlobalConfig::from_jni(env, config)?;
        config.update();
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "system" fn Java_dev_vndx_bindings_NativeKt_getGlobalConfig<'l>(
    mut env: JNIEnv<'l>,
    _class: JClass<'l>,
) -> jobject {
    throw_if_err(&mut env, |env| Ok(GlobalConfig::get_copy().to_jni(env)?))
}

#[no_mangle]
pub unsafe extern "system" fn Java_dev_vndx_bindings_NativeKt_destroyMesh<'l>(
    _env: JNIEnv<'l>,
    _class: JClass<'l>,
    vertices_addr: jlong,
    vertices_len: jlong,
    indices_addr: jlong,
    indices_len: jlong,
) {
    let vertices_ptr =
        slice::from_raw_parts_mut(vertices_addr as *mut f32, vertices_len as usize) as *mut [f32];
    let indices_ptr =
        slice::from_raw_parts_mut(indices_addr as *mut u32, indices_len as usize) as *mut [u32];
    mem::drop(Box::from_raw(vertices_ptr));
    mem::drop(Box::from_raw(indices_ptr));
}

#[no_mangle]
pub unsafe extern "system" fn Java_dev_vndx_bindings_NativeKt_fontProcessBatched<'l>(
    mut env: JNIEnv<'l>,
    _class: JClass<'l>,
    font_addr: jlong,
    commands: JObjectArray<'l>,
) -> jobject {
    throw_if_err(&mut env, |env| {
        // SAFETY: my balls itch
        let font = &mut *(font_addr as u64 as *mut Font);

        let commands_count = env.get_array_length(&commands)? as usize;
        let batched = BatchedCommands {
            commands: (0..commands_count).map(|i| {
                let obj = env.get_object_array_element(&commands, i as i32)?;
                Command::from_jni(env, obj)
            }).collect::<Result<Vec<Command>, SMFError>>()?,
        };

        let results = font.process_batched(batched);

        Ok(results.to_jni(env)?)
    })
}
