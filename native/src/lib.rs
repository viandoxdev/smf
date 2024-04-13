use jni::{JNIEnv, objects::{JClass, JString}, sys::jstring};

#[no_mangle]
pub extern "system" fn Java_dev_vndx_bindings_NativeKt_test_1jni<'l>(mut env: JNIEnv<'l>, _class: JClass<'l>, input: JString<'l>) -> jstring {
    let input: String = env.get_string(&input).expect("Couldn't get java string").into();
    let res = env.new_string(format!("The string '{input}' has {} characters.", input.len())).expect("Couldn't create java string");
    res.into_raw()
}
