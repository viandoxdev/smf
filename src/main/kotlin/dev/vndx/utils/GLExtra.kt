package dev.vndx.utils

import org.lwjgl.opengl.ContextCapabilities
import org.lwjgl.opengl.GL11
import org.lwjgl.opengl.GL15
import org.lwjgl.opengl.GLContext
import java.lang.reflect.Field
import java.lang.reflect.Method

/**
 * Object similar to the GL* classes from LWJGL, exposing APIs to work with addresses rather than buffers.
 * Uses reflection internally to call into LWJGL stuff.
 */
object GLExtra {
    private val fieldGlBufferData: Field = ContextCapabilities::class.java.getDeclaredField("glBufferData")
    private val fieldGlTexImage2D: Field = ContextCapabilities::class.java.getDeclaredField("glTexImage2D")
    private val fieldGlTexSubImage2D: Field = ContextCapabilities::class.java.getDeclaredField("glTexSubImage2D")
    private val methodNglBufferData: Method = GL15::class.java.getDeclaredMethod(
        "nglBufferData",
        Int::class.javaPrimitiveType,
        Long::class.javaPrimitiveType,
        Long::class.javaPrimitiveType,
        Int::class.javaPrimitiveType,
        Long::class.javaPrimitiveType
    )
    private val methodNglTexImage2d = GL11::class.java.getDeclaredMethod(
        "nglTexImage2D",
        Int::class.javaPrimitiveType,
        Int::class.javaPrimitiveType,
        Int::class.javaPrimitiveType,
        Int::class.javaPrimitiveType,
        Int::class.javaPrimitiveType,
        Int::class.javaPrimitiveType,
        Int::class.javaPrimitiveType,
        Int::class.javaPrimitiveType,
        Long::class.javaPrimitiveType,
        Long::class.javaPrimitiveType
    )
    private val methodNglTexSubImage2D = GL11::class.java.getDeclaredMethod(
        "nglTexSubImage2D",
        Int::class.javaPrimitiveType,
        Int::class.javaPrimitiveType,
        Int::class.javaPrimitiveType,
        Int::class.javaPrimitiveType,
        Int::class.javaPrimitiveType,
        Int::class.javaPrimitiveType,
        Int::class.javaPrimitiveType,
        Int::class.javaPrimitiveType,
        Long::class.javaPrimitiveType,
        Long::class.javaPrimitiveType
    )

    init {
        fieldGlBufferData.isAccessible = true
        fieldGlTexImage2D.isAccessible = true
        fieldGlTexSubImage2D.isAccessible = true
        methodNglBufferData.isAccessible = true
        methodNglTexImage2d.isAccessible = true
        methodNglTexSubImage2D.isAccessible = true
    }

    fun glBufferData(target: Int, address: Long, size: Long, usage: Int) {
        val caps = GLContext.getCapabilities()
        val ptr = fieldGlBufferData.get(caps) as Long
        methodNglBufferData.invoke(null, target, size, address, usage, ptr)
    }

    fun glTexImage2D(target: Int, level: Int, internalFormat: Int, width: Int, height: Int, border: Int, format: Int, type: Int, address: Long) {
        val caps = GLContext.getCapabilities()
        val ptr = fieldGlTexImage2D.get(caps) as Long
        methodNglTexImage2d.invoke(null, target, level, internalFormat, width, height, border, format, type, address, ptr)
    }

    fun glTexSubImage2D(target: Int, level: Int, xOffset: Int, yOffset: Int, width: Int, height: Int, format: Int, type: Int, address: Long) {
        val caps = GLContext.getCapabilities()
        val ptr = fieldGlTexSubImage2D.get(caps) as Long
        methodNglTexSubImage2D.invoke(null, target, level, xOffset, yOffset, width, height, format, type, address, ptr)
    }
}

