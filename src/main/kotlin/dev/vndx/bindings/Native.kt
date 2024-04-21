package dev.vndx.bindings

import java.nio.ByteBuffer

object RasterKind {
    val Bitmap: Int = 0
    val SDF: Int = 1
    val MSDF: Int = 2
}

class Font(private val addr: Long, val name: String, val data: ByteBuffer, val config: FontConfig) {
    companion object {
        fun load(data: ByteBuffer, config: FontConfig): Array<Font> {
            return loadFonts(data, config)
        }
    }

    private fun destroy() {
        destroyFont(addr)
    }

    protected fun finalize() {
        destroy()
    }
}

class GlobalConfig(val atlasSize: Int, val glyphPadding: Int, val sinAlpha: Double, val coloringSeed: Long) {
    companion object {
        fun get(): GlobalConfig {
            return getGlobalConfig()
        }
    }

    fun use() {
        setGlobalConfig(this)
    }
}

class FontConfig(val rasterKind: Int, val scale: Float, lineHeight: Float)

private external fun loadFonts(data: ByteBuffer, config: FontConfig): Array<Font>
private external fun destroyFont(addr: Long)
private external fun setGlobalConfig(config: GlobalConfig)
private external fun getGlobalConfig(): GlobalConfig
