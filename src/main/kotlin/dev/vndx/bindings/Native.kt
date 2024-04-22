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

    fun processBatched(commands: Array<Command>): BatchedResults {
        return fontProcessBatched(addr, commands)
    }
}

data class GlobalConfig(val atlasSize: Int, val glyphPadding: Int, val sinAlpha: Double, val coloringSeed: Long) {
    companion object {
        fun get(): GlobalConfig {
            return getGlobalConfig()
        }
    }

    fun use() {
        setGlobalConfig(this)
    }
}

data class Command(val string: String, val multiLine: Boolean, val maxLength: Float?)

data class FontConfig(val rasterKind: Int, val scale: Float, val lineHeight: Float, val language: String)

class Mesh(val tex: Int, val vertices_addr: Long, val vertices_len: Long, val indices_addr: Long, val indices_len: Long) {
    var destroyed = false

    fun destroy() {
        if(!destroyed) {
            destroyed = true;
            destroyMesh(vertices_addr, vertices_len, indices_addr, indices_len)
        }
    }

    protected fun finalize() {
        destroy()
    }
}

sealed class Diff {
    data class TextureUpdate(val tex: Int, val channelCount: Int, val damageMinX: Int, val damageMinY: Int, val damageMaxX: Int, val damageMaxY: Int, val size: Int, val data: Long) : Diff()
    data class TextureCreation(val tex: Int, val channelCount: Int, val size: Int, val data: Long) : Diff()
}

data class BatchedResults(val meshes: Array<Array<Mesh>>, val diffs: Array<Diff>)

private external fun loadFonts(data: ByteBuffer, config: FontConfig): Array<Font>
private external fun destroyFont(addr: Long)
private external fun setGlobalConfig(config: GlobalConfig)
private external fun getGlobalConfig(): GlobalConfig
private external fun destroyMesh(vertices_addr: Long, vertices_len: Long, indices_addr: Long, indices_len: Long)
private external fun fontProcessBatched(fontAddr: Long, commands: Array<Command>): BatchedResults
